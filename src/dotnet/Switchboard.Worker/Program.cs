using Google.Protobuf;
using Npgsql;
using StackExchange.Redis;
using Switchboard.Contracts.V1;
using Switchboard.Storage;

var builder = Host.CreateApplicationBuilder(args);
builder.Logging.AddJsonConsole();
var appDb = builder.Configuration["SWITCHBOARD_APP_DB"] ?? throw new InvalidOperationException("SWITCHBOARD_APP_DB is required");
builder.Services.AddSingleton(_ => NpgsqlDataSource.Create(appDb));
builder.Services.AddSingleton<TenantDb>();
builder.Services.AddSingleton<ConfigStore>();
builder.Services.AddHostedService<EventPersister>();
builder.Services.AddHostedService<RetentionJob>();
builder.Build().Run();

/// <summary>Removes old transcripts one tenant at a time, each inside its own scope.</summary>
public sealed class RetentionJob(IConfiguration config, TenantDb db, ConfigStore store, ILogger<RetentionJob> log) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var days = int.TryParse(config["TRANSCRIPT_RETENTION_DAYS"], out var n) ? n : 30;
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                foreach (var tenant in await db.ListTenantIdsAsync(stoppingToken))
                {
                    var removed = await store.DeleteTranscriptsOlderThanAsync(tenant, DateTimeOffset.UtcNow.AddDays(-days), stoppingToken);
                    if (removed > 0) log.LogInformation("retention tenant={Tenant} removed={Removed}", tenant, removed);
                }
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                log.LogWarning("retention retrying: {Error}", e.GetType().Name);
            }

            await Task.Delay(TimeSpan.FromHours(1), stoppingToken);
        }
    }
}

/// <summary>Moves events from each tenant's stream into Postgres, inside that tenant's scope. Idempotent on event id.</summary>
public sealed class EventPersister(IConfiguration config, TenantDb db, ConfigStore store, ILogger<EventPersister> log) : BackgroundService
{
    private const string Group = "persister";
    private const string Consumer = "worker-1";

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var connection = await ConnectionMultiplexer.ConnectAsync(new ConfigurationOptions { EndPoints = { config["REDIS"] ?? "redis:6379" }, AbortOnConnectFail = false, User = config["REDIS_USER"], Password = config["REDIS_PASSWORD"] });
        var redis = connection.GetDatabase();
        var known = new HashSet<string>();
        while (!stoppingToken.IsCancellationRequested)
        {
            var moved = 0;
            try
            {
                foreach (var tenant in await db.ListTenantIdsAsync(stoppingToken))
                {
                    var key = TenantKeys.Events(tenant);
                    if (known.Add(tenant))
                    {
                        try
                        {
                            await redis.StreamCreateConsumerGroupAsync(key, Group, "0-0", createStream: true);
                        }
                        catch (RedisServerException e) when (e.Message.Contains("BUSYGROUP", StringComparison.Ordinal))
                        {
                        }
                    }

                    foreach (var entry in await redis.StreamReadGroupAsync(key, Group, Consumer, ">", 100))
                    {
                        await PersistAsync(tenant, entry, stoppingToken);
                        await redis.StreamAcknowledgeAsync(key, Group, entry.Id);
                        moved++;
                    }
                }
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                log.LogWarning("persister retrying: {Error}", e.GetType().Name);
                known.Clear();
            }

            if (moved == 0) await Task.Delay(TimeSpan.FromMilliseconds(250), stoppingToken);
        }
    }

    private async Task PersistAsync(string tenant, StreamEntry entry, CancellationToken ct)
    {
        var e = ConversationEvent.Parser.ParseFrom(Convert.FromBase64String(entry["event"]!));
        if (e.TenantId != tenant)
        {
            // An event in the wrong tenant's stream is a bug. It's dropped, not written anywhere.
            log.LogError("tenant mismatch on stream {Stream}: event dropped", tenant);
            return;
        }

        if (e.PayloadCase == ConversationEvent.PayloadOneofCase.Partial) return; // Partials aren't persisted by default.

        string? transcript = e.PayloadCase switch
        {
            ConversationEvent.PayloadOneofCase.Final => e.Final.Text,
            ConversationEvent.PayloadOneofCase.Message => e.Message.Text,
            _ => null,
        };
        await store.PersistEventAsync(tenant, e.EventId, e.ConversationId, (long)e.Sequence, e.Channel.ToString(), e.ConfigVersion, e.PayloadCase.ToString(),
            JsonFormatter.Default.Format(e), e.OccurredAt.ToDateTimeOffset(), transcript, (int)e.Sequence, ct);
    }
}
