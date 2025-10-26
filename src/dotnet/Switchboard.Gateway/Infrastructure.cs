using Channel = System.Threading.Channels.Channel;
using System.Collections.Concurrent;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using Google.Protobuf;
using Google.Protobuf.WellKnownTypes;
using StackExchange.Redis;
using Switchboard.Contracts.V1;
using Switchboard.Domain;
using Switchboard.Storage;

namespace Switchboard.Gateway;

/// <summary>Versioned in-memory config. The hot path reads only this, and it keeps serving if Postgres goes down.</summary>
public sealed class ConfigCache(TenantDb db, ConfigStore store, ILogger<ConfigCache> log) : BackgroundService
{
    private volatile Snapshot _current = new(new Dictionary<string, (string, TenantConfig)>(), new Dictionary<string, string>());

    private sealed record Snapshot(IReadOnlyDictionary<string, (string Version, TenantConfig Config)> Tenants, IReadOnlyDictionary<string, string> Channels);

    public bool Ready => _current.Tenants.Count > 0;

    public (string Version, TenantConfig Config)? Get(string tenant) => _current.Tenants.TryGetValue(tenant, out var entry) ? entry : null;

    public string? Resolve(string kind, string value) => _current.Channels.GetValueOrDefault($"{kind}:{value}");

    public async Task RefreshAsync(CancellationToken ct)
    {
        var tenants = new Dictionary<string, (string, TenantConfig)>();
        var channels = new Dictionary<string, string>();
        foreach (var id in await db.ListTenantIdsAsync(ct))
        {
            if (await store.GetCurrentAsync(id, ct) is not { } current) continue;

            var config = JsonSerializer.Deserialize<TenantConfig>(current.Json, TenantConfig.Json)!;
            tenants[id] = (current.Version.ToString(System.Globalization.CultureInfo.InvariantCulture), config);
            foreach (var did in config.Channels.Dids) channels[$"did:{did}"] = id;
            foreach (var phone in config.Channels.WhatsappPhoneNumberIds) channels[$"whatsapp:{phone}"] = id;
        }

        _current = new Snapshot(tenants, channels);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await RefreshAsync(stoppingToken);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                log.LogWarning("config refresh failed, serving last-known snapshot: {Error}", e.GetType().Name);
            }

            await Task.Delay(TimeSpan.FromSeconds(3), stoppingToken);
        }
    }
}

/// <summary>Non-blocking event emission. A slow or dead Redis can't cost routing latency: events drop with a counter.</summary>
public sealed class EventPublisher(IConfiguration config, ILogger<EventPublisher> log) : BackgroundService
{
    private readonly Channel<ConversationEvent> _queue = Channel.CreateBounded<ConversationEvent>(new BoundedChannelOptions(10_000) { FullMode = BoundedChannelFullMode.DropWrite });
    private long _dropped;

    public long Dropped => Interlocked.Read(ref _dropped);

    public void Emit(ConversationEvent e)
    {
        if (!_queue.Writer.TryWrite(e)) Interlocked.Increment(ref _dropped);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var connection = await ConnectionMultiplexer.ConnectAsync(new ConfigurationOptions
        {
            EndPoints = { config["REDIS"] ?? "redis:6379" },
            AbortOnConnectFail = false,
            User = config["REDIS_USER"],
            Password = config["REDIS_PASSWORD"],
        });
        var redis = connection.GetDatabase();
        await foreach (var e in _queue.Reader.ReadAllAsync(stoppingToken))
        {
            try
            {
                await redis.StreamAddAsync(TenantKeys.Events(e.TenantId), "event", Convert.ToBase64String(e.ToByteArray()));
            }
            catch (Exception ex) when (ex is RedisException or TimeoutException)
            {
                Interlocked.Increment(ref _dropped);
                log.LogWarning("event dropped: redis unavailable. Total dropped {Dropped}", Dropped);
            }
        }
    }
}

public sealed class ConversationLog(EventPublisher publisher, string tenant, string conversation, string version, Contracts.V1.Channel channel)
{
    private ulong _sequence;

    public void Emit(Action<ConversationEvent> fill)
    {
        var e = new ConversationEvent
        {
            EventId = Guid.CreateVersion7().ToString("N"),
            TenantId = tenant,
            ConversationId = conversation,
            Sequence = ++_sequence,
            OccurredAt = Timestamp.FromDateTimeOffset(DateTimeOffset.UtcNow),
            Channel = channel,
            ConfigVersion = version,
        };
        fill(e);
        publisher.Emit(e);
    }
}

public sealed class ContextSigner(IConfiguration config)
{
    private readonly byte[] _secret = Encoding.UTF8.GetBytes(config["INFERENCE_CONTEXT_SECRET"] ?? throw new InvalidOperationException("INFERENCE_CONTEXT_SECRET is required"));

    public TenantContext Sign(string tenant, string conversation, string version) => new()
    {
        TenantId = tenant,
        ConversationId = conversation,
        ConfigVersion = version,
        Signature = Convert.ToHexStringLower(HMACSHA256.HashData(_secret, Encoding.UTF8.GetBytes($"{tenant}|{conversation}|{version}"))),
    };
}

/// <summary>A thin ARI client: only the calls this gateway uses.</summary>
public sealed class AriClient
{
    private readonly HttpClient _http;

    public AriClient(HttpClient http, IConfiguration config)
    {
        _http = http;
        _http.BaseAddress = new Uri((config["ARI_URL"] ?? "http://asterisk:8088") + "/ari/");
        var auth = $"{config["ARI_USER"] ?? "switchboard"}:{config["ASTERISK_ARI_PASSWORD"]}";
        _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", Convert.ToBase64String(Encoding.UTF8.GetBytes(auth)));
    }

    private async Task<JsonElement> PostAsync(string path, CancellationToken ct)
    {
        using var response = await _http.PostAsync(path, null, ct);
        var body = await response.Content.ReadAsStringAsync(ct);
        if (!response.IsSuccessStatusCode) throw new InvalidOperationException($"ARI {path.Split('?')[0]} returned {(int)response.StatusCode}: {body}");
        return string.IsNullOrEmpty(body) ? default : JsonDocument.Parse(body).RootElement.Clone();
    }

    public Task AnswerAsync(string channel, CancellationToken ct) => PostAsync($"channels/{channel}/answer", ct);

    public async Task<string> CreateBridgeAsync(CancellationToken ct) => (await PostAsync("bridges?type=mixing", ct)).GetProperty("id").GetString()!;

    public Task AddToBridgeAsync(string bridge, string channels, CancellationToken ct) => PostAsync($"bridges/{bridge}/addChannel?channel={Uri.EscapeDataString(channels)}", ct);

    /// <summary>Asterisk dials out to this gateway's /media socket, carrying the call id in the URI.</summary>
    public async Task<string> CreateExternalMediaAsync(string callId, CancellationToken ct)
    {
        var data = Uri.EscapeDataString($"f(json)v(call_id={callId})");
        var result = await PostAsync($"channels/externalMedia?app=switchboard&external_host=media_gateway&transport=websocket&encapsulation=none&format=ulaw&transport_data={data}", ct);
        return result.GetProperty("id").GetString()!;
    }

    public Task ContinueAsync(string channel, string context, string extension, CancellationToken ct) =>
        PostAsync($"channels/{channel}/continue?context={context}&extension={Uri.EscapeDataString(extension)}&priority=1", ct);

    public async Task HangupAsync(string channel)
    {
        using var _ = await _http.DeleteAsync($"channels/{channel}");
    }

    public async Task DeleteBridgeAsync(string bridge)
    {
        using var _ = await _http.DeleteAsync($"bridges/{bridge}");
    }
}

/// <summary>Pre-rendered prompts: no live TTS, which removes a latency stage and an injection risk. Until TTS voices are chosen these are tones, not speech.</summary>
public sealed class PromptStore(IConfiguration config)
{
    private readonly ConcurrentDictionary<string, byte[]> _cache = new();
    private readonly string _root = config["PROMPTS_DIR"] ?? "/prompts";

    public byte[] Get(string tenant, string language, string name) => _cache.GetOrAdd($"{tenant}/{language}/{name}", key =>
    {
        var path = Path.Combine(_root, key + ".ulaw");
        return File.Exists(path) ? File.ReadAllBytes(path) : Tone(name == "greeting" ? 200 : 300);
    });

    private static byte[] Tone(int milliseconds)
    {
        var samples = new byte[milliseconds * 8];
        for (var i = 0; i < samples.Length; i++) samples[i] = LinearToUlaw((short)(4000 * Math.Sin(2 * Math.PI * 440 * i / 8000)));
        return samples;
    }

    private static byte LinearToUlaw(short sample)
    {
        const int Bias = 0x84;
        var sign = sample < 0 ? 0x80 : 0;
        var magnitude = Math.Min(Math.Abs((int)sample), 32635) + Bias;
        var exponent = 7;
        for (var mask = 0x4000; (magnitude & mask) == 0 && exponent > 0; mask >>= 1) exponent--;
        return (byte)~(sign | (exponent << 4) | ((magnitude >> (exponent + 3)) & 0x0F));
    }
}

/// <summary>Webhook signature in the style of X-Hub-Signature-256.</summary>
public static class WebhookSignature
{
    public static bool IsValid(string secret, ReadOnlySpan<byte> body, string? header)
    {
        if (header is null || !header.StartsWith("sha256=", StringComparison.Ordinal)) return false;

        var expected = Encoding.UTF8.GetBytes("sha256=" + Convert.ToHexStringLower(HMACSHA256.HashData(Encoding.UTF8.GetBytes(secret), body)));
        return CryptographicOperations.FixedTimeEquals(expected, Encoding.UTF8.GetBytes(header));
    }
}

/// <summary>Caps concurrent conversations per tenant, so one tenant can't exhaust capacity for the others.</summary>
public sealed class TenantLimiter(IConfiguration config)
{
    private readonly int _limit = int.TryParse(config["TENANT_MAX_CONCURRENT"], out var n) ? n : 50;
    private readonly ConcurrentDictionary<string, SemaphoreSlim> _gates = new();

    public bool TryEnter(string tenant) => _gates.GetOrAdd(tenant, _ => new SemaphoreSlim(_limit, _limit)).Wait(0);

    public void Exit(string tenant) => _gates[tenant].Release();
}
