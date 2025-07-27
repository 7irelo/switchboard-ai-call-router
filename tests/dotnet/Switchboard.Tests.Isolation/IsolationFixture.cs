using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Npgsql;
using Switchboard.Domain;
using Switchboard.Storage;
using Testcontainers.PostgreSql;
using Xunit;
using YamlDotNet.Serialization;

namespace Switchboard.Tests.Isolation;

/// <summary>A real Postgres, real migrations, the real API, two seeded tenants with unique canaries.</summary>
public sealed class IsolationFixture : IAsyncLifetime
{
    public const string AppPassword = "isolation-test-app";
    public const string AdminKey = "isolation-test-admin";
    public const string TenantA = "thornveld_vet";
    public const string TenantB = "quillfibre";
    public const string CanaryA = "CANARY-AAAA-7f3e91";
    public const string CanaryB = "CANARY-BBBB-c2d804";

    private readonly PostgreSqlContainer _pg = new PostgreSqlBuilder().WithImage("postgres:17-alpine").Build();

    public WebApplicationFactory<Program> Factory { get; private set; } = null!;
    public CapturingLoggerProvider Logs { get; } = new();
    public string AdminConnection => _pg.GetConnectionString();
    public string AppConnection { get; private set; } = "";
    public HttpClient ClientA { get; private set; } = null!;
    public HttpClient ClientB { get; private set; } = null!;
    public string RepoRoot { get; } = FindRoot();

    /// <summary>Tenant B's real ids: every cross-tenant probe aims at something that exists.</summary>
    public Dictionary<string, string> BIds { get; } = new()
    {
        ["departmentId"] = "noc",
        ["scheduleId"] = "sales_hours",
        ["ruleId"] = "rule-b",
        ["conversationId"] = "conv-b",
        ["version"] = "1",
    };

    public async Task InitializeAsync()
    {
        await _pg.StartAsync();
        await Migrator.ApplyAsync(AdminConnection, AppPassword);
        AppConnection = new NpgsqlConnectionStringBuilder(AdminConnection) { Username = "switchboard_app", Password = AppPassword, MaxPoolSize = 4 }.ConnectionString;

        Factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
        {
            b.UseSetting("SWITCHBOARD_APP_DB", AppConnection);
            b.UseSetting("SWITCHBOARD_ADMIN_KEY", AdminKey);
            b.UseSetting("COST_MATRIX_PATH", Path.Combine(RepoRoot, "configs/cost_matrix.yaml"));
            b.ConfigureLogging(l => l.AddProvider(Logs).SetMinimumLevel(LogLevel.Trace));
        });

        ClientA = await ProvisionAsync(TenantA, CanaryA);
        ClientB = await ProvisionAsync(TenantB, CanaryB);

        var store = Factory.Services.GetRequiredService<ConfigStore>();
        await store.CreateRuleAsync(TenantB, new RuleRow("rule-b", "27005559999", "noc"));
        await store.CreateRuleAsync(TenantA, new RuleRow("rule-a", "27005558888", "on_call_vet"));
        await SeedConversationAsync(store, TenantA, "conv-a", CanaryA);
        await SeedConversationAsync(store, TenantB, "conv-b", CanaryB);
    }

    private async Task<HttpClient> ProvisionAsync(string tenant, string canary)
    {
        var admin = Factory.CreateClient();
        admin.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", AdminKey);
        var config = LoadTenant(tenant) with { PromptOverride = $"Override {canary}" };
        var response = await admin.PostAsync("/admin/tenants", new StringContent(JsonSerializer.Serialize(config, TenantConfig.Json), System.Text.Encoding.UTF8, "application/json"));
        response.EnsureSuccessStatusCode();
        var body = await response.Content.ReadFromJsonAsync<JsonElement>();
        var client = Factory.CreateClient();
        client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", body.GetProperty("api_key").GetString());
        return client;
    }

    private static Task SeedConversationAsync(ConfigStore store, string tenant, string conversation, string canary) =>
        store.PersistEventAsync(tenant, $"evt-{conversation}", conversation, 1, "voice", "1", "final", JsonSerializer.Serialize(new { text = canary }), DateTimeOffset.UtcNow, $"my account says {canary}", 1);

    public TenantConfig LoadTenant(string id)
    {
        var yaml = new DeserializerBuilder().WithAttemptingUnquotedStringTypeDeserialization().Build()
            .Deserialize(new StringReader(File.ReadAllText(Path.Combine(RepoRoot, "configs/tenants", id + ".yaml"))));
        return JsonSerializer.Deserialize<TenantConfig>(new SerializerBuilder().JsonCompatible().Build().Serialize(yaml!), TenantConfig.Json)!;
    }

    public async Task DisposeAsync()
    {
        await Factory.DisposeAsync();
        await _pg.DisposeAsync();
    }

    private static string FindRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "buf.yaml"))) dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("repo root not found");
    }
}

public sealed class CapturingLoggerProvider : ILoggerProvider
{
    private readonly List<string> _lines = [];

    public IReadOnlyList<string> Lines
    {
        get
        {
            lock (_lines) return [.. _lines];
        }
    }

    public ILogger CreateLogger(string categoryName) => new Capture(this);

    public void Dispose()
    {
    }

    private sealed class Capture(CapturingLoggerProvider owner) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            lock (owner._lines) owner._lines.Add(formatter(state, exception) + " " + exception);
        }
    }
}

[CollectionDefinition("isolation")]
public sealed class IsolationCollection : ICollectionFixture<IsolationFixture>;
