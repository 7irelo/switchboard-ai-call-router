using System.Net;
using System.Net.Http.Json;
using System.Text;
using Dapper;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using Switchboard.Storage;
using Xunit;
using static Switchboard.Tests.Isolation.IsolationFixture;

namespace Switchboard.Tests.Isolation;

[Collection("isolation")]
public class RouteMatrixTests(IsolationFixture f)
{
    /// <summary>
    /// Discovers every route. Tenant A's credential aimed at tenant B's real ids must get 404, not 403,
    /// so existence doesn't leak. A collection route must contain none of B's data. A new route with
    /// an unknown parameter fails the build until it's covered here.
    /// </summary>
    [Fact]
    public async Task Tenant_A_can_never_reach_tenant_B_through_any_route()
    {
        var endpoints = f.Factory.Services.GetRequiredService<EndpointDataSource>().Endpoints.OfType<RouteEndpoint>()
            .Where(e => e.RoutePattern.RawText!.StartsWith("/v1", StringComparison.Ordinal)).ToList();
        Assert.True(endpoints.Count >= 20, $"route discovery found only {endpoints.Count}");

        var probed = 0;
        foreach (var endpoint in endpoints)
        {
            var pattern = endpoint.RoutePattern.RawText!;
            var method = endpoint.Metadata.GetMetadata<IHttpMethodMetadata>()!.HttpMethods.Single();
            var hasId = endpoint.RoutePattern.Parameters.Count > 0;
            var path = pattern;
            foreach (var parameter in endpoint.RoutePattern.Parameters)
            {
                Assert.True(f.BIds.TryGetValue(parameter.Name, out var id), $"route {pattern} has parameter {parameter.Name} with no cross-tenant coverage");
                path = System.Text.RegularExpressions.Regex.Replace(path, "\\{" + parameter.Name + "[^}]*\\}", id!);
            }

            if (pattern == "/v1/transcripts" && method == "DELETE") continue; // Covered with its own test below.

            var request = new HttpRequestMessage(new HttpMethod(method), path);
            if (method is "PUT" or "POST") request.Content = new StringContent(BodyFor(pattern), Encoding.UTF8, "application/json");

            // A publish or a write with no id acts on A's own data: B must be unaffected, and nothing of B's appears.
            var response = await f.ClientA.SendAsync(request);
            var body = await response.Content.ReadAsStringAsync();
            Assert.DoesNotContain(CanaryB, body);
            if (hasId && pattern != "/v1/config/versions/{version:int}")
            {
                Assert.True(response.StatusCode == HttpStatusCode.NotFound, $"{method} {path} as tenant A returned {(int)response.StatusCode}: {body}");
            }

            probed++;
        }

        Assert.True(probed >= 20);

        // B's department survived A's DELETE attempt.
        Assert.Equal(HttpStatusCode.OK, (await f.ClientB.GetAsync("/v1/departments/noc")).StatusCode);
    }

    private static string BodyFor(string pattern) => pattern switch
    {
        "/v1/intent-map" => "{}",
        "/v1/prompt-override" => "{\"text\":\"from a\"}",
        "/v1/departments" => "{\"id\":\"probe\",\"target\":\"x\",\"hours\":\"always\",\"closed_action\":\"voicemail\",\"emergency_capable\":false}",
        "/v1/departments/{departmentId}" => "{\"id\":\"noc\",\"target\":\"hacked\",\"hours\":\"always\",\"closed_action\":\"voicemail\",\"emergency_capable\":true}",
        "/v1/rules" => "{\"id\":\"probe\",\"did\":\"1\",\"department_id\":\"noc\"}",
        "/v1/schedules/{scheduleId}" => "{}",
        _ => "{}",
    };

    [Fact]
    public async Task A_rule_or_map_cannot_point_at_another_tenants_department()
    {
        var rule = await f.ClientA.PostAsJsonAsync("/v1/rules", new { id = "x", did = "27005550000", department_id = "noc" });
        Assert.Equal(HttpStatusCode.BadRequest, rule.StatusCode);
    }

    [Fact]
    public async Task Missing_or_wrong_credential_is_rejected()
    {
        var client = f.Factory.CreateClient();
        Assert.Equal(HttpStatusCode.Unauthorized, (await client.GetAsync("/v1/departments")).StatusCode);
        client.DefaultRequestHeaders.Authorization = new("Bearer", "nope");
        Assert.Equal(HttpStatusCode.Unauthorized, (await client.GetAsync("/v1/departments")).StatusCode);
    }

    [Fact]
    public async Task Tenant_header_cannot_override_the_credential()
    {
        var request = new HttpRequestMessage(HttpMethod.Get, "/v1/conversations/conv-b");
        request.Headers.Add("X-Tenant-Id", TenantB);
        Assert.Equal(HttpStatusCode.NotFound, (await f.ClientA.SendAsync(request)).StatusCode);
    }

    [Fact]
    public async Task Delete_by_tenant_removes_only_that_tenants_transcripts()
    {
        var store = f.Factory.Services.GetRequiredService<ConfigStore>();
        await store.PersistEventAsync("thornveld_vet", "evt-del", "conv-del", 1, "voice", "1", "final", "{}", DateTimeOffset.UtcNow, "to delete", 7);
        await f.ClientA.DeleteAsync("/v1/transcripts");
        Assert.Empty(await store.ListTranscriptsAsync(TenantA, "conv-del"));
        Assert.NotEmpty(await store.ListTranscriptsAsync(TenantB, "conv-b"));
        await store.PersistEventAsync(TenantA, "evt-conv-a2", "conv-a", 2, "voice", "1", "final", "{}", DateTimeOffset.UtcNow, $"my account says {CanaryA}", 1);
    }

    [Fact]
    public async Task Retention_removes_only_that_tenants_old_transcripts()
    {
        var store = f.Factory.Services.GetRequiredService<ConfigStore>();
        await store.PersistEventAsync(TenantB, "evt-keep", "conv-keep", 1, "voice", "1", "final", "{}", DateTimeOffset.UtcNow, "stays", 1);
        await store.DeleteTranscriptsOlderThanAsync(TenantA, DateTimeOffset.UtcNow.AddDays(1));
        Assert.NotEmpty(await store.ListTranscriptsAsync(TenantB, "conv-keep"));
        await store.PersistEventAsync(TenantA, "evt-conv-a3", "conv-a", 3, "voice", "1", "final", "{}", DateTimeOffset.UtcNow, $"my account says {CanaryA}", 1);
    }

    [Fact]
    public async Task Logs_contain_no_transcript_text()
    {
        await f.ClientB.GetAsync("/v1/conversations/conv-b/transcript");
        await f.ClientA.GetAsync("/v1/conversations/conv-b/transcript");
        Assert.DoesNotContain(f.Logs.Lines, l => l.Contains("my account says", StringComparison.Ordinal));
        Assert.DoesNotContain(f.Logs.Lines, l => l.Contains(CanaryA, StringComparison.Ordinal) || l.Contains(CanaryB, StringComparison.Ordinal));
    }
}

[Collection("isolation")]
public class DatabaseTests(IsolationFixture f)
{
    /// <summary>Tables that are deliberately not tenant-scoped. Anything else unclassified fails.</summary>
    private static readonly HashSet<string> Global = ["schema_migrations"];

    [Fact]
    public async Task Every_table_has_tenant_id_forced_rls_and_a_checked_policy()
    {
        await using var c = new NpgsqlConnection(f.AdminConnection);
        var tables = (await c.QueryAsync<(string Name, bool Enabled, bool Forced)>(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relkind = 'r'")).ToList();
        Assert.True(tables.Count >= 13);
        foreach (var (name, enabled, forced) in tables)
        {
            if (Global.Contains(name)) continue;

            var hasTenant = await c.ExecuteScalarAsync<bool>("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = @name AND column_name = 'tenant_id')", new { name });
            Assert.True(hasTenant, $"{name} is unclassified: no tenant_id and not listed as global");
            Assert.True(enabled, $"{name} does not enable RLS");
            Assert.True(forced, $"{name} does not force RLS");
            var policy = await c.QuerySingleOrDefaultAsync<(string Qual, string Check)?>("SELECT qual, with_check FROM pg_policies WHERE tablename = @name", new { name });
            Assert.True(policy is not null, $"{name} has no policy");
            Assert.Contains("app.tenant_id", policy!.Value.Qual);
            Assert.Contains("app.tenant_id", policy.Value.Check);
        }
    }

    [Fact]
    public async Task App_role_is_not_privileged()
    {
        await using var c = new NpgsqlConnection(f.AppConnection);
        var (super, bypass) = await c.QuerySingleAsync<(bool, bool)>("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user");
        Assert.False(super);
        Assert.False(bypass);
        var owned = await c.ExecuteScalarAsync<int>("SELECT count(*) FROM pg_tables WHERE schemaname = 'public' AND tableowner = current_user");
        Assert.Equal(0, owned);
    }

    [Fact]
    public async Task Composite_foreign_keys_protect_every_department_reference()
    {
        await using var c = new NpgsqlConnection(f.AdminConnection);
        foreach (var table in new[] { "intent_map", "routing_rules" })
        {
            var cols = await c.ExecuteScalarAsync<int>(
                "SELECT cardinality(conkey) FROM pg_constraint WHERE contype = 'f' AND conrelid = @table::regclass AND confrelid = 'departments'::regclass", new { table });
            Assert.Equal(2, cols);
        }
    }

    [Fact]
    public async Task Raw_app_role_with_no_tenant_sees_zero_rows()
    {
        await using var c = new NpgsqlConnection(f.AppConnection);
        await c.OpenAsync();
        foreach (var table in new[] { "tenants", "departments", "transcripts", "conversation_events", "config_versions", "api_credentials", "channel_bindings" })
        {
            Assert.Equal(0, await c.ExecuteScalarAsync<int>($"SELECT count(*) FROM {table}"));
        }
    }

    [Fact]
    public async Task With_tenant_A_set_writes_to_B_fail_and_B_is_invisible()
    {
        await using var c = new NpgsqlConnection(f.AppConnection);
        await c.OpenAsync();
        await using var tx = await c.BeginTransactionAsync();
        await c.ExecuteAsync("SELECT set_config('app.tenant_id', @t, true)", new { t = TenantA }, tx);
        Assert.Equal(1, await c.ExecuteScalarAsync<int>("SELECT count(DISTINCT tenant_id) FROM departments", transaction: tx));
        Assert.Equal(0, await c.ExecuteScalarAsync<int>("SELECT count(*) FROM departments WHERE tenant_id = @b", new { b = TenantB }, tx));
        Assert.Equal(0, await c.ExecuteAsync("UPDATE departments SET target = 'x' WHERE tenant_id = @b", new { b = TenantB }, tx));
        var ex = await Assert.ThrowsAsync<PostgresException>(() =>
            c.ExecuteAsync("INSERT INTO schedules (tenant_id, id, windows) VALUES (@b, 'evil', '{}')", new { b = TenantB }, tx));
        Assert.Equal(PostgresErrorCodes.InsufficientPrivilege, ex.SqlState);
    }

    /// <summary>A pool of 1 forces reuse: a tenant set on one unit of work must not exist on the next.</summary>
    [Fact]
    public async Task No_tenant_state_survives_on_a_pooled_connection()
    {
        // NoResetOnClose removes Npgsql's own cleanup, which otherwise hides a session-level SET.
        // Only SET LOCAL can pass: found by the mutation check, see docs/failure-log.md.
        var single = new NpgsqlConnectionStringBuilder(f.AppConnection) { MaxPoolSize = 1, NoResetOnClose = true }.ConnectionString;
        await using var source = NpgsqlDataSource.Create(single);
        var db = new TenantDb(source);
        await db.InTenantAsync(TenantA, (c, t) => c.ExecuteAsync("SELECT 1", transaction: t));
        await using var conn = await source.OpenConnectionAsync();
        var leftover = await conn.ExecuteScalarAsync<string?>("SELECT current_setting('app.tenant_id', true)");
        Assert.True(string.IsNullOrEmpty(leftover), $"tenant state survived on the pooled connection: {leftover}");
        Assert.Equal(0, await conn.ExecuteScalarAsync<int>("SELECT count(*) FROM departments"));
    }

    [Fact]
    public async Task Concurrent_tenants_on_a_small_pool_never_cross()
    {
        var db = f.Factory.Services.GetRequiredService<TenantDb>();
        var tasks = Enumerable.Range(0, 200).Select(async i =>
        {
            var tenant = i % 2 == 0 ? TenantA : TenantB;
            var seen = await db.InTenantAsync(tenant, async (c, t) =>
            {
                await Task.Yield();
                return (await c.QueryAsync<string>("SELECT DISTINCT tenant_id FROM departments", transaction: t)).ToList();
            });
            Assert.Equal([tenant], seen);
        });
        await Task.WhenAll(tasks);
    }

    [Fact]
    public async Task Empty_tenant_fails_closed()
    {
        var db = f.Factory.Services.GetRequiredService<TenantDb>();
        await Assert.ThrowsAsync<ArgumentException>(() => db.InTenantAsync("", (_, _) => Task.CompletedTask));
    }

    [Fact]
    public async Task Channel_resolution_returns_the_right_tenant_and_nothing_for_unknown()
    {
        var db = f.Factory.Services.GetRequiredService<TenantDb>();
        Assert.Equal(TenantA, await db.ResolveChannelAsync("did", "27005550101"));
        Assert.Equal(TenantB, await db.ResolveChannelAsync("whatsapp", "wa-quillfibre-01"));
        Assert.Null(await db.ResolveChannelAsync("did", "000"));
    }
}
