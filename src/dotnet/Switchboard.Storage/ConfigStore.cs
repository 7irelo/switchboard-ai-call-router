using System.Collections.Immutable;
using System.Data.Common;
using System.Text.Json;
using Dapper;
using Switchboard.Domain;

namespace Switchboard.Storage;

public sealed record DepartmentRow(string Id, string Target, string Hours, string ClosedAction, bool EmergencyCapable);

public sealed record RuleRow(string Id, string Did, string DepartmentId);

public sealed record ConversationRow(string Id, string Channel, string ConfigVersion, DateTime StartedAt);

public sealed record EventRow(string EventId, string ConversationId, long Sequence, string Type, string Payload, DateTime OccurredAt);

public sealed record TranscriptRow(string ConversationId, int Turn, string Text);

public sealed record SettingsDocument
{
    public ImmutableArray<string> Holidays { get; init; } = [];
    public ImmutableArray<string> Languages { get; init; } = [];
    public ImmutableArray<CustomIntent> CustomIntents { get; init; } = [];
    public required Fallback Fallback { get; init; }
    public ImmutableDictionary<string, double> CostOverrides { get; init; } = ImmutableDictionary<string, double>.Empty;
    public ChannelBindings Channels { get; init; } = new();
}

/// <summary>Draft configuration tables and immutable published snapshots. All access is tenant-scoped.</summary>
public sealed class ConfigStore(TenantDb db)
{
    static ConfigStore() => DefaultTypeMap.MatchNamesWithUnderscores = true;

    /// <summary>Provisions by entering the new tenant's own scope: no RLS bypass is involved.</summary>
    public Task ProvisionAsync(TenantConfig c, string keyHash, CancellationToken ct = default) =>
        db.InTenantAsync(c.TenantId, async (conn, tx) =>
        {
            await conn.ExecuteAsync("INSERT INTO tenants (tenant_id, name) VALUES (@TenantId, @Name)", c, tx);
            await conn.ExecuteAsync("INSERT INTO api_credentials (tenant_id, key_hash) VALUES (@tenant, @keyHash)", new { tenant = c.TenantId, keyHash }, tx);
            await WriteDraftAsync(conn, tx, c);
        }, ct);

    private static async Task WriteDraftAsync(DbConnection conn, DbTransaction tx, TenantConfig c)
    {
        var doc = new SettingsDocument
        {
            Holidays = c.Holidays,
            Languages = c.Languages,
            CustomIntents = c.CustomIntents,
            Fallback = c.Fallback,
            CostOverrides = c.CostOverrides,
            Channels = c.Channels,
        };
        await conn.ExecuteAsync(
            "INSERT INTO tenant_settings (tenant_id, timezone, prompt_override, document) VALUES (@t, @tz, @p, @d::jsonb)",
            new { t = c.TenantId, tz = c.Timezone, p = c.PromptOverride, d = JsonSerializer.Serialize(doc, TenantConfig.Json) }, tx);
        foreach (var (id, windows) in c.Schedules)
        {
            await conn.ExecuteAsync(
                "INSERT INTO schedules (tenant_id, id, windows) VALUES (@t, @id, @w::jsonb)",
                new { t = c.TenantId, id, w = JsonSerializer.Serialize(windows, TenantConfig.Json) }, tx);
        }

        foreach (var d in c.Departments)
        {
            await conn.ExecuteAsync(
                "INSERT INTO departments (tenant_id, id, target, hours, closed_action, emergency_capable) VALUES (@t, @Id, @Target, @Hours, @ClosedAction, @EmergencyCapable)",
                new { t = c.TenantId, d.Id, d.Target, d.Hours, d.ClosedAction, d.EmergencyCapable }, tx);
        }

        foreach (var (intent, dept) in c.IntentMap)
        {
            await conn.ExecuteAsync("INSERT INTO intent_map (tenant_id, intent, department_id) VALUES (@t, @intent, @dept)", new { t = c.TenantId, intent, dept }, tx);
        }

        var n = 0;
        foreach (var r in c.Rules)
        {
            await conn.ExecuteAsync(
                "INSERT INTO routing_rules (tenant_id, id, did, department_id) VALUES (@t, @id, @did, @dept)",
                new { t = c.TenantId, id = $"rule-{++n}", did = r.When.Did, dept = r.Action.Route }, tx);
        }
    }

    public Task<IReadOnlyList<DepartmentRow>> ListDepartmentsAsync(string tenant, CancellationToken ct = default) =>
        db.InTenantAsync<IReadOnlyList<DepartmentRow>>(tenant, async (c, t) =>
            [.. await c.QueryAsync<DepartmentRow>("SELECT id, target, hours, closed_action, emergency_capable FROM departments ORDER BY id", transaction: t)], ct);

    public Task<DepartmentRow?> GetDepartmentAsync(string tenant, string id, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) =>
            c.QuerySingleOrDefaultAsync<DepartmentRow?>("SELECT id, target, hours, closed_action, emergency_capable FROM departments WHERE id = @id", new { id }, t), ct);

    public Task CreateDepartmentAsync(string tenant, DepartmentRow d, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteAsync(
            "INSERT INTO departments (tenant_id, id, target, hours, closed_action, emergency_capable) VALUES (@tenant, @Id, @Target, @Hours, @ClosedAction, @EmergencyCapable)",
            new { tenant, d.Id, d.Target, d.Hours, d.ClosedAction, d.EmergencyCapable }, t), ct);

    public Task<int> UpdateDepartmentAsync(string tenant, DepartmentRow d, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteAsync(
            "UPDATE departments SET target = @Target, hours = @Hours, closed_action = @ClosedAction, emergency_capable = @EmergencyCapable WHERE id = @Id",
            new { d.Id, d.Target, d.Hours, d.ClosedAction, d.EmergencyCapable }, t), ct);

    public Task<int> DeleteDepartmentAsync(string tenant, string id, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteAsync("DELETE FROM departments WHERE id = @id", new { id }, t), ct);

    public Task<string?> GetScheduleAsync(string tenant, string id, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteScalarAsync<string?>("SELECT windows::text FROM schedules WHERE id = @id", new { id }, t), ct);

    public Task<IReadOnlyList<string>> ListScheduleIdsAsync(string tenant, CancellationToken ct = default) =>
        db.InTenantAsync<IReadOnlyList<string>>(tenant, async (c, t) => [.. await c.QueryAsync<string>("SELECT id FROM schedules ORDER BY id", transaction: t)], ct);

    public Task<int> UpdateScheduleAsync(string tenant, string id, string windowsJson, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteAsync("UPDATE schedules SET windows = @windowsJson::jsonb WHERE id = @id", new { id, windowsJson }, t), ct);

    public Task<Dictionary<string, string>> GetIntentMapAsync(string tenant, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, async (c, t) =>
            (await c.QueryAsync<(string Intent, string DepartmentId)>("SELECT intent, department_id FROM intent_map", transaction: t)).ToDictionary(x => x.Intent, x => x.DepartmentId), ct);

    /// <summary>Replaces the map. A department in another tenant fails the composite foreign key.</summary>
    public Task SetIntentMapAsync(string tenant, IReadOnlyDictionary<string, string> map, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, async (c, t) =>
        {
            await c.ExecuteAsync("DELETE FROM intent_map", transaction: t);
            foreach (var (intent, dept) in map)
            {
                await c.ExecuteAsync("INSERT INTO intent_map (tenant_id, intent, department_id) VALUES (@tenant, @intent, @dept)", new { tenant, intent, dept }, t);
            }
        }, ct);

    public Task<IReadOnlyList<RuleRow>> ListRulesAsync(string tenant, CancellationToken ct = default) =>
        db.InTenantAsync<IReadOnlyList<RuleRow>>(tenant, async (c, t) => [.. await c.QueryAsync<RuleRow>("SELECT id, did, department_id FROM routing_rules ORDER BY id", transaction: t)], ct);

    public Task<RuleRow?> GetRuleAsync(string tenant, string id, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.QuerySingleOrDefaultAsync<RuleRow?>("SELECT id, did, department_id FROM routing_rules WHERE id = @id", new { id }, t), ct);

    public Task CreateRuleAsync(string tenant, RuleRow r, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteAsync(
            "INSERT INTO routing_rules (tenant_id, id, did, department_id) VALUES (@tenant, @Id, @Did, @DepartmentId)", new { tenant, r.Id, r.Did, r.DepartmentId }, t), ct);

    public Task<int> DeleteRuleAsync(string tenant, string id, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteAsync("DELETE FROM routing_rules WHERE id = @id", new { id }, t), ct);

    public Task<string?> GetPromptOverrideAsync(string tenant, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteScalarAsync<string?>("SELECT prompt_override FROM tenant_settings", transaction: t), ct);

    public Task<int> SetPromptOverrideAsync(string tenant, string text, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteAsync("UPDATE tenant_settings SET prompt_override = @text", new { text }, t), ct);

    /// <summary>Reads the draft tables as a config. Does not publish.</summary>
    public async Task<TenantConfig> BuildDraftAsync(string tenant, CancellationToken ct = default)
    {
        return await db.InTenantAsync(tenant, async (c, t) => await BuildAsync(c, t, tenant), ct);
    }

    private static async Task<TenantConfig> BuildAsync(DbConnection c, DbTransaction t, string tenant)
    {
        var name = await c.ExecuteScalarAsync<string>("SELECT name FROM tenants", transaction: t) ?? "";
        var settings = await c.QuerySingleAsync<(string Timezone, string PromptOverride, string Document)>(
            "SELECT timezone, prompt_override, document::text FROM tenant_settings", transaction: t);
        var doc = JsonSerializer.Deserialize<SettingsDocument>(settings.Document, TenantConfig.Json)!;
        var schedules = (await c.QueryAsync<(string Id, string Windows)>("SELECT id, windows::text FROM schedules", transaction: t))
            .ToImmutableDictionary(x => x.Id, x => JsonSerializer.Deserialize<ImmutableDictionary<string, ImmutableArray<ImmutableArray<string>>>>(x.Windows, TenantConfig.Json)!);
        var departments = (await c.QueryAsync<DepartmentRow>("SELECT id, target, hours, closed_action, emergency_capable FROM departments ORDER BY id", transaction: t))
            .Select(d => new Department { Id = d.Id, Target = d.Target, Hours = d.Hours, ClosedAction = d.ClosedAction, EmergencyCapable = d.EmergencyCapable })
            .ToImmutableArray();
        var map = (await c.QueryAsync<(string Intent, string DepartmentId)>("SELECT intent, department_id FROM intent_map", transaction: t))
            .ToImmutableDictionary(x => x.Intent, x => x.DepartmentId);
        var rules = (await c.QueryAsync<RuleRow>("SELECT id, did, department_id FROM routing_rules ORDER BY id", transaction: t))
            .Select(r => new RoutingRule { When = new RuleWhen { Did = r.Did }, Action = new RuleAction { Route = r.DepartmentId } })
            .ToImmutableArray();
        return new TenantConfig
        {
            TenantId = tenant,
            Name = name,
            Timezone = settings.Timezone,
            PromptOverride = settings.PromptOverride,
            Holidays = doc.Holidays,
            Languages = doc.Languages,
            CustomIntents = doc.CustomIntents,
            Fallback = doc.Fallback,
            CostOverrides = doc.CostOverrides,
            Channels = doc.Channels,
            Schedules = schedules,
            Departments = departments,
            IntentMap = map,
            Rules = rules,
        };
    }

    /// <summary>Validates the draft and publishes it as the next immutable version. Returns errors, or the version.</summary>
    public Task<(int Version, IReadOnlyList<string> Errors)> PublishAsync(string tenant, CostMatrix matrix, CancellationToken ct = default) =>
        db.InTenantAsync<(int, IReadOnlyList<string>)>(tenant, async (c, t) =>
        {
            var config = await BuildAsync(c, t, tenant);
            var errors = config.Validate(matrix);
            if (errors.Count > 0) return (0, errors);

            var version = await c.ExecuteScalarAsync<int>("SELECT COALESCE(MAX(version), 0) + 1 FROM config_versions", transaction: t);
            await c.ExecuteAsync(
                "INSERT INTO config_versions (tenant_id, version, config) VALUES (@tenant, @version, @json::jsonb)",
                new { tenant, version, json = JsonSerializer.Serialize(config, TenantConfig.Json) }, t);
            await c.ExecuteAsync("UPDATE tenants SET current_version = @version", new { version }, t);
            await c.ExecuteAsync("DELETE FROM channel_bindings", transaction: t);
            foreach (var did in config.Channels.Dids)
            {
                await c.ExecuteAsync("INSERT INTO channel_bindings (tenant_id, kind, value) VALUES (@tenant, 'did', @did)", new { tenant, did }, t);
            }

            foreach (var id in config.Channels.WhatsappPhoneNumberIds)
            {
                await c.ExecuteAsync("INSERT INTO channel_bindings (tenant_id, kind, value) VALUES (@tenant, 'whatsapp', @id)", new { tenant, id }, t);
            }

            return (version, []);
        }, ct);

    public Task<string?> GetVersionAsync(string tenant, int version, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteScalarAsync<string?>("SELECT config::text FROM config_versions WHERE version = @version", new { version }, t), ct);

    public Task<(int Version, string Json)?> GetCurrentAsync(string tenant, CancellationToken ct = default) =>
        db.InTenantAsync<(int Version, string Json)?>(tenant, async (c, t) =>
        {
            var row = await c.QuerySingleOrDefaultAsync<(int Version, string Json)?>(
                "SELECT v.version, v.config::text FROM config_versions v JOIN tenants x ON x.current_version = v.version AND x.tenant_id = v.tenant_id", transaction: t);
            return row;
        }, ct);

    public Task<IReadOnlyList<ConversationRow>> ListConversationsAsync(string tenant, CancellationToken ct = default) =>
        db.InTenantAsync<IReadOnlyList<ConversationRow>>(tenant, async (c, t) =>
            [.. await c.QueryAsync<ConversationRow>("SELECT id, channel, config_version, started_at FROM conversations ORDER BY started_at DESC LIMIT 200", transaction: t)], ct);

    public Task<ConversationRow?> GetConversationAsync(string tenant, string id, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.QuerySingleOrDefaultAsync<ConversationRow?>("SELECT id, channel, config_version, started_at FROM conversations WHERE id = @id", new { id }, t), ct);

    public Task<IReadOnlyList<EventRow>> ListEventsAsync(string tenant, string conversationId, CancellationToken ct = default) =>
        db.InTenantAsync<IReadOnlyList<EventRow>>(tenant, async (c, t) =>
            [.. await c.QueryAsync<EventRow>("SELECT event_id, conversation_id, sequence, type, payload::text AS payload, occurred_at FROM conversation_events WHERE conversation_id = @conversationId ORDER BY sequence", new { conversationId }, t)], ct);

    public Task<IReadOnlyList<TranscriptRow>> ListTranscriptsAsync(string tenant, string conversationId, CancellationToken ct = default) =>
        db.InTenantAsync<IReadOnlyList<TranscriptRow>>(tenant, async (c, t) =>
            [.. await c.QueryAsync<TranscriptRow>("SELECT conversation_id, turn, text FROM transcripts WHERE conversation_id = @conversationId ORDER BY turn", new { conversationId }, t)], ct);

    public Task<int> DeleteTranscriptsAsync(string tenant, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, (c, t) => c.ExecuteAsync("DELETE FROM transcripts", transaction: t), ct);

    /// <summary>Persists an event and its conversation. Idempotent on event id, since delivery is at-least-once.</summary>
    public Task PersistEventAsync(string tenant, string eventId, string conversationId, long sequence, string channel, string configVersion, string type, string payloadJson, DateTimeOffset occurredAt, string? transcriptText, int turn, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, async (c, t) =>
        {
            await c.ExecuteAsync(
                "INSERT INTO conversations (tenant_id, id, channel, config_version) VALUES (@tenant, @conversationId, @channel, @configVersion) ON CONFLICT DO NOTHING",
                new { tenant, conversationId, channel, configVersion }, t);
            await c.ExecuteAsync(
                "INSERT INTO conversation_events (tenant_id, event_id, conversation_id, sequence, type, payload, occurred_at) VALUES (@tenant, @eventId, @conversationId, @sequence, @type, @payloadJson::jsonb, @occurredAt) ON CONFLICT DO NOTHING",
                new { tenant, eventId, conversationId, sequence, type, payloadJson, occurredAt }, t);
            if (transcriptText is not null)
            {
                await c.ExecuteAsync(
                    "INSERT INTO transcripts (tenant_id, conversation_id, turn, text) VALUES (@tenant, @conversationId, @turn, @transcriptText) ON CONFLICT DO NOTHING",
                    new { tenant, conversationId, turn, transcriptText }, t);
            }
        }, ct);

    /// <summary>First delivery of a provider message returns true; a duplicate returns false.</summary>
    public Task<bool> TryRecordMessageAsync(string tenant, string providerMessageId, CancellationToken ct = default) =>
        db.InTenantAsync(tenant, async (c, t) => await c.ExecuteAsync(
            "INSERT INTO message_dedup (tenant_id, provider_message_id) VALUES (@tenant, @providerMessageId) ON CONFLICT DO NOTHING",
            new { tenant, providerMessageId }, t) == 1, ct);
}
