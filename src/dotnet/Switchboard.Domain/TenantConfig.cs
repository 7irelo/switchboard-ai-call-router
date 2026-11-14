using System.Collections.Immutable;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Switchboard.Domain;

public static class Intents
{
    public const string Emergency = "emergency";

    public static readonly ImmutableArray<string> Canonical =
    [
        "billing", "payments_arrears", "sales_new", "sales_upgrade", "cancel", "tech_support", "outage_fault",
        "appointment_book", "appointment_change", "order_status", "complaint", "account_admin", "hours_location",
        "speak_to_human", Emergency, "wrong_number", "out_of_scope",
    ];
}

public sealed record Department
{
    public required string Id { get; init; }
    public required string Target { get; init; }
    public required string Hours { get; init; }
    public string ClosedAction { get; init; } = "voicemail";
    public bool EmergencyCapable { get; init; }
}

public sealed record CustomIntent
{
    public required string Id { get; init; }
    public string Description { get; init; } = "";
    public ImmutableArray<string> Examples { get; init; } = [];
    public required string Department { get; init; }
}

public sealed record Fallback
{
    public required string Department { get; init; }
    public ImmutableDictionary<string, string> DtmfMenu { get; init; } = ImmutableDictionary<string, string>.Empty;
}

public sealed record RuleWhen
{
    public string? Did { get; init; }
}

public sealed record RuleAction
{
    public required string Route { get; init; }
}

public sealed record RoutingRule
{
    public required RuleWhen When { get; init; }
    public required RuleAction Action { get; init; }
}

public sealed record ChannelBindings
{
    public ImmutableArray<string> Dids { get; init; } = [];
    public ImmutableArray<string> WhatsappPhoneNumberIds { get; init; } = [];
}

/// <summary>
/// The receptionist agent for a tenant: who it is and what it knows. When disabled, calls get the router.
/// Knowledge is the business's own facts (hours, address, services, prices, policies), read to the model as-is.
/// </summary>
public sealed record AgentProfile
{
    public const int MaxKnowledgeChars = 6000;
    public const int MaxGreetingChars = 300;

    public bool Enabled { get; init; }
    public string Name { get; init; } = "";
    /// <summary>A Deepgram Flux voice (flux-gemma-en), an ElevenLabs voice (eleven-&lt;voice id&gt;) or a local Kokoro voice (kokoro-af_heart).</summary>
    public string Voice { get; init; } = "flux-gemma-en";
    /// <summary>A different voice for calls to one of the agent's numbers, for side-by-side comparison.</summary>
    public ImmutableDictionary<string, string> VoiceByExtension { get; init; } = ImmutableDictionary<string, string>.Empty;

    public const string VoicePattern = "^(flux-[a-z]+-[a-z]{2}|kokoro-[a-z]{2}_[a-z]+|eleven-[A-Za-z0-9]{20}|piper-[a-z]{2}_[A-Z]{2}-[a-z_]+-(x_low|low|medium|high)|realtime-[a-z]+|realtime\\+flux-[a-z]+-[a-z]{2})$";

    public string VoiceFor(string dialled) => VoiceByExtension.TryGetValue(dialled, out var voice) ? voice : Voice;
    public string Greeting { get; init; } = "";
    public string Knowledge { get; init; } = "";
    /// <summary>The agent's own numbers. They resolve to this tenant like any other number.</summary>
    public ImmutableArray<string> Extensions { get; init; } = [];
    public string CallerIdName { get; init; } = "";
    public string CallerIdNumber { get; init; } = "";
    /// <summary>What each department handles, in words, so the agent can choose where to transfer.</summary>
    public ImmutableDictionary<string, string> DepartmentNotes { get; init; } = ImmutableDictionary<string, string>.Empty;
}

/// <summary>An immutable tenant configuration. Each decision records the version it used.</summary>
public sealed record TenantConfig
{
    public static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public const int MaxPromptOverrideChars = 500;

    [JsonPropertyName("tenant")] public required string TenantId { get; init; }
    public string Name { get; init; } = "";
    public required string Timezone { get; init; }
    public ChannelBindings Channels { get; init; } = new();
    public ImmutableDictionary<string, ImmutableDictionary<string, ImmutableArray<ImmutableArray<string>>>> Schedules { get; init; }
        = ImmutableDictionary<string, ImmutableDictionary<string, ImmutableArray<ImmutableArray<string>>>>.Empty;
    public ImmutableArray<string> Holidays { get; init; } = [];
    public required ImmutableArray<Department> Departments { get; init; }
    public required ImmutableDictionary<string, string> IntentMap { get; init; }
    public ImmutableArray<CustomIntent> CustomIntents { get; init; } = [];
    public ImmutableArray<RoutingRule> Rules { get; init; } = [];
    public ImmutableArray<string> Languages { get; init; } = [];
    public string PromptOverride { get; init; } = "";
    public ImmutableDictionary<string, double> CostOverrides { get; init; } = ImmutableDictionary<string, double>.Empty;
    public required Fallback Fallback { get; init; }
    public AgentProfile Agent { get; init; } = new();

    [JsonIgnore] public IEnumerable<string> AllIntents => Intents.Canonical.Concat(CustomIntents.Select(c => c.Id).OrderBy(x => x, StringComparer.Ordinal));

    public Department DepartmentById(string id) => Departments.First(d => d.Id == id);

    public string DepartmentFor(string intent)
    {
        foreach (var custom in CustomIntents)
        {
            if (custom.Id == intent) return custom.Department;
        }

        return IntentMap[intent];
    }

    public bool IsOpen(string departmentId, DateTimeOffset at)
    {
        var hours = DepartmentById(departmentId).Hours;
        if (hours == "always") return true;

        var zone = TimeZoneInfo.FindSystemTimeZoneById(Timezone);
        var local = TimeZoneInfo.ConvertTime(at, zone);
        if (Holidays.Contains(local.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture))) return false;

        var day = local.DayOfWeek switch
        {
            DayOfWeek.Monday => "mon",
            DayOfWeek.Tuesday => "tue",
            DayOfWeek.Wednesday => "wed",
            DayOfWeek.Thursday => "thu",
            DayOfWeek.Friday => "fri",
            DayOfWeek.Saturday => "sat",
            _ => "sun",
        };
        if (!Schedules[hours].TryGetValue(day, out var windows)) return false;

        var now = local.ToString("HH:mm", CultureInfo.InvariantCulture);
        foreach (var window in windows)
        {
            if (string.CompareOrdinal(window[0], now) <= 0 && string.CompareOrdinal(now, window[1]) < 0) return true;
        }

        return false;
    }

    /// <summary>Returns every problem with this config. Empty means it may be published.</summary>
    public IReadOnlyList<string> Validate(CostMatrix matrix)
    {
        var errors = new List<string>();
        var ids = Departments.Select(d => d.Id).ToHashSet();
        foreach (var intent in Intents.Canonical)
        {
            if (!IntentMap.TryGetValue(intent, out var dept) || !ids.Contains(dept)) errors.Add($"{intent} maps to no department");
        }

        foreach (var custom in CustomIntents)
        {
            if (!ids.Contains(custom.Department)) errors.Add($"custom intent {custom.Id} maps to no department");
            if (Intents.Canonical.Contains(custom.Id)) errors.Add($"custom intent {custom.Id} shadows a canonical intent");
        }

        foreach (var dept in Departments)
        {
            if (dept.Hours != "always" && !Schedules.ContainsKey(dept.Hours)) errors.Add($"department {dept.Id} uses unknown schedule {dept.Hours}");
        }

        if (IntentMap.TryGetValue(Intents.Emergency, out var emergency) && ids.Contains(emergency))
        {
            var d = DepartmentById(emergency);
            if (d.Hours != "always" || !d.EmergencyCapable) errors.Add("the emergency department must be emergency-capable and always open");
        }

        if (!ids.Contains(Fallback.Department)) errors.Add("fallback department unknown");
        foreach (var target in Fallback.DtmfMenu.Values)
        {
            if (!ids.Contains(target)) errors.Add($"dtmf target {target} unknown");
        }

        foreach (var rule in Rules)
        {
            if (!ids.Contains(rule.Action.Route)) errors.Add($"rule target {rule.Action.Route} unknown");
        }

        if (PromptOverride.Length > MaxPromptOverrideChars) errors.Add($"prompt_override exceeds {MaxPromptOverrideChars} characters");
        if (PromptOverride.Contains("</transcript>", StringComparison.OrdinalIgnoreCase)) errors.Add("prompt_override contains a reserved delimiter");

        if (Agent.Enabled)
        {
            if (Agent.Name.Length is 0 or > 40) errors.Add("agent name must be 1 to 40 characters");
            foreach (var voice in Agent.VoiceByExtension.Values.Prepend(Agent.Voice))
            {
                if (!System.Text.RegularExpressions.Regex.IsMatch(voice, AgentProfile.VoicePattern)) errors.Add($"agent voice {voice} must be a flux voice or a kokoro voice");
            }

            foreach (var extension in Agent.VoiceByExtension.Keys)
            {
                if (!Agent.Extensions.Contains(extension)) errors.Add($"voice override for {extension}, which isn't an agent extension");
            }
            if (Agent.Greeting.Length is 0 or > AgentProfile.MaxGreetingChars) errors.Add($"agent greeting must be 1 to {AgentProfile.MaxGreetingChars} characters");
            if (Agent.Knowledge.Length > AgentProfile.MaxKnowledgeChars) errors.Add($"agent knowledge exceeds {AgentProfile.MaxKnowledgeChars} characters");
            foreach (var text in new[] { Agent.Greeting, Agent.Knowledge }.Concat(Agent.DepartmentNotes.Values))
            {
                if (text.Contains("</business>", StringComparison.OrdinalIgnoreCase) || text.Contains("</caller>", StringComparison.OrdinalIgnoreCase)) errors.Add("agent text contains a reserved delimiter");
            }

            foreach (var dept in Agent.DepartmentNotes.Keys)
            {
                if (!ids.Contains(dept)) errors.Add($"agent note for unknown department {dept}");
            }

            foreach (var number in Agent.Extensions)
            {
                if (!System.Text.RegularExpressions.Regex.IsMatch(number, "^[0-9]{3,15}$")) errors.Add($"agent extension {number} must be 3 to 15 digits");
                if (Channels.Dids.Contains(number)) errors.Add($"agent extension {number} is also a router number");
            }
        }

        foreach (var (key, value) in CostOverrides)
        {
            if (!matrix.Costs.TryGetValue(key, out var current)) errors.Add($"unknown cost {key}");
            else if (matrix.Protected.Contains(key) && value < current) errors.Add($"override may not lower protected cost {key}");
        }

        return errors;
    }
}
