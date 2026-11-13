using System.Collections.Immutable;
using System.Globalization;
using System.Text.Json;
using Switchboard.Contracts.V1;
using Switchboard.Domain;

namespace Switchboard.Gateway;

/// <summary>How an agent conversation ended, from the gateway's side.</summary>
public enum AgentOutcomeKind { Fallback, Route, Hangup }

public sealed record AgentOutcome(AgentOutcomeKind Kind, Decision? Decision = null, string Reason = "")
{
    public static AgentOutcome Fallback(string reason) => new(AgentOutcomeKind.Fallback, Reason: reason);

    public static AgentOutcome Hangup(string reason) => new(AgentOutcomeKind.Hangup, Reason: reason);

    public static AgentOutcome RouteTo(TenantConfig config, string department, string reason, DateTimeOffset at) =>
        new(AgentOutcomeKind.Route, new Decision(ActionKind.Route, department, 0, reason, [], !config.IsOpen(department, at)), reason);
}

/// <summary>Builds what the agent is told about the tenant at the start of a call.</summary>
public static class AgentBriefing
{
    public static bool IsAgentNumber(TenantConfig config, string dialled) => config.Agent.Enabled && config.Agent.Extensions.Contains(dialled);

    public static AgentStart Build(TenantConfig config, TenantContext context, DateTimeOffset now, string callerNumber, string purpose = "", string dialled = "")
    {
        var agent = config.Agent;
        var start = new AgentStart
        {
            Context = context,
            BusinessName = config.Name.Replace(" (fictional)", "", StringComparison.Ordinal),
            AgentName = agent.Name,
            Voice = agent.VoiceFor(dialled),
            Greeting = agent.Greeting,
            Knowledge = agent.Knowledge,
            LocalTime = LocalTime(config, now),
            Purpose = purpose,
            CallerNumber = callerNumber,
        };
        foreach (var d in config.Departments)
        {
            var handles = agent.DepartmentNotes.TryGetValue(d.Id, out var note) && note.Length > 0
                ? note
                : string.Join(", ", config.IntentMap.Where(kv => kv.Value == d.Id).Select(kv => kv.Key.Replace('_', ' ')).Order(StringComparer.Ordinal));
            start.Departments.Add(new AgentDepartment
            {
                Id = d.Id,
                Handles = handles,
                OpenNow = config.IsOpen(d.Id, now),
                WhenClosed = d.ClosedAction == "voicemail" ? "voicemail" : $"the {config.Fallback.Department} department",
                Emergency = d.EmergencyCapable,
            });
        }

        return start;
    }

    public static string LocalTime(TenantConfig config, DateTimeOffset now)
    {
        var local = TimeZoneInfo.ConvertTime(now, TimeZoneInfo.FindSystemTimeZoneById(config.Timezone));
        return local.ToString("dddd d MMMM yyyy, HH:mm", CultureInfo.InvariantCulture);
    }
}

/// <summary>
/// Carries out the agent's tool calls. The model only asks: every argument is checked here against this tenant,
/// and anything outside it is refused with a reason the model can read.
/// </summary>
public static class AgentTools
{
    public const int MaxField = 500;

    public sealed record Result(bool Ok, string Content, string Detail, string? ApprovedTransfer = null, bool Hangup = false);

    public static Result Execute(TenantConfig config, ToolCall call, ConversationLog log, DateTimeOffset now)
    {
        JsonElement args;
        try
        {
            using var doc = JsonDocument.Parse(call.ArgumentsJson);
            args = doc.RootElement.Clone();
        }
        catch (JsonException)
        {
            return Refuse("the arguments weren't valid JSON", "bad_json");
        }

        if (args.ValueKind != JsonValueKind.Object) return Refuse("the arguments must be an object", "bad_json");

        switch (call.Name)
        {
            case "transfer_to_department":
            {
                var department = Text(args, "department");
                if (!config.Departments.Any(d => d.Id == department))
                {
                    var valid = string.Join(", ", config.Departments.Select(d => d.Id));
                    return Refuse($"there's no department called '{department}'. Use one of: {valid}", "unknown_department");
                }

                var open = config.IsOpen(department, now);
                var closedAction = config.DepartmentById(department).ClosedAction;
                var content = open
                    ? Json(new { status = "connecting", department })
                    : Json(new { status = "closed", department, handled_by = closedAction == "voicemail" ? "voicemail" : config.Fallback.Department });
                return new Result(true, content, department, ApprovedTransfer: department);
            }

            case "take_message":
            {
                var name = Text(args, "caller_name");
                var number = Text(args, "callback_number");
                var message = Text(args, "message");
                if (name.Length == 0 || number.Length == 0 || message.Length == 0) return Refuse("a name, a callback number and the message are all needed", "incomplete");
                var forDepartment = Text(args, "for_department");
                if (forDepartment.Length > 0 && !config.Departments.Any(d => d.Id == forDepartment)) forDepartment = "";
                var urgent = args.TryGetProperty("urgent", out var u) && u.ValueKind == JsonValueKind.True;
                log.Emit(e => e.MessageTaken = new MessageTaken { CallerName = name, CallbackNumber = number, Message = message, Urgent = urgent, ForDepartment = forDepartment });
                return new Result(true, Json(new { status = "saved" }), "message");
            }

            case "book_callback":
            {
                var name = Text(args, "caller_name");
                var number = Text(args, "callback_number");
                var when = Text(args, "preferred_time");
                var reason = Text(args, "reason");
                if (name.Length == 0 || number.Length == 0 || when.Length == 0) return Refuse("a name, a callback number and a preferred time are all needed", "incomplete");
                log.Emit(e => e.CallbackBooked = new CallbackBooked { CallerName = name, CallbackNumber = number, PreferredTime = when, Reason = reason });
                return new Result(true, Json(new { status = "booked", note = "Staff call back at the requested time when they can; no exact time is guaranteed." }), "callback");
            }

            case "end_call":
                return new Result(true, Json(new { status = "ending" }), "end_call", Hangup: true);

            default:
                return Refuse($"there's no tool called {call.Name}", "unknown_tool");
        }
    }

    private static Result Refuse(string error, string detail) => new(false, Json(new { error }), detail);

    private static string Text(JsonElement args, string name)
    {
        if (!args.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.String) return "";
        var text = value.GetString()!.Trim();
        return text.Length > MaxField ? text[..MaxField] : text;
    }

    private static string Json(object value) => JsonSerializer.Serialize(value);
}

/// <summary>The call's lines, kept for the post-call summary. Never logged.</summary>
public sealed class CallRecord
{
    private readonly List<SummaryLine> _lines = [];

    public void Add(string role, string text)
    {
        if (text.Length == 0) return;
        lock (_lines) _lines.Add(new SummaryLine { Role = role, Text = text });
    }

    public ImmutableArray<SummaryLine> Lines
    {
        get
        {
            lock (_lines) return [.. _lines];
        }
    }
}

/// <summary>Reads the chan_websocket control events the agent needs.</summary>
public static class MediaEvent
{
    /// <summary>The correlation id of a MEDIA_MARK_PROCESSED event, or null for anything else.</summary>
    public static string? MarkProcessed(ReadOnlySpan<byte> json)
    {
        if (json.IndexOf("MEDIA_MARK_PROCESSED"u8) < 0) return null;
        try
        {
            var reader = new Utf8JsonReader(json);
            using var doc = JsonDocument.ParseValue(ref reader);
            var root = doc.RootElement;
            return root.TryGetProperty("event", out var e) && e.GetString() == "MEDIA_MARK_PROCESSED" && root.TryGetProperty("correlation_id", out var id)
                ? id.GetString()
                : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
