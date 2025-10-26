using System.Collections.Concurrent;
using System.Diagnostics;

namespace Switchboard.Gateway;

/// <summary>One conversation is one trace; each stage is a span. Transcript text is never an attribute.</summary>
public static class Tracing
{
    public const string SourceName = "Switchboard.Gateway";
    public static readonly ActivitySource Source = new(SourceName);

    public static Activity? StartConversation(string tenant, string conversation, string version, string channel)
    {
        var activity = Source.StartActivity("conversation", ActivityKind.Server);
        activity?.SetTag("tenant.id", tenant);
        activity?.SetTag("langfuse.session.id", conversation);
        activity?.SetTag("langfuse.trace.tags", new[] { tenant, channel });
        activity?.SetTag("config.version", version);
        activity?.SetTag("channel", channel);
        return activity;
    }

    public static Activity? StartStage(string stage, string tenant)
    {
        var activity = Source.StartActivity(stage);
        activity?.SetTag("tenant.id", tenant);
        return activity;
    }
}

/// <summary>Per-tenant LLM allowance, so one tenant can't spend the capacity meant for another.</summary>
public sealed class LlmBudget(IConfiguration config, TimeProvider clock)
{
    private readonly int _perMinute = int.TryParse(config["TENANT_LLM_PER_MINUTE"], out var n) ? n : 120;
    private readonly ConcurrentDictionary<string, Window> _windows = new();

    private sealed class Window
    {
        public long Minute;
        public int Used;
    }

    public bool TryConsume(string tenant)
    {
        var minute = clock.GetUtcNow().ToUnixTimeSeconds() / 60;
        var window = _windows.GetOrAdd(tenant, _ => new Window());
        lock (window)
        {
            if (window.Minute != minute)
            {
                window.Minute = minute;
                window.Used = 0;
            }

            if (window.Used >= _perMinute) return false;

            window.Used++;
            return true;
        }
    }
}
