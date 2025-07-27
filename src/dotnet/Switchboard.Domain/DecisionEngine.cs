using System.Collections.Immutable;

namespace Switchboard.Domain;

public enum ActionKind { Route, Reception, Clarify, Escalate }

public sealed record DecisionInput
{
    public IReadOnlyDictionary<string, double> Scores { get; init; } = ImmutableDictionary<string, double>.Empty;
    public required DateTimeOffset At { get; init; }
    public int ClarifyTurns { get; init; }
    public bool LlmContributed { get; init; }
    public bool LlmAvailable { get; init; } = true;
    public bool Degraded { get; init; }
    public string? Did { get; init; }
    public string? Dtmf { get; init; }
    public bool LexiconHit { get; init; }
}

public sealed record Decision(
    ActionKind Action,
    string? Department,
    double ExpectedCost,
    string Reason,
    ImmutableArray<string> Candidates,
    bool Closed);

/// <summary>
/// Picks the action with the lowest expected misroute cost (docs/PLAN.md section 8, docs/adr/0002).
/// Pure: no I/O and no clock, so it can be tested exhaustively and still runs when inference is down.
/// </summary>
public static class DecisionEngine
{
    public const int MaxClarifyTurns = 2;
    public const double DegradedThresholdFactor = 0.5;
    private const double Eps = 1e-12;

    public static Decision Decide(TenantConfig t, CostMatrix m, DecisionInput input)
    {
        // Layer 0: deterministic rules.
        if (input.Dtmf is not null && t.Fallback.DtmfMenu.TryGetValue(input.Dtmf, out var menuTarget))
        {
            return new Decision(ActionKind.Route, menuTarget, 0, "dtmf", [], !t.IsOpen(menuTarget, input.At));
        }

        if (input.Did is not null)
        {
            foreach (var rule in t.Rules)
            {
                if (rule.When.Did == input.Did)
                {
                    var target = rule.Action.Route;
                    return new Decision(ActionKind.Route, target, 0, "did_rule", [], !t.IsOpen(target, input.At));
                }
            }
        }

        if (input.LexiconHit)
        {
            // Someone who says "not breathing" is routed, not questioned, even if inference is down.
            return new Decision(ActionKind.Route, t.DepartmentFor(Intents.Emergency), 0, "emergency_lexicon", [], false);
        }

        var probs = Normalise(t, input);
        var factor = input.Degraded ? DegradedThresholdFactor : 1.0;
        var tauLlm = m.TauLlm * factor;
        var tauClarify = m.TauClarify * factor;

        var bestCost = double.PositiveInfinity;
        var bestAction = ActionKind.Route;
        string? bestDept = null;
        foreach (var dept in t.Departments.Select(d => d.Id).OrderBy(x => x, StringComparer.Ordinal))
        {
            if (probs[Intents.Emergency] >= m.VoicemailGuard && ReachesVoicemail(t, dept, input.At)) continue;

            var ec = Expected(probs, i => RouteCost(t, m, i, dept, input.At));
            if (ec < bestCost)
            {
                (bestCost, bestAction, bestDept) = (ec, ActionKind.Route, dept);
            }
        }

        var fallback = t.Fallback.Department;
        if (t.IsOpen(fallback, input.At))
        {
            var ec = Expected(probs, i => i == Intents.Emergency ? m["emergency_to_wrong_live_department"] : m["reception"]);
            if (ec < bestCost)
            {
                (bestCost, bestAction, bestDept) = (ec, ActionKind.Reception, fallback);
            }
        }

        var closed = bestAction == ActionKind.Route && !t.IsOpen(bestDept!, input.At);

        if (bestCost <= tauLlm + Eps) return new Decision(bestAction, bestDept, bestCost, "confident", [], closed);
        if (!input.LlmContributed && input.LlmAvailable) return new Decision(ActionKind.Escalate, null, bestCost, "above_tau_llm", [], false);

        if (input.ClarifyTurns < MaxClarifyTurns && bestCost > tauClarify + Eps)
        {
            var clarify = Expected(probs, i => ClarifyCost(m, i, input.ClarifyTurns));
            if (clarify < bestCost) return new Decision(ActionKind.Clarify, null, clarify, "above_tau_clarify", Candidates(t, probs), false);
        }

        return new Decision(bestAction, bestDept, bestCost, "best_available", [], closed);
    }

    public static double RouteCost(TenantConfig t, CostMatrix m, string intent, string dept, DateTimeOffset at)
    {
        var isEmergency = intent == Intents.Emergency;
        var trueDept = t.DepartmentFor(intent);

        if (!t.IsOpen(dept, at))
        {
            if (t.DepartmentById(dept).ClosedAction == "voicemail")
            {
                if (isEmergency) return m["emergency_to_voicemail"];
                if (trueDept == dept) return m["correct_route"];
                return t.IsOpen(trueDept, at) ? m["voicemail_while_open"] : m["unrelated_department"];
            }

            // A closed department that hands to reception reaches a person.
            if (isEmergency) return m["emergency_to_wrong_live_department"];
            return trueDept == dept ? m["correct_route"] : m["reception"];
        }

        if (trueDept == dept) return m["correct_route"];
        if (isEmergency) return m["emergency_to_wrong_live_department"];
        if (dept == t.DepartmentFor(Intents.Emergency)) return m["non_emergency_to_emergency_route"];

        foreach (var other in t.AllIntents)
        {
            if (t.DepartmentFor(other) == dept && m.Adjacent(intent, other)) return m["adjacent_department"];
        }

        return m["unrelated_department"];
    }

    private static bool ReachesVoicemail(TenantConfig t, string dept, DateTimeOffset at) =>
        !t.IsOpen(dept, at) && t.DepartmentById(dept).ClosedAction == "voicemail";

    private static double ClarifyCost(CostMatrix m, string intent, int turns)
    {
        if (intent != Intents.Emergency) return m["clarify"];
        return turns == 0 ? m["emergency_clarify"] : m["emergency_to_wrong_live_department"];
    }

    private static double Expected(Dictionary<string, double> probs, Func<string, double> cost)
    {
        var total = 0.0;
        foreach (var (intent, p) in probs) total += p * cost(intent);
        return total;
    }

    /// <summary>Keeps this tenant's intents only; anything else is dropped. No usable scores becomes a uniform prior.</summary>
    private static Dictionary<string, double> Normalise(TenantConfig t, DecisionInput input)
    {
        var probs = new Dictionary<string, double>();
        foreach (var intent in t.AllIntents)
        {
            probs[intent] = input.Scores.TryGetValue(intent, out var s) ? Math.Max(0, s) : 0;
        }

        var total = probs.Values.Sum();
        foreach (var intent in probs.Keys.ToList())
        {
            probs[intent] = total > Eps ? probs[intent] / total : 1.0 / probs.Count;
        }

        return probs;
    }

    /// <summary>Top two intents in different departments: asking about one department tells us nothing.</summary>
    private static ImmutableArray<string> Candidates(TenantConfig t, Dictionary<string, double> probs)
    {
        var picked = ImmutableArray.CreateBuilder<string>();
        var seen = new HashSet<string>();
        foreach (var (intent, _) in probs.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.Ordinal))
        {
            if (seen.Add(t.DepartmentFor(intent))) picked.Add(intent);
            if (picked.Count == 2) break;
        }

        return picked.ToImmutable();
    }
}
