namespace Switchboard.Domain;

public sealed record TurnContext
{
    public required TenantConfig Config { get; init; }
    public required CostMatrix Matrix { get; init; }
    public required EmergencyLexicon Lexicon { get; init; }
    public required DateTimeOffset At { get; init; }
    public string Transcript { get; init; } = "";
    public IReadOnlyDictionary<string, double> Scores { get; init; } = new Dictionary<string, double>();
    public int ClarifyTurns { get; init; }
    public bool Degraded { get; init; }
    public bool LlmAvailable { get; init; } = true;
    public string? Did { get; init; }
    public string? Dtmf { get; init; }
}

/// <summary>Runs one turn to a non-escalate decision. Voice and messages share this, so policy has one home.</summary>
public static class DecisionFlow
{
    /// <param name="ctx">The turn.</param>
    /// <param name="escalate">Asks the LLM layer. Returns fused scores, or null if it gave no usable answer.</param>
    public static async Task<Decision> ResolveAsync(TurnContext ctx, Func<TurnContext, Task<IReadOnlyDictionary<string, double>?>> escalate)
    {
        var input = new DecisionInput
        {
            At = ctx.At,
            Scores = ctx.Scores,
            ClarifyTurns = ctx.ClarifyTurns,
            LlmAvailable = ctx.LlmAvailable,
            Degraded = ctx.Degraded,
            Did = ctx.Did,
            Dtmf = ctx.Dtmf,
            LexiconHit = ctx.Lexicon.Matches(ctx.Transcript),
        };
        var matrix = ctx.Config.CostOverrides.IsEmpty ? ctx.Matrix : ctx.Matrix.WithOverrides(ctx.Config.CostOverrides);
        var decision = DecisionEngine.Decide(ctx.Config, matrix, input);
        if (decision.Action != ActionKind.Escalate) return decision;

        var fused = await escalate(ctx);
        return DecisionEngine.Decide(ctx.Config, matrix, input with { Scores = fused ?? ctx.Scores, LlmContributed = true });
    }

    /// <summary>What a decision does once it leaves the engine: the department's target, or how it's closed.</summary>
    public static string TargetFor(TenantConfig config, Decision decision)
    {
        var department = config.DepartmentById(decision.Department!);
        if (!decision.Closed) return department.Target;
        return department.ClosedAction == "voicemail"
            ? department.Target + "-vm"
            : config.DepartmentById(config.Fallback.Department).Target;
    }
}
