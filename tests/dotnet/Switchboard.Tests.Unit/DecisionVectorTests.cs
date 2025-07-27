using System.Collections.Immutable;
using System.Text.Json;
using Switchboard.Domain;
using Xunit;
using YamlDotNet.Serialization;

namespace Switchboard.Tests.Unit;

public static class Repo
{
    public static string Root { get; } = Find();

    private static string Find()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "buf.yaml"))) dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("repo root not found");
    }

    public static CostMatrix Matrix() => CostMatrix.FromYaml(File.ReadAllText(Path.Combine(Root, "configs/cost_matrix.yaml")));

    /// <summary>Tenant YAML goes through JSON so tests use the same parser as published snapshots.</summary>
    public static TenantConfig Tenant(string id)
    {
        var yaml = new DeserializerBuilder().WithAttemptingUnquotedStringTypeDeserialization().Build().Deserialize(new StringReader(File.ReadAllText(Path.Combine(Root, "configs/tenants", id + ".yaml"))));
        var json = new SerializerBuilder().JsonCompatible().Build().Serialize(yaml!);
        return JsonSerializer.Deserialize<TenantConfig>(json, TenantConfig.Json)!;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, TenantConfig> Cache = new();

    public static TenantConfig CachedTenant(string id) => Cache.GetOrAdd(id, Tenant);

    public static string[] TenantIds() => [.. Directory.GetFiles(Path.Combine(Root, "configs/tenants"), "*.yaml").Select(f => Path.GetFileNameWithoutExtension(f)!)];
}

public class DecisionVectorTests
{
    /// <summary>The Python eval twin and this production engine must give the same answer on every shared vector.</summary>
    [Fact]
    public void Matches_shared_golden_vectors()
    {
        var matrix = Repo.Matrix();
        var tenants = Repo.TenantIds().ToDictionary(id => id, Repo.Tenant);
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Repo.Root, "tests/golden/decision_vectors.json")));
        var count = 0;
        foreach (var v in doc.RootElement.EnumerateArray())
        {
            var scores = v.GetProperty("scores").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetDouble());
            var input = new DecisionInput
            {
                Scores = scores,
                At = v.GetProperty("at").GetDateTimeOffset(),
                ClarifyTurns = v.GetProperty("clarify_turns").GetInt32(),
                LlmContributed = v.GetProperty("llm_contributed").GetBoolean(),
                LlmAvailable = v.GetProperty("llm_available").GetBoolean(),
                Degraded = v.GetProperty("degraded").GetBoolean(),
                Dtmf = v.GetProperty("dtmf").ValueKind == JsonValueKind.Null ? null : v.GetProperty("dtmf").GetString(),
                LexiconHit = v.GetProperty("lexicon_hit").GetBoolean(),
            };
            var got = DecisionEngine.Decide(tenants[v.GetProperty("tenant").GetString()!], matrix, input);
            var e = v.GetProperty("expect");
            var context = $"vector {count}: {v}";
            Assert.True(e.GetProperty("action").GetString() == got.Action.ToString().ToLowerInvariant(), context + " got " + got);
            Assert.True((e.GetProperty("department").ValueKind == JsonValueKind.Null ? null : e.GetProperty("department").GetString()) == got.Department, context + " got " + got);
            Assert.True(e.GetProperty("reason").GetString() == got.Reason, context + " got " + got);
            Assert.True(e.GetProperty("closed").GetBoolean() == got.Closed, context + " got " + got);
            Assert.Equal(e.GetProperty("candidates").EnumerateArray().Select(x => x.GetString()!), got.Candidates);
            Assert.True(Math.Abs(e.GetProperty("expected_cost").GetDouble() - got.ExpectedCost) < 1e-8, context + " got " + got);
            count++;
        }

        Assert.True(count >= 500);
    }
}

public class DecisionEngineTests
{
    private static readonly CostMatrix M = Repo.Matrix();
    private static readonly DateTimeOffset Open = DateTimeOffset.Parse("2026-11-04T08:00:00Z");
    private static readonly DateTimeOffset Night = DateTimeOffset.Parse("2026-11-04T20:30:00Z");

    [Fact]
    public void Seed_tenants_are_valid()
    {
        foreach (var id in Repo.TenantIds()) Assert.Empty(Repo.Tenant(id).Validate(M));
        Assert.Equal(5, Repo.TenantIds().Length);
    }

    [Fact]
    public void Confident_billing_routes_to_accounts()
    {
        var d = DecisionEngine.Decide(Repo.Tenant("thornveld_vet"), M, new DecisionInput { At = Open, Scores = new Dictionary<string, double> { ["billing"] = 1 } });
        Assert.Equal((ActionKind.Route, "accounts"), (d.Action, d.Department));
    }

    [Fact]
    public void Emergency_tail_stops_a_confident_route()
    {
        var d = DecisionEngine.Decide(Repo.Tenant("thornveld_vet"), M, new DecisionInput
        {
            At = Open,
            Scores = new Dictionary<string, double> { ["appointment_book"] = 0.9, ["emergency"] = 0.1 },
        });
        Assert.Equal(ActionKind.Escalate, d.Action);
    }

    [Fact]
    public void Inference_down_with_lexicon_hit_reaches_emergency_at_night()
    {
        var d = DecisionEngine.Decide(Repo.Tenant("thornveld_vet"), M, new DecisionInput { At = Night, LlmAvailable = false, LexiconHit = true });
        Assert.Equal("on_call_vet", d.Department);
    }

    [Fact]
    public void Another_tenants_custom_intent_is_ignored()
    {
        var d = DecisionEngine.Decide(Repo.Tenant("thornveld_vet"), M, new DecisionInput { At = Open, Scores = new Dictionary<string, double> { ["claims_new"] = 1, ["billing"] = 0.01 } });
        Assert.Equal("accounts", d.Department);
    }

    [Fact]
    public void Override_cannot_lower_emergency_penalty()
    {
        foreach (var key in M.Protected)
        {
            Assert.Throws<ArgumentException>(() => M.WithOverrides(ImmutableDictionary<string, double>.Empty.Add(key, M[key] - 1)));
        }

        var bad = Repo.Tenant("thornveld_vet") with { CostOverrides = ImmutableDictionary<string, double>.Empty.Add("emergency_to_voicemail", 1) };
        Assert.Contains(bad.Validate(M), e => e.Contains("protected"));
    }

    /// <summary>Randomised check of the invariant the design exists for.</summary>
    [Fact]
    public void Emergency_mass_never_reaches_voicemail()
    {
        var rng = new Random(42);
        var ids = Repo.TenantIds();
        for (var n = 0; n < 5000; n++)
        {
            var t = Repo.CachedTenant(ids[rng.Next(ids.Length)]);
            var share = 0.02 + (rng.NextDouble() * 0.98);
            var other = Intents.Canonical[rng.Next(Intents.Canonical.Length)];
            var scores = new Dictionary<string, double> { [other] = 1 - share };
            scores[Intents.Emergency] = other == Intents.Emergency ? 1.0 : share;
            var d = DecisionEngine.Decide(t, M, new DecisionInput
            {
                At = Open.AddMinutes(rng.Next(0, 60 * 24 * 60)),
                Scores = scores,
                LlmContributed = true,
                ClarifyTurns = rng.Next(0, 3),
            });
            if (d is { Action: ActionKind.Route, Closed: true })
            {
                Assert.NotEqual("voicemail", t.DepartmentById(d.Department!).ClosedAction);
            }
        }
    }
}
