using System.Collections.Immutable;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace Switchboard.Domain;

public sealed record CostMatrix
{
    public required ImmutableDictionary<string, double> Costs { get; init; }
    public required ImmutableArray<ImmutableHashSet<string>> Adjacency { get; init; }
    public required ImmutableHashSet<string> Protected { get; init; }
    public required double TauLlm { get; init; }
    public required double TauClarify { get; init; }
    public required double VoicemailGuard { get; init; }

    public double this[string key] => Costs[key];

    public bool Adjacent(string a, string b)
    {
        foreach (var group in Adjacency)
        {
            if (group.Contains(a) && group.Contains(b)) return true;
        }

        return false;
    }

    /// <summary>A tenant may change costs, but can never lower a protected emergency penalty.</summary>
    public CostMatrix WithOverrides(IReadOnlyDictionary<string, double> overrides)
    {
        var costs = Costs;
        foreach (var (key, value) in overrides)
        {
            if (!Costs.TryGetValue(key, out var current)) throw new ArgumentException($"unknown cost {key}");
            if (Protected.Contains(key) && value < current) throw new ArgumentException($"override may not lower protected cost {key}");
            costs = costs.SetItem(key, value);
        }

        return this with { Costs = costs };
    }

    public static CostMatrix FromYaml(string yaml)
    {
        var raw = new DeserializerBuilder().WithNamingConvention(UnderscoredNamingConvention.Instance)
            .IgnoreUnmatchedProperties().Build().Deserialize<RawMatrix>(yaml);
        return new CostMatrix
        {
            Costs = raw.Costs.ToImmutableDictionary(),
            Adjacency = [.. raw.AdjacencyGroups.Select(g => g.ToImmutableHashSet())],
            Protected = raw.ProtectedCosts.ToImmutableHashSet(),
            TauLlm = raw.Thresholds["tau_llm"],
            TauClarify = raw.Thresholds["tau_clarify"],
            VoicemailGuard = raw.EmergencyVoicemailGuard,
        };
    }

    private sealed class RawMatrix
    {
        public Dictionary<string, double> Costs { get; set; } = [];
        public List<List<string>> AdjacencyGroups { get; set; } = [];
        public List<string> ProtectedCosts { get; set; } = [];
        public double EmergencyVoicemailGuard { get; set; }
        public Dictionary<string, double> Thresholds { get; set; } = [];
    }
}
