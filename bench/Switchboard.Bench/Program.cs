using System.Text.Json;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using Switchboard.Domain;
using YamlDotNet.Serialization;

BenchmarkRunner.Run<DecisionBenchmarks>(DefaultConfig.Instance.WithOptions(ConfigOptions.DisableOptimizationsValidator));

/// <summary>Stage budget for the decision engine is 5 ms, so its cost should be well under that.</summary>
[MemoryDiagnoser]
[JsonExporterAttribute.Full]
public class DecisionBenchmarks
{
    private TenantConfig _tenant = null!;
    private CostMatrix _matrix = null!;
    private DecisionInput _confident = null!;
    private DecisionInput _uncertain = null!;
    private EmergencyLexicon _lexicon = null!;

    [GlobalSetup]
    public void Setup()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "buf.yaml"))) dir = dir.Parent;
        var root = dir?.FullName ?? throw new InvalidOperationException("repo root not found");
        _matrix = CostMatrix.FromYaml(File.ReadAllText(Path.Combine(root, "configs/cost_matrix.yaml")));
        _lexicon = EmergencyLexicon.FromYaml(File.ReadAllText(Path.Combine(root, "configs/emergency_lexicon.yaml")));
        var yaml = new DeserializerBuilder().WithAttemptingUnquotedStringTypeDeserialization().Build()
            .Deserialize(new StringReader(File.ReadAllText(Path.Combine(root, "configs/tenants/quillfibre.yaml"))));
        _tenant = JsonSerializer.Deserialize<TenantConfig>(new SerializerBuilder().JsonCompatible().Build().Serialize(yaml!), TenantConfig.Json)!;
        var at = DateTimeOffset.Parse("2026-11-04T08:00:00Z");
        _confident = new DecisionInput { At = at, Scores = new Dictionary<string, double> { ["billing"] = 0.97, ["complaint"] = 0.03 } };
        _uncertain = new DecisionInput { At = at, LlmContributed = true, Scores = Intents.Canonical.ToDictionary(i => i, _ => 1.0) };
    }

    [Benchmark]
    public Decision Confident() => DecisionEngine.Decide(_tenant, _matrix, _confident);

    [Benchmark]
    public Decision UncertainAllIntents() => DecisionEngine.Decide(_tenant, _matrix, _uncertain);

    [Benchmark]
    public bool LexiconScan() => _lexicon.Matches("my water has been off since tuesday and i have paid and nobody is answering the phone");
}
