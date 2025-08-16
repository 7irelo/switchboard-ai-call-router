using System.Text.RegularExpressions;
using Xunit;

namespace Switchboard.Tests.Architecture;

/// <summary>Source-level rules that keep tenant data behind TenantDb and policy in one place.</summary>
public partial class ArchitectureTests
{
    private static readonly string Root = FindRoot();

    private static string FindRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "buf.yaml"))) dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("repo root not found");
    }

    private static IEnumerable<string> Sources() =>
        Directory.EnumerateFiles(Path.Combine(Root, "src/dotnet"), "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"));

    [GeneratedRegex(@"new\s+NpgsqlConnection\s*\(|\.OpenConnectionAsync\s*\(|\.CreateConnection\s*\(")]
    private static partial Regex DirectConnection();

    [Fact]
    public void Only_TenantDb_and_the_migrator_open_database_connections()
    {
        var allowed = new[] { "TenantDb.cs", "Migrator.cs" };
        var offenders = Sources().Where(f => !allowed.Contains(Path.GetFileName(f)) && DirectConnection().IsMatch(File.ReadAllText(f))).ToList();
        Assert.True(offenders.Count == 0, "direct connection use: " + string.Join(", ", offenders));
    }

    [Fact]
    public void Domain_stays_pure()
    {
        var banned = new[] { "Npgsql", "System.Net", "DateTime.Now", "DateTimeOffset.Now", "DateTimeOffset.UtcNow", "StackExchange" };
        foreach (var file in Sources().Where(f => f.Contains("Switchboard.Domain")))
        {
            var text = File.ReadAllText(file);
            foreach (var word in banned) Assert.False(text.Contains(word, StringComparison.Ordinal), $"{file} uses {word}");
        }
    }

    [Fact]
    public void Redis_keys_only_come_from_the_tenant_key_wrapper()
    {
        var offenders = Sources().Where(f => Path.GetFileName(f) != "TenantKeys.cs" && File.ReadAllText(f).Contains("\"sb:", StringComparison.Ordinal)).ToList();
        Assert.True(offenders.Count == 0, "raw redis keys: " + string.Join(", ", offenders));
    }
}
