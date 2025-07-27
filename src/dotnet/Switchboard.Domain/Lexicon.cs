using System.Collections.Immutable;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace Switchboard.Domain;

/// <summary>Layer 0 safety net: a narrow phrase list. A hit routes straight to the emergency department.</summary>
public sealed record EmergencyLexicon(ImmutableArray<string> Phrases)
{
    public bool Matches(string text)
    {
        var lowered = string.Join(' ', text.ToLowerInvariant().Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        foreach (var phrase in Phrases)
        {
            if (lowered.Contains(phrase, StringComparison.Ordinal)) return true;
        }

        return false;
    }

    public static EmergencyLexicon FromYaml(string yaml)
    {
        var raw = new DeserializerBuilder().WithNamingConvention(UnderscoredNamingConvention.Instance)
            .IgnoreUnmatchedProperties().Build().Deserialize<Raw>(yaml);
        return new EmergencyLexicon([.. raw.Phrases.Select(p => p.ToLowerInvariant())]);
    }

    private sealed class Raw
    {
        public List<string> Phrases { get; set; } = [];
    }
}
