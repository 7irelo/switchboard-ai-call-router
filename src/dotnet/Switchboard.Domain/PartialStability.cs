namespace Switchboard.Domain;

/// <summary>
/// Tracks the top intent across partial transcripts. Interim results are coarse and early intent is noisy,
/// so nothing acts on a partial until the same intent has led for several updates in a row.
/// </summary>
public sealed class PartialStability(int requiredUpdates = 3, double minProbability = 0.6)
{
    private string? _leader;
    private int _streak;

    public int Flips { get; private set; }

    public string? StableIntent => _streak >= requiredUpdates ? _leader : null;

    public string? Update(IReadOnlyDictionary<string, double> scores)
    {
        if (scores.Count == 0) return StableIntent;

        var top = scores.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.Ordinal).First();
        if (top.Key != _leader)
        {
            if (_leader is not null) Flips++;
            _leader = top.Key;
            _streak = 0;
        }

        _streak = top.Value >= minProbability ? _streak + 1 : 0;
        return StableIntent;
    }
}
