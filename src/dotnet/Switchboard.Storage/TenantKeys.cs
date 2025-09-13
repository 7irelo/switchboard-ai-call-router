namespace Switchboard.Storage;

/// <summary>The only place Redis keys are built, so every key carries its tenant.</summary>
public static class TenantKeys
{
    private const string Prefix = "sb:t:";

    public static string Events(string tenantId)
    {
        if (string.IsNullOrWhiteSpace(tenantId) || tenantId.Contains(':')) throw new ArgumentException("invalid tenant id", nameof(tenantId));
        return $"{Prefix}{tenantId}:events";
    }
}
