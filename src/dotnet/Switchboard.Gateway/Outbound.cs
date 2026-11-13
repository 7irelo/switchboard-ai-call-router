using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using Switchboard.Domain;

namespace Switchboard.Gateway;

public sealed record OutboundRequest(string Tenant, string Destination, string Purpose, bool Consent);

public sealed record OutboundCall(string Tenant, string Destination, string Purpose);

/// <summary>
/// The agent calling someone, with its own name and number as caller ID. Locally the only destination is the
/// softphone ("softphone"); a number is dialled only when a trunk is configured, and only with recorded consent.
/// </summary>
public sealed class OutboundCalls(AriClient ari, ConfigCache configs, IConfiguration config, ILogger<OutboundCalls> log)
{
    public const int MaxPurpose = 300;
    private static readonly TimeSpan PendingFor = TimeSpan.FromMinutes(2);
    private readonly ConcurrentDictionary<string, (OutboundCall Call, DateTimeOffset At)> _pending = new();
    private readonly string? _trunk = string.IsNullOrWhiteSpace(config["OUTBOUND_TRUNK"]) ? null : config["OUTBOUND_TRUNK"]!.Trim();

    public string? Validate(OutboundRequest r)
    {
        if (!r.Consent) return "consent is required: only call people who agreed to be called";
        if (r.Purpose.Trim().Length is 0 or > MaxPurpose) return $"purpose must be 1 to {MaxPurpose} characters";
        if (configs.Get(r.Tenant) is not { } entry || !entry.Config.Agent.Enabled || entry.Config.Agent.Extensions.IsEmpty) return "this tenant has no receptionist agent";
        if (r.Destination == "softphone") return null;
        if (_trunk is null) return "no trunk is configured, so the only destination is \"softphone\"";
        return System.Text.RegularExpressions.Regex.IsMatch(r.Destination, "^\\+?[0-9]{6,15}$") ? null : "destination must be a phone number";
    }

    public async Task<string> StartAsync(OutboundRequest r, CancellationToken ct)
    {
        var agent = configs.Get(r.Tenant)!.Value.Config.Agent;
        var token = Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(12));
        foreach (var (key, value) in _pending)
        {
            if (DateTimeOffset.UtcNow - value.At > PendingFor) _pending.TryRemove(key, out _);
        }

        _pending[token] = (new OutboundCall(r.Tenant, r.Destination, r.Purpose.Trim()), DateTimeOffset.UtcNow);
        var endpoint = r.Destination == "softphone" ? "PJSIP/phone" : $"PJSIP/{r.Destination.TrimStart('+')}@{_trunk}";
        var number = agent.CallerIdNumber.Length > 0 ? agent.CallerIdNumber : agent.Extensions[0];
        var callerId = $"\"{(agent.CallerIdName.Length > 0 ? agent.CallerIdName : agent.Name).Replace("\"", "", StringComparison.Ordinal)}\" <{number}>";
        await ari.OriginateAsync(endpoint, $"outbound,{token}", callerId, ct);
        log.LogInformation("outbound call started tenant={Tenant} destination_kind={Kind}", r.Tenant, r.Destination == "softphone" ? "softphone" : "trunk");
        return token;
    }

    public OutboundCall? Claim(string token) => _pending.TryRemove(token, out var value) ? value.Call : null;
}
