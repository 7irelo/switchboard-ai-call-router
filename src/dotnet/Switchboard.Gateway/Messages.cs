using Channel = System.Threading.Channels.Channel;
using System.Collections.Concurrent;
using System.Text.Json;
using System.Threading.Channels;
using Grpc.Core;
using Switchboard.Contracts.V1;
using Switchboard.Domain;
using Switchboard.Storage;

namespace Switchboard.Gateway;

public sealed record InboundMessage(string Tenant, string From, string MessageId, string? Text, string Type);

public sealed record OutboundReply(string Tenant, string To, string Text, string? Department);

/// <summary>Records replies, standing in for the provider's send API so tests can assert on them.</summary>
public sealed class MockOutbox
{
    private readonly ConcurrentQueue<OutboundReply> _sent = new();

    public void Send(OutboundReply reply) => _sent.Enqueue(reply);

    public IReadOnlyList<OutboundReply> For(string tenant) => [.. _sent.Where(r => r.Tenant == tenant)];
}

/// <summary>Takes verified webhooks after the 200 has gone back. Duplicates are dropped on message id.</summary>
public sealed class MessageProcessor(CallServicesForMessages s, ILogger<MessageProcessor> log) : BackgroundService
{
    private readonly Channel<InboundMessage> _queue = Channel.CreateBounded<InboundMessage>(10_000);
    private readonly ConcurrentDictionary<string, ThreadState> _threads = new();

    private sealed class ThreadState
    {
        public int ClarifyTurns;
        public string Transcript = "";
    }

    public bool Enqueue(InboundMessage message) => _queue.Writer.TryWrite(message);

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await foreach (var message in _queue.Reader.ReadAllAsync(stoppingToken))
        {
            try
            {
                await HandleAsync(message, stoppingToken);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                log.LogError("message handling failed: {Error}", e.GetType().Name);
                if (s.Configs.Get(message.Tenant) is { } entry)
                {
                    s.Outbox.Send(new OutboundReply(message.Tenant, message.From, "Sorry, something went wrong. A person will follow up.", entry.Config.Fallback.Department));
                }
            }
        }
    }

    private async Task HandleAsync(InboundMessage m, CancellationToken ct)
    {
        if (s.Configs.Get(m.Tenant) is not { } entry) return;
        if (!await s.Store.TryRecordMessageAsync(m.Tenant, m.MessageId, ct)) return; // At-least-once delivery: a duplicate.

        if (m.Type != "text" || string.IsNullOrWhiteSpace(m.Text))
        {
            s.Outbox.Send(new OutboundReply(m.Tenant, m.From, "Sorry, I can only read text messages at the moment. Please type your request.", null));
            return;
        }

        var key = $"{m.Tenant}:{m.From}";
        using var trace = Tracing.StartConversation(m.Tenant, "msg", entry.Version, "message");
        var thread = _threads.GetOrAdd(key, _ => new ThreadState());
        thread.Transcript = (thread.Transcript + " " + m.Text).Trim();
        var conversation = "msg-" + Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(key)))[..16];
        var events = new ConversationLog(s.Events, m.Tenant, conversation, entry.Version, Contracts.V1.Channel.Message);
        events.Emit(e => e.Message = new MessageReceived { ProviderMessageId = m.MessageId, Text = m.Text });

        var allowed = entry.Config.AllIntents.ToList();
        var context = s.Signer.Sign(m.Tenant, conversation, entry.Version);
        Dictionary<string, double> scores = [];
        var inferenceDown = false;
        try
        {
            var request = new ClassifyTextRequest { Context = context, Text = thread.Transcript };
            request.AllowedIntents.AddRange(allowed);
            var response = await s.Inference.ClassifyTextAsync(request, deadline: DateTime.UtcNow.AddSeconds(2), cancellationToken: ct);
            scores = response.Scores.ToDictionary(x => x.Intent, x => x.Probability);
        }
        catch (RpcException)
        {
            inferenceDown = true;
        }

        var decision = await DecisionFlow.ResolveAsync(new TurnContext
        {
            Config = entry.Config,
            Matrix = s.Matrix,
            Lexicon = s.Lexicon,
            At = DateTimeOffset.UtcNow,
            Transcript = thread.Transcript,
            Scores = scores,
            ClarifyTurns = thread.ClarifyTurns,
            Degraded = inferenceDown,
            LlmAvailable = !inferenceDown && s.Budget.TryConsume(m.Tenant),
        }, async ctx =>
        {
            try
            {
                var request = new EscalateRequest { Context = context, Transcript = thread.Transcript, PromptOverride = entry.Config.PromptOverride };
                request.AllowedIntents.AddRange(allowed);
                request.EncoderScores.AddRange(ctx.Scores.Select(kv => new IntentScore { Intent = kv.Key, Probability = kv.Value }));
                var response = await s.Inference.EscalateAsync(request, deadline: DateTime.UtcNow.AddSeconds(5), cancellationToken: ct);
                return response.FusedScores.Count > 0 ? response.FusedScores.ToDictionary(x => x.Intent, x => x.Probability) : null;
            }
            catch (RpcException)
            {
                return null;
            }
        });

        events.Emit(e => e.Decision = new DecisionMade { Action = decision.Action.ToString().ToLowerInvariant(), DepartmentId = decision.Department ?? "", ExpectedCost = decision.ExpectedCost, Reason = decision.Reason });
        if (decision.Action == ActionKind.Clarify)
        {
            thread.ClarifyTurns++;
            var options = string.Join(" or ", decision.Candidates.Select(c => c.Replace('_', ' ')));
            s.Outbox.Send(new OutboundReply(m.Tenant, m.From, $"Could you tell me a bit more? Is this about {options}?", null));
            return;
        }

        _threads.TryRemove(key, out _);
        events.Emit(e => e.Routed = new RouteExecuted { DepartmentId = decision.Department ?? "", Target = DecisionFlow.TargetFor(entry.Config, decision) });
        s.Outbox.Send(new OutboundReply(m.Tenant, m.From, $"Thanks. I've passed this to {decision.Department?.Replace('_', ' ')}.", decision.Department));
    }
}

public sealed record CallServicesForMessages(
    ConfigCache Configs,
    ConfigStore Store,
    CostMatrix Matrix,
    EmergencyLexicon Lexicon,
    Inference.InferenceClient Inference,
    EventPublisher Events,
    ContextSigner Signer,
    MockOutbox Outbox,
    LlmBudget Budget);

/// <summary>Parses the WhatsApp-style webhook shape. Unknown phone numbers yield nothing, so they can't reach a tenant.</summary>
public static class WebhookParser
{
    public static List<InboundMessage> Parse(JsonElement root, Func<string, string?> resolveTenant)
    {
        var messages = new List<InboundMessage>();
        foreach (var entry in root.GetProperty("entry").EnumerateArray())
        {
            foreach (var change in entry.GetProperty("changes").EnumerateArray())
            {
                var value = change.GetProperty("value");
                // The tenant comes from the trusted phone_number_id, never from anything the sender controls.
                if (!value.TryGetProperty("metadata", out var metadata)) continue;

                var tenant = resolveTenant(metadata.GetProperty("phone_number_id").GetString() ?? "");
                if (tenant is null || !value.TryGetProperty("messages", out var list)) continue;

                foreach (var m in list.EnumerateArray())
                {
                    var type = m.GetProperty("type").GetString() ?? "unknown";
                    var text = type == "text" ? m.GetProperty("text").GetProperty("body").GetString() : null;
                    messages.Add(new InboundMessage(tenant, m.GetProperty("from").GetString() ?? "", m.GetProperty("id").GetString() ?? "", text, type));
                }
            }
        }

        return messages;
    }
}
