using Channel = System.Threading.Channels.Channel;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using Google.Protobuf;
using Grpc.Core;
using Switchboard.Contracts.V1;
using Switchboard.Domain;

namespace Switchboard.Gateway;

/// <summary>The dependencies shared by every call.</summary>
public sealed record CallServices(
    AriClient Ari,
    ConfigCache Configs,
    CostMatrix Matrix,
    EmergencyLexicon Lexicon,
    Inference.InferenceClient Inference,
    EventPublisher Events,
    ContextSigner Signer,
    PromptStore Prompts,
    TenantLimiter Limiter,
    StasisTracker Stasis,
    LlmBudget Budget,
    SpeculationOptions Speculation,
    ILogger<CallManager> Log);

/// <summary>Ablation L4: start one LLM call on a stable partial. Off unless configured.</summary>
public sealed record SpeculationOptions(bool Enabled, EarlyCommitMode EarlyCommit = EarlyCommitMode.Off);

/// <summary>Ablation L3: stop listening before end of speech. Saves time, and risks cutting callers off.</summary>
public enum EarlyCommitMode { Off, Stable, EmergencyOnly }

/// <summary>ARI returns from externalMedia before the channel enters Stasis, so bridging must wait for the event.</summary>
public sealed class StasisTracker
{
    private readonly ConcurrentDictionary<string, TaskCompletionSource> _entered = new();

    private TaskCompletionSource For(string channel) => _entered.GetOrAdd(channel, _ => new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously));

    public void Entered(string channel) => For(channel).TrySetResult();

    public async Task WaitAsync(string channel, TimeSpan timeout, CancellationToken ct)
    {
        try
        {
            await For(channel).Task.WaitAsync(timeout, ct);
        }
        finally
        {
            _entered.TryRemove(channel, out _);
        }
    }
}

/// <summary>Tracks live calls, and links ARI events and media sockets to them by channel id.</summary>
public sealed class CallManager(CallServices services)
{
    private readonly ConcurrentDictionary<string, CallSession> _calls = new();

    public int Active => _calls.Count;

    public void OnStasisStart(JsonElement e)
    {
        var channel = e.GetProperty("channel");
        var id = channel.GetProperty("id").GetString()!;
        if (channel.GetProperty("name").GetString()!.StartsWith("WebSocket/", StringComparison.Ordinal))
        {
            services.Stasis.Entered(id);
            return;
        }

        var did = e.GetProperty("args").EnumerateArray().Select(a => a.GetString()).FirstOrDefault() ?? "";
        var session = new CallSession(id, did, services);
        if (_calls.TryAdd(id, session))
        {
            _ = Task.Run(async () =>
            {
                try
                {
                    await session.RunAsync();
                }
                finally
                {
                    _calls.TryRemove(id, out _);
                }
            });
        }
    }

    public void OnEnded(string channelId)
    {
        if (_calls.TryGetValue(channelId, out var session)) session.CallerGone();
    }

    public void OnDtmf(string channelId, string digit)
    {
        if (_calls.TryGetValue(channelId, out var session)) session.Dtmf(digit);
    }

    public async Task HandleMediaAsync(string callId, WebSocket socket)
    {
        if (_calls.TryGetValue(callId, out var session)) await session.RunMediaAsync(socket);
        else await socket.CloseAsync(WebSocketCloseStatus.PolicyViolation, "unknown call", CancellationToken.None);
    }
}

/// <summary>One caller's state machine: answer, greet, listen, decide, act. Always ends somewhere safe.</summary>
public sealed class CallSession(string channelId, string did, CallServices s)
{
    private const int MaxReprompts = 1;
    // Long enough for a slow CPU Whisper to answer: a late route beats losing what the caller said.
    private static readonly TimeSpan TurnTimeout = TimeSpan.FromSeconds(25);
    private static readonly TimeSpan LlmDeadline = TimeSpan.FromMilliseconds(550);
    private static readonly TimeSpan InferenceTurnDeadline = TimeSpan.FromSeconds(30);

    // Bounded: it drops with a counter rather than growing.
    private readonly Channel<byte[]> _audio = Channel.CreateBounded<byte[]>(new BoundedChannelOptions(500) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true });
    private readonly Channel<string> _digits = Channel.CreateUnbounded<string>();
    private readonly TaskCompletionSource<WebSocket> _media = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource _mediaClosed = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly CancellationTokenSource _gone = new();
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private volatile bool _inTurn;

    public void CallerGone() => _gone.Cancel();

    public void Dtmf(string digit) => _digits.Writer.TryWrite(digit);

    public async Task RunMediaAsync(WebSocket socket)
    {
        var buffer = new byte[4096];
        var started = false;
        try
        {
            using var message = new MemoryStream();
            while (socket.State == WebSocketState.Open)
            {
                var result = await socket.ReceiveAsync(buffer, _gone.Token);
                if (result.MessageType == WebSocketMessageType.Close) break;

                message.Write(buffer, 0, result.Count);
                if (!result.EndOfMessage) continue;

                if (result.MessageType == WebSocketMessageType.Binary)
                {
                    if (_inTurn) _audio.Writer.TryWrite(message.ToArray());
                }
                else if (!started && Encoding.UTF8.GetString(message.GetBuffer(), 0, (int)message.Length).Contains("MEDIA_START", StringComparison.Ordinal))
                {
                    started = true;
                    _media.TrySetResult(socket);
                }

                message.SetLength(0);
            }
        }
        catch (Exception e) when (e is OperationCanceledException or WebSocketException)
        {
        }
        finally
        {
            _media.TrySetCanceled();
            _mediaClosed.TrySetResult();
        }
    }

    private async Task SendPromptAsync(WebSocket socket, byte[] ulaw)
    {
        await _sendLock.WaitAsync(_gone.Token);
        try
        {
            // 160 bytes is the optimal ulaw frame; whole multiples leave nothing to be dropped.
            const int Chunk = 8000;
            for (var i = 0; i < ulaw.Length; i += Chunk)
            {
                await socket.SendAsync(ulaw.AsMemory(i, Math.Min(Chunk, ulaw.Length - i)), WebSocketMessageType.Binary, true, _gone.Token);
            }
        }
        finally
        {
            _sendLock.Release();
        }
    }

    private async Task SendCommandAsync(WebSocket socket, string command)
    {
        await _sendLock.WaitAsync(_gone.Token);
        try
        {
            await socket.SendAsync(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new { command })), WebSocketMessageType.Text, true, _gone.Token);
        }
        finally
        {
            _sendLock.Release();
        }
    }

    public async Task RunAsync()
    {
        var tenant = s.Configs.Resolve("did", did);
        if (tenant is null || s.Configs.Get(tenant) is not { } entry)
        {
            s.Log.LogWarning("no tenant for the dialled number: hanging up");
            await s.Ari.HangupAsync(channelId);
            return;
        }

        if (!s.Limiter.TryEnter(tenant))
        {
            await s.Ari.ContinueAsync(channelId, "departments", entry.Config.DepartmentById(entry.Config.Fallback.Department).Target, CancellationToken.None);
            return;
        }

        string? bridge = null;
        string? mediaChannel = null;
        var conversation = "call-" + channelId;
        var log = new ConversationLog(s.Events, tenant, conversation, entry.Version, Contracts.V1.Channel.Voice);
        using var trace = Tracing.StartConversation(tenant, conversation, entry.Version, "voice");
        try
        {
            var ct = _gone.Token;
            log.Emit(e => e.Started = new ConversationStarted { EntryPoint = did });
            await s.Ari.AnswerAsync(channelId, ct);
            bridge = await s.Ari.CreateBridgeAsync(ct);
            mediaChannel = await s.Ari.CreateExternalMediaAsync(channelId, ct);
            await s.Stasis.WaitAsync(mediaChannel, TimeSpan.FromSeconds(5), ct);
            await s.Ari.AddToBridgeAsync(bridge, $"{channelId},{mediaChannel}", ct);
            var socket = await _media.Task.WaitAsync(TimeSpan.FromSeconds(5), ct);

            var language = entry.Config.Languages.FirstOrDefault() ?? "en";
            await SendPromptAsync(socket, s.Prompts.Get(tenant, language, "greeting"));
            var (decision, stages) = await ConverseAsync(entry.Config, entry.Version, conversation, tenant, language, socket, log);

            var target = DecisionFlow.TargetFor(entry.Config, decision);
            var routeClock = Stopwatch.StartNew();
            using var routeSpan = Tracing.StartStage("route_command", tenant);
            await s.Ari.DeleteBridgeAsync(bridge);
            bridge = null;
            await s.Ari.HangupAsync(mediaChannel);
            mediaChannel = null;
            await s.Ari.ContinueAsync(channelId, "departments", target, ct);
            stages["route_cmd"] = routeClock.Elapsed.TotalMilliseconds;
            routeSpan?.Stop();
            trace?.SetTag("department", decision.Department);
            trace?.SetTag("decision.reason", decision.Reason);
            trace?.SetTag("decision.expected_cost", decision.ExpectedCost);
            foreach (var (stage, ms) in stages) trace?.SetTag($"stage_ms.{stage}", ms);

            log.Emit(e =>
            {
                e.Decision = new DecisionMade { Action = decision.Action.ToString().ToLowerInvariant(), DepartmentId = decision.Department ?? "", ExpectedCost = decision.ExpectedCost, Reason = decision.Reason };
                e.Decision.StageMs.Add(stages);
            });
            log.Emit(e => e.Routed = new RouteExecuted { DepartmentId = decision.Department ?? "", Target = target });
            // Metadata only: no transcript text is ever logged.
            s.Log.LogInformation("routed tenant={Tenant} conversation={Conversation} department={Department} reason={Reason} version={Version} listen_ms={ListenMs:F0} decision_ms={DecisionMs:F2} route_cmd_ms={RouteMs:F2}",
                tenant, conversation, decision.Department, decision.Reason, entry.Version, stages.GetValueOrDefault("listen"), stages.GetValueOrDefault("eos_to_decision"), stages["route_cmd"]);
        }
        catch (Exception e) when (_gone.IsCancellationRequested && e is OperationCanceledException or RpcException)
        {
            // The caller has gone, so there's nobody left to route. A cancelled stream isn't a failure.
            log.Emit(ev => ev.Ended = new ConversationEnded { Reason = "caller_hung_up" });
        }
        catch (Exception e)
        {
            // A caller always ends up somewhere: any failure here sends them to the tenant's fallback.
            s.Log.LogError("call failed, sending to fallback: {Error}: {Message}", e.GetType().Name, e.Message);
            try
            {
                await s.Ari.ContinueAsync(channelId, "departments", entry.Config.DepartmentById(entry.Config.Fallback.Department).Target, CancellationToken.None);
            }
            catch (Exception inner)
            {
                s.Log.LogError("fallback failed too: {Error}", inner.GetType().Name);
                await s.Ari.HangupAsync(channelId);
            }
        }
        finally
        {
            s.Limiter.Exit(tenant);
            if (mediaChannel is not null) await s.Ari.HangupAsync(mediaChannel);
            if (bridge is not null) await s.Ari.DeleteBridgeAsync(bridge);
        }
    }

    private async Task<(Decision Decision, Dictionary<string, double> Stages)> ConverseAsync(TenantConfig config, string version, string conversation, string tenant, string language, WebSocket socket, ConversationLog log)
    {
        var allowed = config.AllIntents.ToList();
        var context = s.Signer.Sign(tenant, conversation, version);
        var clarifyTurns = 0;
        var reprompts = 0;
        var degraded = false;
        var transcript = "";
        while (true)
        {
            Activity? listenSpan = Tracing.StartStage("listen", tenant);
            var listenClock = Stopwatch.StartNew();
            var turn = await ListenAsync(config, context, allowed, socket, log);
            var listenMs = listenClock.Elapsed.TotalMilliseconds;
            listenSpan?.Stop();
            degraded |= turn.Degraded;
            if (!string.IsNullOrEmpty(turn.Text)) transcript = (transcript + " " + turn.Text).Trim();

            var noSpeech = string.IsNullOrWhiteSpace(transcript) && turn.Digit is null;
            if (noSpeech && !turn.InferenceDown && reprompts < MaxReprompts)
            {
                reprompts++;
                await SendPromptAsync(socket, s.Prompts.Get(tenant, language, "reprompt"));
                continue;
            }

            var stages = new Dictionary<string, double> { ["asr_engine_ms"] = turn.EncoderMs, ["listen"] = listenMs };
            var clock = Stopwatch.StartNew();
            var llmMs = 0.0;
            using var decideSpan = Tracing.StartStage("decide", tenant);
            var decision = await DecisionFlow.ResolveAsync(new TurnContext
            {
                Config = config,
                Matrix = s.Matrix,
                Lexicon = s.Lexicon,
                At = DateTimeOffset.UtcNow,
                Transcript = transcript,
                Scores = turn.Scores,
                ClarifyTurns = clarifyTurns,
                Degraded = degraded || turn.InferenceDown,
                AsrConfidence = turn.AsrConfidence,
                LlmAvailable = !turn.InferenceDown && (turn.Speculative is not null || s.Budget.TryConsume(tenant)),
                Did = did,
                Dtmf = turn.Digit,
            }, async ctx =>
            {
                var llmClock = Stopwatch.StartNew();
                using var llmSpan = Tracing.StartStage("llm", tenant);
                try
                {
                    // A speculative answer is only good for the words it heard.
                    if (turn.Speculative is { } speculative && speculative.Transcript == transcript) return await speculative.Answer;

                    return await EscalateAsync(config, context, allowed, transcript, ctx.Scores, _gone.Token);
                }
                finally
                {
                    llmMs = llmClock.Elapsed.TotalMilliseconds;
                }
            });
            stages["eos_to_decision"] = clock.Elapsed.TotalMilliseconds;
            stages["llm"] = llmMs;

            if (decision.Action != ActionKind.Clarify) return (decision, stages);

            clarifyTurns++;
            var turnNumber = (uint)clarifyTurns;
            log.Emit(e =>
            {
                e.Clarify = new ClarificationAsked { Turn = turnNumber };
                e.Clarify.Candidates.AddRange(decision.Candidates);
            });
            // In degraded mode a question can't rely on ASR, so the prompt offers the keypad.
            var prompt = degraded || turn.InferenceDown ? "dtmf_menu" : "clarify";
            await SendPromptAsync(socket, s.Prompts.Get(tenant, language, prompt));
        }
    }

    private sealed record SpeculativeCall(string Transcript, Task<IReadOnlyDictionary<string, double>?> Answer);

    private sealed record TurnResult(string Text, Dictionary<string, double> Scores, bool Degraded, bool InferenceDown, string? Digit, double EncoderMs, SpeculativeCall? Speculative = null, double AsrConfidence = 0);

    private async Task<TurnResult> ListenAsync(TenantConfig config, TenantContext context, List<string> allowed, WebSocket socket, ConversationLog log)
    {
        var stability = new PartialStability();
        var asrConfidence = 0.0;
        SpeculativeCall? speculative = null;
        var latestPartial = "";
        using var cancelSpeculation = CancellationTokenSource.CreateLinkedTokenSource(_gone.Token);
        var text = "";
        var scores = new Dictionary<string, double>();
        var degraded = false;
        var encoderMs = 0.0;
        using var turn = CancellationTokenSource.CreateLinkedTokenSource(_gone.Token);
        turn.CancelAfter(TurnTimeout);
        var digitTask = _digits.Reader.ReadAsync(turn.Token).AsTask();
        var flushed = false;
        _inTurn = true;
        try
        {
            using var call = s.Inference.StreamTurn(deadline: DateTime.UtcNow + InferenceTurnDeadline, cancellationToken: turn.Token);
            var start = new TurnStart { Context = context, ReplayKey = did };
            start.AllowedIntents.AddRange(allowed);
            await call.RequestStream.WriteAsync(new StreamTurnRequest { Start = start }, turn.Token);

            var pump = Task.Run(async () =>
            {
                await foreach (var frame in _audio.Reader.ReadAllAsync(turn.Token))
                {
                    await call.RequestStream.WriteAsync(new StreamTurnRequest { Audio = new AudioChunk { Ulaw = ByteString.CopyFrom(frame) } }, turn.Token);
                }
            }, turn.Token);

            var read = Task.Run(async () =>
            {
                await foreach (var m in call.ResponseStream.ReadAllAsync(turn.Token))
                {
                    switch (m.PayloadCase)
                    {
                        case StreamTurnResponse.PayloadOneofCase.Transcript:
                            if (!flushed && m.Transcript.Text.Length > 0)
                            {
                                flushed = true;
                                await SendCommandAsync(socket, "FLUSH_MEDIA"); // Barge-in.
                            }

                            if (!m.Transcript.IsFinal) latestPartial = m.Transcript.Text;

                            if (m.Transcript.IsFinal)
                            {
                                text = (text + " " + m.Transcript.Text).Trim();
                                var final = m.Transcript;
                                asrConfidence = final.AsrConfidence;
                                log.Emit(e => e.Final = new TranscriptFinal { Text = final.Text, AsrConfidence = final.AsrConfidence });
                            }

                            break;
                        case StreamTurnResponse.PayloadOneofCase.Score:
                            scores = m.Score.Scores.ToDictionary(x => x.Intent, x => x.Probability);
                            encoderMs += m.Score.EncoderMs;
                            var stable = m.Score.FromFinal ? null : stability.Update(scores);
                            if (stable is not null && (s.Speculation.EarlyCommit == EarlyCommitMode.Stable || (s.Speculation.EarlyCommit == EarlyCommitMode.EmergencyOnly && stable == Intents.Emergency)))
                            {
                                text = (text + " " + latestPartial).Trim();
                                return;
                            }

                            if (s.Speculation.Enabled && stable is not null && speculative is null && s.Budget.TryConsume(context.TenantId))
                            {
                                // At most one per turn, cancelled if the words move on. It counts in the cost metric.
                                var words = latestPartial;
                                var seen = scores;
                                speculative = new SpeculativeCall(words, EscalateAsync(config, context, allowed, words, seen, cancelSpeculation.Token));
                            }

                            break;
                        case StreamTurnResponse.PayloadOneofCase.Degraded:
                            degraded = true;
                            var reason = m.Degraded.Reason;
                            log.Emit(e => e.Degraded = new DegradedMode { Reason = reason });
                            break;
                        case StreamTurnResponse.PayloadOneofCase.EndOfSpeech:
                            return;
                    }
                }
            }, turn.Token);

            var finished = await Task.WhenAny(read, digitTask);
            _inTurn = false;
            if (finished == digitTask && digitTask.IsCompletedSuccessfully)
            {
                await cancelSpeculation.CancelAsync();
                return new TurnResult(text, scores, degraded, false, digitTask.Result, encoderMs);
            }

            await read; // Surfaces a stream failure.
            await turn.CancelAsync();
            if (speculative is not null && speculative.Transcript != text)
            {
                await cancelSpeculation.CancelAsync();
                speculative = null;
            }

            if (speculative is not null)
            {
                // The caller's answer outlives this turn's token source only through this completed task.
                var answer = await speculative.Answer;
                speculative = speculative with { Answer = Task.FromResult(answer) };
            }

            return new TurnResult(text, scores, degraded, false, null, encoderMs, speculative, asrConfidence);
        }
        catch (RpcException) when (turn.IsCancellationRequested && !_gone.IsCancellationRequested)
        {
            return new TurnResult(text, scores, degraded, false, null, encoderMs); // The turn timed out: inference is fine.
        }
        catch (RpcException e) when (!_gone.IsCancellationRequested)
        {
            // Inference is down: rules and the keypad still work.
            s.Log.LogWarning("inference unavailable: {Status}", e.StatusCode);
            log.Emit(ev => ev.Degraded = new DegradedMode { Reason = "inference_unavailable" });
            return await WaitForDigitOrTimeoutAsync(digitTask, text, turn);
        }
        catch (OperationCanceledException) when (!_gone.IsCancellationRequested)
        {
            return new TurnResult(text, scores, degraded, false, null, encoderMs); // The turn timed out.
        }
        finally
        {
            _inTurn = false;
            while (_audio.Reader.TryRead(out _))
            {
            }
        }
    }

    private static async Task<TurnResult> WaitForDigitOrTimeoutAsync(Task<string> digitTask, string text, CancellationTokenSource turn)
    {
        try
        {
            return new TurnResult(text, [], true, true, await digitTask.WaitAsync(TimeSpan.FromSeconds(5), turn.Token), 0);
        }
        catch (Exception e) when (e is TimeoutException or OperationCanceledException)
        {
            return new TurnResult(text, [], true, true, null, 0);
        }
    }

    private async Task<IReadOnlyDictionary<string, double>?> EscalateAsync(TenantConfig config, TenantContext context, List<string> allowed, string transcript, IReadOnlyDictionary<string, double> scores, CancellationToken ct)
    {
        var request = new EscalateRequest { Context = context, Transcript = transcript, PromptOverride = config.PromptOverride };
        request.AllowedIntents.AddRange(allowed);
        request.EncoderScores.AddRange(scores.Select(kv => new IntentScore { Intent = kv.Key, Probability = kv.Value }));
        foreach (var custom in config.CustomIntents)
        {
            var c = new Contracts.V1.CustomIntent { Id = custom.Id, Description = custom.Description };
            c.Examples.AddRange(custom.Examples);
            request.CustomIntents.Add(c);
        }

        try
        {
            var response = await s.Inference.EscalateAsync(request, deadline: DateTime.UtcNow + LlmDeadline, cancellationToken: ct);
            return response.FusedScores.Count > 0 ? response.FusedScores.ToDictionary(x => x.Intent, x => x.Probability) : null;
        }
        catch (RpcException)
        {
            return null; // The deadline passed or LLM is down: carry on without it.
        }
    }
}

/// <summary>Holds the ARI event socket open and dispatches events, reconnecting if Asterisk restarts.</summary>
public sealed class AriEventLoop(IConfiguration config, CallManager calls, ILogger<AriEventLoop> log) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var baseUrl = (config["ARI_URL"] ?? "http://asterisk:8088").Replace("http", "ws", StringComparison.Ordinal);
        var key = Uri.EscapeDataString($"{config["ARI_USER"] ?? "switchboard"}:{config["ASTERISK_ARI_PASSWORD"]}");
        var uri = new Uri($"{baseUrl}/ari/events?app=switchboard&subscribeAll=false&api_key={key}");
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                using var socket = new ClientWebSocket();
                await socket.ConnectAsync(uri, stoppingToken);
                log.LogInformation("ARI event socket connected");
                var buffer = new byte[65536];
                using var message = new MemoryStream();
                while (socket.State == WebSocketState.Open)
                {
                    var result = await socket.ReceiveAsync(buffer, stoppingToken);
                    if (result.MessageType == WebSocketMessageType.Close) break;

                    message.Write(buffer, 0, result.Count);
                    if (!result.EndOfMessage) continue;

                    using var doc = JsonDocument.Parse(message.ToArray());
                    message.SetLength(0);
                    Dispatch(doc.RootElement);
                }
            }
            catch (Exception e) when (e is WebSocketException or HttpRequestException)
            {
                log.LogWarning("ARI socket unavailable: {Error}", e.GetType().Name);
            }

            await Task.Delay(TimeSpan.FromSeconds(1), stoppingToken);
        }
    }

    private void Dispatch(JsonElement e)
    {
        switch (e.GetProperty("type").GetString())
        {
            case "StasisStart":
                calls.OnStasisStart(e);
                break;
            case "ChannelDestroyed":
            case "ChannelHangupRequest":
                calls.OnEnded(e.GetProperty("channel").GetProperty("id").GetString()!);
                break;
            case "ChannelDtmfReceived":
                calls.OnDtmf(e.GetProperty("channel").GetProperty("id").GetString()!, e.GetProperty("digit").GetString()!);
                break;
        }
    }
}
