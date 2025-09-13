using System.Text.Json;
using Grpc.Net.Client;
using Npgsql;
using Switchboard.Contracts.V1;
using Switchboard.Domain;
using Switchboard.Gateway;
using Switchboard.Storage;

var builder = WebApplication.CreateBuilder(args);
builder.Logging.AddJsonConsole();
var cfg = builder.Configuration;
var appDb = cfg["SWITCHBOARD_APP_DB"] ?? throw new InvalidOperationException("SWITCHBOARD_APP_DB is required");
var webhookSecret = cfg["WHATSAPP_APP_SECRET"] ?? throw new InvalidOperationException("WHATSAPP_APP_SECRET is required");
var configDir = cfg["CONFIG_DIR"] ?? "configs";

builder.Services.AddSingleton(_ => NpgsqlDataSource.Create(appDb));
builder.Services.AddSingleton<TenantDb>();
builder.Services.AddSingleton<ConfigStore>();
builder.Services.AddSingleton<ConfigCache>();
builder.Services.AddHostedService(p => p.GetRequiredService<ConfigCache>());
builder.Services.AddSingleton(_ => CostMatrix.FromYaml(File.ReadAllText(Path.Combine(configDir, "cost_matrix.yaml"))));
builder.Services.AddSingleton(_ => EmergencyLexicon.FromYaml(File.ReadAllText(Path.Combine(configDir, "emergency_lexicon.yaml"))));
builder.Services.AddSingleton(_ => new Inference.InferenceClient(GrpcChannel.ForAddress(cfg["INFERENCE_URL"] ?? "http://inference:50051")));
builder.Services.AddSingleton<EventPublisher>();
builder.Services.AddHostedService(p => p.GetRequiredService<EventPublisher>());
builder.Services.AddSingleton<ContextSigner>();
builder.Services.AddSingleton<PromptStore>();
builder.Services.AddSingleton<TenantLimiter>();
builder.Services.AddSingleton<StasisTracker>();
builder.Services.AddHttpClient<AriClient>();
builder.Services.AddSingleton<CallServices>();
builder.Services.AddSingleton<CallManager>();
builder.Services.AddHostedService<AriEventLoop>();
builder.Services.AddSingleton<MockOutbox>();
builder.Services.AddSingleton<CallServicesForMessages>();
builder.Services.AddSingleton<MessageProcessor>();
builder.Services.AddHostedService(p => p.GetRequiredService<MessageProcessor>());

var app = builder.Build();
app.UseWebSockets();

app.MapGet("/healthz", (ConfigCache c, EventPublisher e, CallManager calls) =>
    c.Ready ? Results.Ok(new { status = "ok", active_calls = calls.Active, dropped_events = e.Dropped }) : Results.StatusCode(503));

app.Map("/media", async (HttpContext ctx, CallManager calls) =>
{
    var callId = ctx.Request.Query["call_id"].ToString();
    if (!ctx.WebSockets.IsWebSocketRequest || callId.Length == 0)
    {
        ctx.Response.StatusCode = 400;
        return;
    }

    using var socket = await ctx.WebSockets.AcceptWebSocketAsync("media");
    await calls.HandleMediaAsync(callId, socket);
});

app.MapPost("/webhooks/whatsapp", async (HttpContext ctx, ConfigCache configs, MessageProcessor processor) =>
{
    using var buffer = new MemoryStream();
    await ctx.Request.Body.CopyToAsync(buffer);
    var body = buffer.ToArray();
    if (!WebhookSignature.IsValid(webhookSecret, body, ctx.Request.Headers["X-Hub-Signature-256"].FirstOrDefault())) return Results.Unauthorized();

    try
    {
        using var doc = JsonDocument.Parse(body);
        foreach (var message in WebhookParser.Parse(doc.RootElement, phone => configs.Resolve("whatsapp", phone))) processor.Enqueue(message);
    }
    catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException)
    {
        return Results.BadRequest();
    }

    return Results.Ok(); // Answer quickly; processing is asynchronous.
});

// Test-only surface for the mock provider. It's enabled by configuration and is never on by default.
if (cfg["ENABLE_MOCK_OUTBOX"] == "true")
{
    app.MapGet("/mock/outbox/{tenant}", (string tenant, MockOutbox outbox) => Results.Ok(outbox.For(tenant)));
}

app.Run();
