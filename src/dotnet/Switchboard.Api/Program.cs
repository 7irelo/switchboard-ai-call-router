using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Npgsql;
using Switchboard.Domain;
using Switchboard.Storage;

var builder = WebApplication.CreateBuilder(args);
builder.Logging.AddJsonConsole();

var appConnection = builder.Configuration["SWITCHBOARD_APP_DB"] ?? throw new InvalidOperationException("SWITCHBOARD_APP_DB is required");
var matrixPath = builder.Configuration["COST_MATRIX_PATH"] ?? "configs/cost_matrix.yaml";
var adminKey = builder.Configuration["SWITCHBOARD_ADMIN_KEY"];

builder.Services.AddSingleton(_ => NpgsqlDataSource.Create(appConnection));
builder.Services.AddSingleton<TenantDb>();
builder.Services.AddSingleton<ConfigStore>();
builder.Services.AddSingleton(_ => CostMatrix.FromYaml(File.ReadAllText(matrixPath)));
builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    o.SerializerOptions.DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
});
builder.Services.AddOpenApi();

if (builder.Configuration["SWITCHBOARD_ADMIN_DB"] is { } adminDb)
{
    await Migrator.ApplyAsync(adminDb, builder.Configuration["SWITCHBOARD_APP_PASSWORD"] ?? throw new InvalidOperationException("SWITCHBOARD_APP_PASSWORD is required"));
}

var app = builder.Build();
app.MapOpenApi();
app.MapGet("/healthz", () => Results.Ok(new { status = "ok" })).WithMetadata(new PublicRoute());

// Tenant-scoped API. The tenant comes only from the credential, never from the request.
var v1 = app.MapGroup("/v1").AddEndpointFilter(async (ctx, next) =>
{
    var db = ctx.HttpContext.RequestServices.GetRequiredService<TenantDb>();
    var tenant = await Auth.TenantFromBearerAsync(ctx.HttpContext.Request, db);
    if (tenant is null) return Results.Unauthorized();
    ctx.HttpContext.Items[Auth.TenantKey] = tenant;
    return await next(ctx);
});

static string T(HttpContext ctx) => (string)ctx.Items[Auth.TenantKey]!;

static IResult OneOr404<TRow>(TRow? row) => row is null ? Results.NotFound() : Results.Ok(row);

static IResult Affected(int rows) => rows == 0 ? Results.NotFound() : Results.NoContent();

v1.MapGet("/departments", async (HttpContext c, ConfigStore s) => Results.Ok(await s.ListDepartmentsAsync(T(c))));
v1.MapGet("/departments/{departmentId}", async (string departmentId, HttpContext c, ConfigStore s) => OneOr404(await s.GetDepartmentAsync(T(c), departmentId)));
v1.MapPost("/departments", async (DepartmentRow d, HttpContext c, ConfigStore s) =>
{
    await s.CreateDepartmentAsync(T(c), d);
    return Results.Created($"/v1/departments/{d.Id}", d);
});
v1.MapPut("/departments/{departmentId}", async (string departmentId, DepartmentRow d, HttpContext c, ConfigStore s) =>
    Affected(await s.UpdateDepartmentAsync(T(c), d with { Id = departmentId })));
v1.MapDelete("/departments/{departmentId}", async (string departmentId, HttpContext c, ConfigStore s) => Affected(await s.DeleteDepartmentAsync(T(c), departmentId)));

v1.MapGet("/schedules", async (HttpContext c, ConfigStore s) => Results.Ok(await s.ListScheduleIdsAsync(T(c))));
v1.MapGet("/schedules/{scheduleId}", async (string scheduleId, HttpContext c, ConfigStore s) =>
    await s.GetScheduleAsync(T(c), scheduleId) is { } json ? Results.Content(json, "application/json") : Results.NotFound());
v1.MapPut("/schedules/{scheduleId}", async (string scheduleId, JsonElement windows, HttpContext c, ConfigStore s) =>
    Affected(await s.UpdateScheduleAsync(T(c), scheduleId, windows.GetRawText())));

v1.MapGet("/intent-map", async (HttpContext c, ConfigStore s) => Results.Ok(await s.GetIntentMapAsync(T(c))));
v1.MapPut("/intent-map", async (Dictionary<string, string> map, HttpContext c, ConfigStore s) =>
{
    try
    {
        await s.SetIntentMapAsync(T(c), map);
        return Results.NoContent();
    }
    catch (PostgresException e) when (e.SqlState == PostgresErrorCodes.ForeignKeyViolation)
    {
        return Results.BadRequest(new { error = "unknown department" });
    }
});

v1.MapGet("/rules", async (HttpContext c, ConfigStore s) => Results.Ok(await s.ListRulesAsync(T(c))));
v1.MapGet("/rules/{ruleId}", async (string ruleId, HttpContext c, ConfigStore s) => OneOr404(await s.GetRuleAsync(T(c), ruleId)));
v1.MapPost("/rules", async (RuleRow r, HttpContext c, ConfigStore s) =>
{
    try
    {
        await s.CreateRuleAsync(T(c), r);
        return Results.Created($"/v1/rules/{r.Id}", r);
    }
    catch (PostgresException e) when (e.SqlState == PostgresErrorCodes.ForeignKeyViolation)
    {
        return Results.BadRequest(new { error = "unknown department" });
    }
});
v1.MapDelete("/rules/{ruleId}", async (string ruleId, HttpContext c, ConfigStore s) => Affected(await s.DeleteRuleAsync(T(c), ruleId)));

v1.MapGet("/prompt-override", async (HttpContext c, ConfigStore s) => Results.Ok(new { text = await s.GetPromptOverrideAsync(T(c)) }));
v1.MapPut("/prompt-override", async (PromptOverride body, HttpContext c, ConfigStore s) =>
{
    if (body.Text.Length > TenantConfig.MaxPromptOverrideChars) return Results.BadRequest(new { error = "too long" });
    return Affected(await s.SetPromptOverrideAsync(T(c), body.Text));
});

v1.MapPost("/config/publish", async (HttpContext c, ConfigStore s, CostMatrix m) =>
{
    var (version, errors) = await s.PublishAsync(T(c), m);
    return errors.Count > 0 ? Results.UnprocessableEntity(new { errors }) : Results.Ok(new { version });
});
v1.MapGet("/config/versions/{version:int}", async (int version, HttpContext c, ConfigStore s) =>
    await s.GetVersionAsync(T(c), version) is { } json ? Results.Content(json, "application/json") : Results.NotFound());

v1.MapGet("/conversations", async (HttpContext c, ConfigStore s) => Results.Ok(await s.ListConversationsAsync(T(c))));
v1.MapGet("/conversations/{conversationId}", async (string conversationId, HttpContext c, ConfigStore s) => OneOr404(await s.GetConversationAsync(T(c), conversationId)));
v1.MapGet("/conversations/{conversationId}/events", async (string conversationId, HttpContext c, ConfigStore s) =>
    await s.GetConversationAsync(T(c), conversationId) is null ? Results.NotFound() : Results.Ok(await s.ListEventsAsync(T(c), conversationId)));
v1.MapGet("/conversations/{conversationId}/transcript", async (string conversationId, HttpContext c, ConfigStore s) =>
    await s.GetConversationAsync(T(c), conversationId) is null ? Results.NotFound() : Results.Ok(await s.ListTranscriptsAsync(T(c), conversationId)));
v1.MapDelete("/transcripts", async (HttpContext c, ConfigStore s) => Results.Ok(new { deleted = await s.DeleteTranscriptsAsync(T(c)) }));

// Provisioning uses a separate operator credential and is the only unscoped entry point.
app.MapPost("/admin/tenants", async (HttpRequest req, TenantConfig config, ConfigStore s, CostMatrix m) =>
{
    if (string.IsNullOrEmpty(adminKey) || !Auth.BearerEquals(req, adminKey)) return Results.Unauthorized();
    var errors = config.Validate(m);
    if (errors.Count > 0) return Results.UnprocessableEntity(new { errors });

    var apiKey = "sbk_" + Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(24));
    try
    {
        await s.ProvisionAsync(config, Auth.Hash(apiKey));
    }
    catch (PostgresException e) when (e.SqlState == PostgresErrorCodes.UniqueViolation)
    {
        return Results.Conflict(new { error = "tenant exists" });
    }

    var (version, publishErrors) = await s.PublishAsync(config.TenantId, m);
    return publishErrors.Count > 0 ? Results.UnprocessableEntity(new { errors = publishErrors }) : Results.Ok(new { api_key = apiKey, version });
}).WithMetadata(new AdminRoute());

app.Run();

public sealed record PromptOverride(string Text);

public sealed record PublicRoute;

public sealed record AdminRoute;

public static class Auth
{
    public const string TenantKey = "switchboard.tenant";

    public static string Hash(string key) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(key)));

    private static string? Bearer(HttpRequest request)
    {
        var header = request.Headers.Authorization.ToString();
        return header.StartsWith("Bearer ", StringComparison.Ordinal) ? header["Bearer ".Length..] : null;
    }

    public static async Task<string?> TenantFromBearerAsync(HttpRequest request, TenantDb db) =>
        Bearer(request) is { } key ? await db.AuthenticateAsync(Hash(key)) : null;

    public static bool BearerEquals(HttpRequest request, string expected) =>
        Bearer(request) is { } key && CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(key), Encoding.UTF8.GetBytes(expected));
}

public partial class Program;
