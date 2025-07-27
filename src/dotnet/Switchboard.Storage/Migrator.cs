using System.Reflection;
using Dapper;
using Npgsql;

namespace Switchboard.Storage;

/// <summary>Applies embedded SQL migrations with a privileged connection, then sets the app role's password.</summary>
public static class Migrator
{
    public static async Task ApplyAsync(string adminConnectionString, string appPassword)
    {
        await using var connection = new NpgsqlConnection(adminConnectionString);
        await connection.OpenAsync();
        await connection.ExecuteAsync("SELECT pg_advisory_lock(727001)");
        await connection.ExecuteAsync("CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())");

        var assembly = Assembly.GetExecutingAssembly();
        foreach (var resource in assembly.GetManifestResourceNames().Where(n => n.EndsWith(".sql", StringComparison.Ordinal)).Order(StringComparer.Ordinal))
        {
            if (await connection.ExecuteScalarAsync<int>("SELECT count(*) FROM schema_migrations WHERE name = @resource", new { resource }) > 0) continue;

            await using var stream = assembly.GetManifestResourceStream(resource)!;
            using var reader = new StreamReader(stream);
            var sql = await reader.ReadToEndAsync();
            await using var tx = await connection.BeginTransactionAsync();
            await connection.ExecuteAsync(sql, transaction: tx);
            await connection.ExecuteAsync("INSERT INTO schema_migrations (name) VALUES (@resource)", new { resource }, tx);
            await tx.CommitAsync();
        }

        // ALTER ROLE can't take parameters, so the password is quoted by the server.
        var quoted = await connection.ExecuteScalarAsync<string>("SELECT quote_literal(@appPassword)", new { appPassword });
        await connection.ExecuteAsync($"ALTER ROLE switchboard_app PASSWORD {quoted}");
        await connection.ExecuteAsync("SELECT pg_advisory_unlock(727001)");
    }
}
