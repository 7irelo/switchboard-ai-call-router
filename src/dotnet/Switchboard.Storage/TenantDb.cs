using System.Data.Common;
using Dapper;
using Npgsql;

namespace Switchboard.Storage;

/// <summary>
/// The only way to reach tenant data. Every unit of work runs in one transaction with
/// SET LOCAL app.tenant_id, so nothing survives on a pooled connection. Architecture tests
/// reject direct connection use anywhere else.
/// </summary>
public sealed class TenantDb(NpgsqlDataSource dataSource)
{
    public async Task<T> InTenantAsync<T>(string tenantId, Func<DbConnection, DbTransaction, Task<T>> work, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(tenantId)) throw new ArgumentException("a tenant is required: failing closed", nameof(tenantId));

        await using var connection = await dataSource.OpenConnectionAsync(ct);
        await using var transaction = await connection.BeginTransactionAsync(ct);
        // is_local = true is SET LOCAL: the value ends with the transaction.
        await connection.ExecuteAsync("SELECT set_config('app.tenant_id', @tenantId, true)", new { tenantId }, transaction);
        var result = await work(connection, transaction);
        await transaction.CommitAsync(ct);
        return result;
    }

    public Task InTenantAsync(string tenantId, Func<DbConnection, DbTransaction, Task> work, CancellationToken ct = default) =>
        InTenantAsync<object?>(tenantId, async (c, t) =>
        {
            await work(c, t);
            return null;
        }, ct);

    /// <summary>Resolves a tenant from a trusted channel: a dialled number or a WhatsApp phone_number_id.</summary>
    public Task<string?> ResolveChannelAsync(string kind, string value, CancellationToken ct = default) =>
        ScalarAsync("SELECT resolve_channel(@kind, @value)", new { kind, value }, ct);

    public Task<string?> AuthenticateAsync(string keyHash, CancellationToken ct = default) =>
        ScalarAsync("SELECT authenticate_key(@keyHash)", new { keyHash }, ct);

    public async Task<IReadOnlyList<string>> ListTenantIdsAsync(CancellationToken ct = default)
    {
        await using var connection = await dataSource.OpenConnectionAsync(ct);
        return [.. await connection.QueryAsync<string>("SELECT list_tenant_ids()")];
    }

    private async Task<string?> ScalarAsync(string sql, object args, CancellationToken ct)
    {
        await using var connection = await dataSource.OpenConnectionAsync(ct);
        return await connection.ExecuteScalarAsync<string?>(sql, args);
    }
}
