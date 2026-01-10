"""Remove one isolation or safety control at a time and require a test to go red.

This is what lets the repo claim its tests work. Each mutant runs in a throwaway copy, so the
working tree is never touched. A mutant that survives means a control has no real test: exit 1.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIG = "src/dotnet/Switchboard.Storage/Migrations/001_init.sql"
DB = "src/dotnet/Switchboard.Storage/TenantDb.cs"
API = "src/dotnet/Switchboard.Api/Program.cs"
ENGINE = "src/dotnet/Switchboard.Domain/DecisionEngine.cs"
CONFIG = "src/dotnet/Switchboard.Domain/TenantConfig.cs"
MATRIX = "src/dotnet/Switchboard.Domain/CostMatrix.cs"
ISO = "tests/dotnet/Switchboard.Tests.Isolation"
UNIT = "tests/dotnet/Switchboard.Tests.Unit"
ARCH = "tests/dotnet/Switchboard.Tests.Architecture"


@dataclass(frozen=True)
class Mutant:
    name: str
    file: str
    old: str
    new: str
    tests: str


MUTANTS = [
    Mutant(
        "drop FORCE RLS",
        MIG,
        "    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);\n",
        "",
        ISO,
    ),
    Mutant(
        "drop ENABLE RLS",
        MIG,
        "    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);\n",
        "",
        ISO,
    ),
    Mutant(
        "policy allows every row",
        MIG,
        "USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))",
        "USING (true) WITH CHECK (true)",
        ISO,
    ),
    Mutant(
        "policy lacks WITH CHECK",
        MIG,
        " WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))",
        " WITH CHECK (true)",
        ISO,
    ),
    Mutant(
        "one table left out of RLS",
        MIG,
        "'conversation_events', 'transcripts', 'message_dedup']",
        "'conversation_events', 'message_dedup']",
        ISO,
    ),
    Mutant(
        "app role gets BYPASSRLS",
        MIG,
        "CREATE ROLE switchboard_app LOGIN NOBYPASSRLS NOSUPERUSER",
        "CREATE ROLE switchboard_app LOGIN BYPASSRLS NOSUPERUSER",
        ISO,
    ),
    Mutant(
        "foreign key loses tenant_id",
        MIG,
        "  FOREIGN KEY (tenant_id, department_id) REFERENCES departments (tenant_id, id) ON DELETE CASCADE\n);\n\nCREATE TABLE routing_rules",
        "  department_ref int\n);\n\nCREATE TABLE routing_rules",
        ISO,
    ),
    Mutant(
        "tenant setting survives the transaction (SET, not SET LOCAL)",
        DB,
        "SELECT set_config('app.tenant_id', @tenantId, true)",
        "SELECT set_config('app.tenant_id', @tenantId, false)",
        ISO,
    ),
    Mutant(
        "tenant scope never set",
        DB,
        """await connection.ExecuteAsync("SELECT set_config('app.tenant_id', @tenantId, true)", new { tenantId }, transaction);""",
        "",
        ISO,
    ),
    Mutant(
        "missing tenant no longer fails closed",
        DB,
        """if (string.IsNullOrWhiteSpace(tenantId)) throw new ArgumentException("a tenant is required: failing closed", nameof(tenantId));""",
        "",
        ISO,
    ),
    Mutant(
        "API trusts a caller-supplied tenant header",
        API,
        "ctx.HttpContext.Items[Auth.TenantKey] = tenant;",
        """ctx.HttpContext.Items[Auth.TenantKey] = ctx.HttpContext.Request.Headers["X-Tenant-Id"].FirstOrDefault() ?? tenant;""",
        ISO,
    ),
    Mutant(
        "API skips authentication",
        API,
        "if (tenant is null) return Results.Unauthorized();",
        """tenant ??= "thornveld_vet";""",
        ISO,
    ),
    Mutant(
        "API leaks existence with 403",
        API,
        "static IResult OneOr404<TRow>(TRow? row) => row is null ? Results.NotFound() : Results.Ok(row);",
        "static IResult OneOr404<TRow>(TRow? row) => row is null ? Results.StatusCode(403) : Results.Ok(row);",
        ISO,
    ),
    Mutant(
        "transcript text is logged",
        "src/dotnet/Switchboard.Storage/ConfigStore.cs",
        """public Task<IReadOnlyList<TranscriptRow>> ListTranscriptsAsync(string tenant, string conversationId, CancellationToken ct = default) =>
        db.InTenantAsync<IReadOnlyList<TranscriptRow>>(tenant, async (c, t) =>
            [.. await c.QueryAsync<TranscriptRow>("SELECT conversation_id, turn, text FROM transcripts WHERE conversation_id = @conversationId ORDER BY turn", new { conversationId }, t)], ct);""",
        """public Task<IReadOnlyList<TranscriptRow>> ListTranscriptsAsync(string tenant, string conversationId, CancellationToken ct = default) =>
        db.InTenantAsync<IReadOnlyList<TranscriptRow>>(tenant, async (c, t) =>
        {
            var rows = (await c.QueryAsync<TranscriptRow>("SELECT conversation_id, turn, text FROM transcripts WHERE conversation_id = @conversationId ORDER BY turn", new { conversationId }, t)).ToList();
            Log?.Invoke(string.Join(' ', rows.Select(r => r.Text)));
            return (IReadOnlyList<TranscriptRow>)rows;
        }, ct);

    public static Action<string>? Log;""",
        "SKIP",
    ),
    Mutant(
        "decision engine keeps another tenant's intents",
        ENGINE,
        "foreach (var intent in t.AllIntents)\n        {\n            probs[intent] = input.Scores.TryGetValue(intent, out var s) ? Math.Max(0, s) : 0;\n        }",
        "foreach (var intent in t.AllIntents.Concat(input.Scores.Keys).Distinct())\n        {\n            probs[intent] = input.Scores.TryGetValue(intent, out var s) ? Math.Max(0, s) : 0;\n        }",
        UNIT,
    ),
    Mutant(
        "voicemail guard removed",
        ENGINE,
        "if (probs[Intents.Emergency] >= m.VoicemailGuard && ReachesVoicemail(t, dept, input.At)) continue;",
        "",
        UNIT,
    ),
    Mutant(
        "lexicon route removed",
        ENGINE,
        "if (input.LexiconHit)",
        "if (input.LexiconHit && input.ClarifyTurns < 0)",
        UNIT,
    ),
    Mutant(
        "tenant may lower an emergency penalty",
        MATRIX,
        """if (Protected.Contains(key) && value < current) throw new ArgumentException($"override may not lower protected cost {key}");""",
        "",
        UNIT,
    ),
    Mutant(
        "config validation lets an override lower a penalty",
        CONFIG,
        """else if (matrix.Protected.Contains(key) && value < current) errors.Add($"override may not lower protected cost {key}");""",
        "",
        UNIT,
    ),
    Mutant(
        "emergency department may have opening hours",
        CONFIG,
        """if (d.Hours != "always" || !d.EmergencyCapable) errors.Add("the emergency department must be emergency-capable and always open");""",
        "",
        UNIT,
    ),
    Mutant(
        "low ASR confidence is ignored",
        "src/dotnet/Switchboard.Domain/DecisionFlow.cs",
        "Degraded = ctx.Degraded || (ctx.AsrConfidence > 0 && ctx.AsrConfidence < LowAsrConfidence),",
        "Degraded = ctx.Degraded,",
        UNIT,
    ),
    Mutant(
        "redis key loses its tenant",
        "src/dotnet/Switchboard.Storage/TenantKeys.cs",
        'return $"{Prefix}{tenantId}:events";',
        'return $"{Prefix}shared:events";',
        UNIT,
    ),
    Mutant(
        "webhook accepts any signature",
        "src/dotnet/Switchboard.Gateway/Infrastructure.cs",
        'if (header is null || !header.StartsWith("sha256=", StringComparison.Ordinal)) return false;',
        "return true;\n#pragma warning disable CS0162",
        UNIT,
    ),
    Mutant(
        "llm budget never runs out",
        "src/dotnet/Switchboard.Gateway/Tracing.cs",
        "if (window.Used >= _perMinute) return false;",
        "",
        UNIT,
    ),
    Mutant(
        "tenant call limit never applies",
        "src/dotnet/Switchboard.Gateway/Infrastructure.cs",
        ".Wait(0);",
        ".Wait(0) || true;",
        UNIT,
    ),
    Mutant(
        "retention turns off row security",
        "src/dotnet/Switchboard.Storage/ConfigStore.cs",
        'c.ExecuteAsync("DELETE FROM transcripts WHERE created_at < @cutoff", new { cutoff }, t)',
        'c.ExecuteAsync("SET LOCAL row_security = off; DELETE FROM transcripts WHERE created_at < @cutoff", new { cutoff }, t)',
        ISO,
    ),
    Mutant(
        "direct database connection outside TenantDb",
        API,
        "app.Run();",
        """_ = new NpgsqlConnection(appConnection);\napp.Run();""",
        ARCH,
    ),
]
# The logging mutant needs a code path that doesn't exist in production, so it has its own test
# (Logs_contain_no_transcript_text) and isn't run as a source mutation.
MUTANTS = [m for m in MUTANTS if m.tests != "SKIP"]


def copy_tree(dst: Path) -> None:
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for rel in listed:
        src = ROOT / rel
        if src.is_file():
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def run_tests(work: Path, project: str) -> tuple[bool, str]:
    proc = subprocess.run(
        ["dotnet", "test", str(work / project), "--nologo", "-v", "q"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=work,
    )  # fmt: skip
    output = proc.stdout + proc.stderr
    failed = [ln.strip() for ln in output.splitlines() if "[FAIL]" in ln]
    if "error CS" in output or "error MSB" in output:
        raise RuntimeError(f"mutant did not compile, which proves nothing:\n{output[-1500:]}")
    if proc.returncode != 0 and not failed:
        # A red build with no failing test proves nothing, so it must not count as caught.
        raise RuntimeError(f"tests did not run:\n{output[-1500:]}")
    names = [ln.split("[FAIL]")[0].rstrip().split(".")[-1] for ln in failed[:3]]
    return bool(failed), "; ".join(names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="run mutants whose name contains this text")
    parser.add_argument("--out", default="runs/mutation_check.json")
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="sb-mutation-") as tmp:
        work = Path(tmp)
        copy_tree(work)
        for m in MUTANTS:
            if args.only and args.only not in m.name:
                continue
            path = work / m.file
            original = path.read_text(encoding="utf-8")
            if m.old not in original:
                print(f"ERROR  {m.name}: control text not found in {m.file}; the mutant is stale")
                results.append({"name": m.name, "status": "stale"})
                continue
            path.write_text(original.replace(m.old, m.new), encoding="utf-8", newline="\n")
            started = time.time()
            try:
                caught, by = run_tests(work, m.tests)
                status = "caught" if caught else "SURVIVED"
            except RuntimeError as err:
                status, by = "no-compile", str(err)[:400]
            finally:
                path.write_text(original, encoding="utf-8", newline="\n")
            print(f"{status:<10} {m.name}  [{time.time() - started:.0f}s]  {by}", flush=True)
            results.append({"name": m.name, "status": status, "caught_by": by})

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    bad = [r for r in results if r["status"] != "caught"]
    print(f"\n{len(results) - len(bad)}/{len(results)} mutants caught")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
