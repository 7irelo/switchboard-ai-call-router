"""Real calls, real spans, a real Collector: each tenant's sink must hold only that tenant, and no transcript."""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

from run_e2e import CALLS, COMPOSE, compose, place_call

ROOT = Path(__file__).resolve().parents[2]
LAYERED = [*COMPOSE, "-f", str(ROOT / "deploy/compose.observability.yaml"), "--profile", "traces"]
SPOKEN = ["not breathing", "my router", "no water", "on fire", "gas leak"]


def main() -> int:
    import subprocess

    env = {**os.environ, "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector:4318"}

    def layered(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([*LAYERED, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, check=False)

    print(layered("up", "-d", "--wait", "--wait-timeout", "120", "otel-collector", "gateway", "inference").stderr[-200:])
    time.sleep(6)
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        results = list(pool.map(place_call, CALLS))
    time.sleep(8)
    failures = [f"call {did} exited {code}" for did, code, _ in results if code != 0]
    report: dict = {"tenants": {}}
    # The Collector image has no shell, so the volume is read from a throwaway container.
    dump = subprocess.run(
        ["docker", "run", "--rm", "-v", "switchboard_trace-files:/traces:ro", "debian:bookworm-slim", "sh", "-c", "cd /traces && for f in *.jsonl; do echo \"==$f\"; cat \"$f\"; echo; done"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    ).stdout  # fmt: skip
    sinks: dict[str, str] = {}
    for part in dump.split("==")[1:]:
        name, _, body = part.partition("\n")
        sinks[name.replace(".jsonl", "").strip()] = body
    for tenant, _ in CALLS.values():
        body = sinks.get(tenant, "")
        spans = body.count('"spanId"')
        report["tenants"][tenant] = {"spans": spans}
        if spans == 0:
            failures.append(f"{tenant}: no spans reached its sink")
        for other, _ in CALLS.values():
            if other != tenant and f'"stringValue":"{other}"' in body:
                failures.append(f"LEAK: {other} appears in {tenant}'s sink")
        for words in SPOKEN:
            if words in body:
                failures.append(f"LEAK: transcript text in {tenant}'s spans")
    report["quarantine_spans"] = sinks.get("quarantine", "").count('"spanId"')
    report["failures"] = failures
    report["ok"] = not failures
    (ROOT / "runs/trace_isolation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    # Back to the default stack, with tracing off.
    compose("up", "-d", "--wait", "--wait-timeout", "120", "gateway", "inference", check=False)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
