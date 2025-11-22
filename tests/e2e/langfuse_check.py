"""A real call's spans must reach self-hosted Langfuse, tagged by tenant, with no transcript text."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx2 as httpx
from run_e2e import COMPOSE, compose, place_call

ROOT = Path(__file__).resolve().parents[2]
LAYERED = [*COMPOSE, "-f", str(ROOT / "deploy/compose.observability.yaml"), "--profile", "langfuse"]
PUBLIC, SECRET = "pk-lf-local", "sk-lf-local"
BASE = "http://127.0.0.1:3000"


def main() -> int:
    auth = base64.b64encode(f"{PUBLIC}:{SECRET}".encode()).decode()
    env = {**os.environ, "LANGFUSE_OTLP_AUTH": auth, "OTEL_EXPORTER_OTLP_ENDPOINT": "http://otel-collector-langfuse:4318"}
    up = subprocess.run([*LAYERED, "up", "-d", "langfuse-web", "langfuse-worker", "otel-collector-langfuse", "gateway"], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    print(up.stderr[-300:])
    report: dict = {"ok": False}
    try:
        for _ in range(60):
            try:
                if httpx.get(f"{BASE}/api/public/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        else:
            report["failure"] = "Langfuse never became healthy"
            return 1
        time.sleep(15)
        for did in ("27005550103", "27005550101"):
            place_call(did)
        traces: list[dict] = []
        for _ in range(24):
            time.sleep(5)
            response = httpx.get(f"{BASE}/api/public/traces", params={"limit": 50}, auth=(PUBLIC, SECRET), timeout=20)
            if response.status_code == 200:
                traces = response.json().get("data", [])
                if len(traces) >= 2:
                    break
        blob = json.dumps(traces)
        report = {
            "traces": len(traces),
            "names": sorted({t.get("name") or "" for t in traces}),
            "tags": sorted({tag for t in traces for tag in (t.get("tags") or [])}),
            "sessions": len({t.get("sessionId") for t in traces}),
            "transcript_leak": any(w in blob for w in ("not breathing", "my router keeps")),
        }
        report["ok"] = len(traces) >= 2 and not report["transcript_leak"]
        return 0 if report["ok"] else 1
    finally:
        print(json.dumps(report, indent=1))
        (ROOT / "runs/langfuse_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        subprocess.run([*LAYERED, "stop", "langfuse-web", "langfuse-worker", "otel-collector-langfuse", "clickhouse", "minio", "langfuse-db", "langfuse-redis"], capture_output=True, check=False)
        compose("up", "-d", "--wait", "--wait-timeout", "120", "gateway", check=False)


if __name__ == "__main__":
    sys.exit(main())
