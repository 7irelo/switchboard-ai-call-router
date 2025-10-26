"""Toxiproxy on the gateway-to-inference link: added latency, then a black hole, during real calls."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx2 as httpx
from run_e2e import COMPOSE, compose, place_call, sink_log

ROOT = Path(__file__).resolve().parents[2]
LAYERED = [*COMPOSE, "-f", str(ROOT / "deploy/compose.observability.yaml"), "--profile", "chaos"]
TOXI = "http://127.0.0.1:8474"
DID = "27005550103"


def layered(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([*LAYERED, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, **env}, check=False)


def run_case(name: str, toxic: dict | None, expected: set[str]) -> dict:
    httpx.post(f"{TOXI}/reset", timeout=5)
    if toxic:
        response = httpx.post(f"{TOXI}/proxies/inference/toxics", json=toxic, timeout=5)
        response.raise_for_status()
    before = sink_log()
    started = time.time()
    _, code, _ = place_call(DID)
    seen = set(re.findall(r"INVITE sip:([a-z0-9-]+)@", sink_log()[len(before) :]))
    logs = compose("logs", "--no-color", "--since", str(int(started) - 1), "gateway", check=False).stdout
    timing = re.findall(r"decision_ms=([0-9.]+)", logs)
    return {
        "case": name,
        "caller_exit": code,
        "reached": sorted(seen),
        "ok": code == 0 and bool(seen) and seen <= expected,
        "decision_ms": float(timing[-1]) if timing else None,
        "degraded": "inference unavailable" in logs,
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    env = {"INFERENCE_URL": "http://toxiproxy:50052"}
    layered(env, "up", "-d", "--wait", "--wait-timeout", "60", "toxiproxy")
    httpx.post(f"{TOXI}/proxies", json={"name": "inference", "listen": "0.0.0.0:50052", "upstream": "inference:50051", "enabled": True}, timeout=10)
    layered(env, "up", "-d", "--wait", "--wait-timeout", "120", "gateway")
    time.sleep(6)
    try:
        results = [
            run_case("no_toxic", None, {"quillfibre-frontline"}),
            run_case("latency_300ms", {"name": "lat", "type": "latency", "stream": "downstream", "attributes": {"latency": 300, "jitter": 50}}, {"quillfibre-frontline"}),
            # Connection held but silent. The caller hangs up at about 8.5s, so this checks nothing crashes.
            run_case("slow_bandwidth", {"name": "bw", "type": "bandwidth", "stream": "upstream", "attributes": {"rate": 8}}, {"quillfibre-frontline"}),
        ]
    finally:
        compose("up", "-d", "--wait", "--wait-timeout", "120", "gateway", check=False)
        layered({}, "stop", "toxiproxy")
    for result in results:
        print(json.dumps(result))
    (ROOT / "runs/chaos.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
