"""Stops one dependency at a time, places a real call, and checks the caller still ends up somewhere safe."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from run_e2e import compose, sink_log

ROOT = Path(__file__).resolve().parents[2]
DID = "27005550103"  # quillfibre: the scripted line routes to frontline, which is always open.


def long_call(salt: str, hold_ms: int) -> int:
    # The caller holds long enough for reprompts and keypad timeouts to play out.
    script = (
        f"sed 's/milliseconds=\"8000\"/milliseconds=\"{hold_ms}\"/' /sipp/uac.xml > /tmp/uac.xml && "
        f"sipp asterisk:5060 -sf /tmp/uac.xml -s {DID} -m 1 -nostdin -key salt {salt} -timeout {hold_ms // 1000 + 15} -timeout_error >/dev/null 2>&1"
    )
    return compose("--profile", "test", "run", "--rm", "caller", "sh", "-c", script, check=False).returncode


def scenario(name: str, service: str, hold_ms: int, expected: set[str]) -> dict:
    before = sink_log()
    compose("stop", service)
    started = time.time()
    try:
        code = long_call(f"{name}{int(started)}", hold_ms)
    finally:
        # Wait until it's healthy, so one scenario's recovery can't bleed into the next.
        compose("up", "-d", "--wait", "--wait-timeout", "120", service)
        time.sleep(5)  # At most the gateway's reconnect backoff.
    seen = set(re.findall(r"INVITE sip:([a-z0-9-]+)@", sink_log()[len(before):]))
    ok = code == 0 and bool(seen) and seen <= expected
    return {"scenario": name, "stopped": service, "caller_exit": code, "reached": sorted(seen), "ok": ok, "seconds": round(time.time() - started, 1)}


def main() -> int:
    results = [
        # Inference down: rules, then keypad, then the tenant's fallback.
        scenario("inference_down", "inference", 30000, {"quillfibre-frontline"}),
        # Postgres down: the config cache serves the last-known snapshot, so routing is unaffected.
        scenario("postgres_down", "postgres", 8000, {"quillfibre-frontline"}),
        # Redis down: events drop with a counter, and the call doesn't notice.
        scenario("redis_down", "redis", 8000, {"quillfibre-frontline"}),
    ]
    time.sleep(3)
    # A dependency coming back must not leave a service dead.
    states = compose("ps", "-a", "--format", "{{.Service}} {{.State}}", check=False).stdout
    dead = [line for line in states.splitlines() if line.split()[-1] != "running"]
    results.append({"scenario": "all_services_survive", "not_running": dead, "ok": not dead})
    for result in results:
        print(json.dumps(result))
    out = ROOT / "runs/failure_injection.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
