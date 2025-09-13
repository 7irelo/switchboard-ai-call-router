"""Open-loop SIP load through real Asterisk with replay ASR, reporting stage timings from the gateway.

What this measures: end-of-speech signal to decision, and the ARI route command. It does NOT measure
T_route as pre-registered, because replay ASR has no endpointing wait and no network to Deepgram.
That number needs live credentials.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/e2e"))
from run_e2e import compose  # noqa: E402

LINE = re.compile(r"routed tenant=(\S+) .*?decision_ms=([0-9.]+) route_cmd_ms=([0-9.]+)")


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))] if ordered else float("nan")


def summary(values: list[float]) -> dict:
    return {"n": len(values), "p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99), "max": max(values, default=float("nan"))}


def level(rate: float, calls: int) -> dict:
    since = str(int(time.time()) - 1)
    # SIPp's -r is open-loop: arrivals don't wait for earlier calls, which avoids coordinated omission.
    script = (
        f"sipp asterisk:5060 -sf /sipp/uac.xml -s 27005550103 -r {rate} -rp 1000 -m {calls} -l {calls} -nostdin "
        f"-key salt load{int(time.time())} -timeout {int(calls / rate) + 40} >/dev/null 2>&1; echo exit=$?"
    )
    started = time.time()
    result = compose("--profile", "test", "run", "--rm", "caller", "sh", "-c", script, check=False)
    logs = compose("logs", "--no-color", "--since", since, "gateway", check=False).stdout
    matches = LINE.findall(logs)
    return {
        "rate_cps": rate,
        "calls_offered": calls,
        "calls_routed": len(matches),
        "fallbacks": logs.count("sending to fallback"),
        "sipp": (result.stdout.strip().splitlines() or [""])[-1],
        "wall_seconds": round(time.time() - started, 1),
        "eos_to_decision_ms": summary([float(m[1]) for m in matches]),
        "route_cmd_ms": summary([float(m[2]) for m in matches]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="0.5:10,1:20,2:30,4:40")
    parser.add_argument("--out", default="runs/load_e2e.json")
    args = parser.parse_args()
    results = []
    for item in args.levels.split(","):
        rate, calls = item.split(":")
        result = level(float(rate), int(calls))
        print(json.dumps(result), flush=True)
        results.append(result)
    report = {
        "asr": "replay",
        "not_t_route": True,
        "host": {"platform": platform.platform(), "processor": platform.processor()},
        "pinned_cores": False,
        "median_route_ms": statistics.median([r["route_cmd_ms"]["p50"] for r in results if r["calls_routed"]] or [0]),
        "levels": results,
    }
    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
