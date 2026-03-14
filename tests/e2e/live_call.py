"""Real SIP calls through real Asterisk into live Deepgram: the closest this repo gets to T_route without people.

The speech is synthetic English, so accuracy here means nothing. The timing is an estimate with an error of a
few hundred ms, because the gateway doesn't know exactly when the caller starts speaking. Calls that fell over
to Whisper are counted apart, so a Deepgram failure can't pass as a Deepgram latency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from real_asr import CALLER_LEAD_MS, EXPECTED, ROOT, speak, speech_ms, up
from run_e2e import compose, sink_log

BUDGET_MS = 1500


def degraded_since(started: float) -> int:
    query = f"select count(*) from conversation_events where tenant_id='quillfibre' and type='Degraded' and occurred_at > to_timestamp({started})"
    out = compose("exec", "-T", "postgres", "psql", "-U", "postgres", "-d", "switchboard", "-tAc", query, check=False).stdout.strip()
    return int(out) if out.isdigit() else -1


def heard_since(started: float) -> list[str]:
    query = f"select payload->'final'->>'text' from conversation_events where tenant_id='quillfibre' and type='Final' and occurred_at > to_timestamp({started}) order by occurred_at"
    out = compose("exec", "-T", "postgres", "psql", "-U", "postgres", "-d", "switchboard", "-tAc", query, check=False).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def one_call(name: str, spoken: float, hold_ms: int) -> dict:
    before = sink_log()
    started = time.time()
    code = speak(f"{name}{int(started)}", "/sipp/real_speech.pcap", hold_ms)
    reached = sorted(set(re.findall(r"INVITE sip:([a-z0-9-]+)@", sink_log()[len(before) :])))
    logs = compose("logs", "--no-color", "--since", str(int(started) - 1), "gateway", check=False).stdout
    line = re.search(r"routed tenant=\S+ .*?department=(\S+) reason=(\S+) .*?listen_ms=([0-9]+) decision_ms=([0-9.]+) route_cmd_ms=([0-9.]+)", logs)
    time.sleep(2)  # The worker writes events shortly after the call.
    result: dict = {"caller_exit": code, "reached": reached, "correct": reached == [EXPECTED], "degraded_events": degraded_since(started), "heard": heard_since(started)}
    if line:
        listen, decision = float(line[3]), float(line[4])
        result |= {"reason": line[2], "listen_ms": listen, "decision_ms": decision, "route_cmd_ms": float(line[5]), "approx_t_route_ms": round(listen + decision - spoken - CALLER_LEAD_MS)}
    return result


def summarise(calls: list[dict]) -> dict:
    # The first call is reported apart: it pays for cold starts that later calls don't.
    usable = [c for c in calls[1:] if "approx_t_route_ms" in c and c["degraded_events"] == 0]
    times = sorted(c["approx_t_route_ms"] for c in usable)
    return {
        "calls": len(calls),
        "routed_correctly": sum(c["correct"] for c in calls),
        "fell_back_to_whisper": sum(c["degraded_events"] != 0 for c in calls),
        "cold_first_call_ms": calls[0].get("approx_t_route_ms") if calls else None,
        "warm_on_deepgram": len(times),
        "warm_p50_ms": times[len(times) // 2] if times else None,
        "warm_p95_ms": times[min(len(times) - 1, round(0.95 * (len(times) - 1)))] if times else None,
        "warm_max_ms": times[-1] if times else None,
        "within_budget": sum(t <= BUDGET_MS for t in times),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--hosts", default="api.deepgram.com,api.eu.deepgram.com")
    parser.add_argument("--hold-ms", type=int, default=30000)
    parser.add_argument("--out", default="runs/live_call.json")
    args = parser.parse_args()
    spoken = speech_ms()
    report: dict = {"reportable": False, "voice": "synthetic English", "not_preregistered_t_route": True, "budget_ms": BUDGET_MS, "spoken_ms": spoken, "arms": {}, "rows": {}}
    for host in args.hosts.split(","):
        up({"ASR_MODE": "live", "DEEPGRAM_HOST": host, "WHISPER_SIZE": "tiny", "INFERENCE_MODEL": "/models/tfidf.pkl"}, "inference", "gateway")
        calls = []
        for n in range(args.calls):
            calls.append(one_call(f"live{n}", spoken, args.hold_ms))
            print(host, n, json.dumps(calls[-1]), flush=True)
        report["rows"][host] = calls
        report["arms"][host] = summarise(calls)
        print(host, json.dumps(report["arms"][host]), flush=True)
        (ROOT / args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    up({"ASR_MODE": "replay", "WHISPER_SIZE": "", "INFERENCE_MODEL": ""}, "inference", "gateway")
    Path(ROOT / args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
