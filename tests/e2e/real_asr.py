"""Real speech through real Asterisk into real ASR, with no paid services, and three ablation modes.

- whisper: the caller's speech is transcribed by Whisper, and routing follows what was actually heard.
- failover: the Deepgram adapter is pointed at a host that doesn't resolve, so a real call must fail over to
  Whisper, declare itself degraded and still reach the right department.
- early commit (L3): replay ASR, measuring how much listening time it saves.
The latency here is an approximation: see `approx_t_route_ms`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from run_e2e import COMPOSE, compose, sink_log

ROOT = Path(__file__).resolve().parents[2]
DID = "27005550103"  # quillfibre. "my router..." is tech_support, which goes to frontline, open at all hours.
EXPECTED = "quillfibre-frontline"
CALLER_LEAD_MS = 500  # The scenario pauses this long after ACK before speaking.


def up(env: dict, *services: str) -> None:
    result = subprocess.run(
        [*COMPOSE, "up", "-d", "--wait", "--wait-timeout", "400", *services], env={**os.environ, **env},
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )  # fmt: skip
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-600:])
    time.sleep(8)


def speak(salt: str, pcap: str, hold_ms: int) -> int:
    script = (
        f"sed -e 's#/sipp/speech.pcap#{pcap}#' -e 's/milliseconds=\"8000\"/milliseconds=\"{hold_ms}\"/' /sipp/uac.xml > /tmp/uac.xml && "
        f"sipp asterisk:5060 -sf /tmp/uac.xml -s {DID} -m 1 -nostdin -key salt {salt} -timeout {hold_ms // 1000 + 20} -timeout_error >/dev/null 2>&1"
    )
    return compose("--profile", "test", "run", "--rm", "caller", "sh", "-c", script, check=False).returncode


def speech_ms() -> float:
    return float(compose("--profile", "test", "run", "--rm", "caller", "cat", "/sipp/spoken.ms", check=False).stdout.strip().splitlines()[-1])


def run_case(name: str, env: dict, pcap: str, hold_ms: int, spoken_ms: float | None) -> dict:
    up(env, "inference", "gateway")
    before = sink_log()
    started = time.time()
    code = speak(f"{name}{int(started)}", pcap, hold_ms)
    seen = set(re.findall(r"INVITE sip:([a-z0-9-]+)@", sink_log()[len(before) :]))
    logs = compose("logs", "--no-color", "--since", str(int(started) - 1), "gateway", check=False).stdout
    line = re.search(r"routed tenant=\S+ .*?department=(\S+) reason=(\S+) .*?listen_ms=([0-9]+) decision_ms=([0-9.]+) route_cmd_ms=([0-9.]+)", logs)
    result: dict = {"case": name, "caller_exit": code, "reached": sorted(seen), "ok": code == 0 and seen == {EXPECTED} and line is not None}
    if line:
        listen, decision, route = float(line[3]), float(line[4]), float(line[5])
        result |= {"department": line[1], "reason": line[2], "listen_ms": listen, "decision_ms": decision, "route_cmd_ms": route}
        if spoken_ms is not None:
            # The turn opens about when the caller answers, so this has an error of a few hundred ms.
            result["approx_t_route_ms"] = round(listen + decision - spoken_ms - CALLER_LEAD_MS)
    return result


def degraded_recorded() -> bool:
    query = "select count(*) from conversation_events where tenant_id='quillfibre' and type='Degraded' and occurred_at > now() - interval '5 minutes'"
    out = compose("exec", "-T", "postgres", "psql", "-U", "postgres", "-d", "switchboard", "-tAc", query, check=False).stdout.strip()
    return out.isdigit() and int(out) > 0


def main() -> int:
    spoken = speech_ms()
    results = []
    # Char n-grams: the keyword model can't survive Whisper hearing "router" as "rooter".
    base = {"WHISPER_SIZE": "tiny", "ASR_MODE": "whisper", "INFERENCE_MODEL": "/models/tfidf.pkl"}
    results.append(run_case("whisper_real_speech", base, "/sipp/real_speech.pcap", 30000, spoken))
    failover = {"ASR_MODE": "live", "INFERENCE_MODEL": "/models/tfidf.pkl", "WHISPER_SIZE": "tiny", "DEEPGRAM_API_KEY": "not-a-real-key", "DEEPGRAM_HOST": "deepgram.invalid"}
    case = run_case("deepgram_unreachable_fails_over_to_whisper", failover, "/sipp/real_speech.pcap", 30000, spoken)
    time.sleep(3)
    case["degraded_event_recorded"] = degraded_recorded()
    case["ok"] = case["ok"] and case["degraded_event_recorded"]
    results.append(case)
    # L3 on replay ASR: same words, three commit policies.
    for mode in ("off", "stable", "emergency"):
        results.append(run_case(f"early_commit_{mode}", {"ASR_MODE": "replay", "EARLY_COMMIT": mode, "WHISPER_SIZE": "", "INFERENCE_MODEL": ""}, "/sipp/speech.pcap", 8000, None))
    up({"ASR_MODE": "replay", "EARLY_COMMIT": "off", "WHISPER_SIZE": "", "INFERENCE_MODEL": ""}, "inference", "gateway")
    for result in results:
        print(json.dumps(result))
    (ROOT / "runs/real_asr.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
