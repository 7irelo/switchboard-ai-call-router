"""Real SIP calls through real Asterisk, plus the webhook path, with canary scans afterwards.

Needs `poe stack-up` and seeded tenants. Exits non-zero on any wrong route or cross-tenant leak.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx2 as httpx

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "deploy/compose.yaml"), "--env-file", str(ROOT / ".env")]
GATEWAY = "http://127.0.0.1:8090"
API = "http://127.0.0.1:8080"
CALLS = {
    "27005550101": ("thornveld_vet", "thornveld-oncall"),
    "27005550102": ("sandrivier_metro", "sandrivier-faults"),
    "27005550103": ("quillfibre", "quillfibre-frontline"),
    "27005550104": ("marula_ridge", "marula-duty"),
    "27005550105": ("blue_aloe", "bluealoe-assist"),
}
CANARIES = {
    "thornveld_vet": ("wa-thornveld-01", "CANARY-THORN-41ab"),
    "quillfibre": ("wa-quillfibre-01", "CANARY-QUILL-77cd"),
}
ALL_TARGETS = {t for _, t in CALLS.values()}
RUN = str(int(time.time()))  # Message ids must be new each run: dedup correctly survives restarts.


def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([*COMPOSE, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", check=check)


def place_call(did: str) -> tuple[str, int, str]:
    result = compose(
        "--profile", "test", "run", "--rm", "caller", "sipp", "asterisk:5060", "-sf", "/sipp/uac.xml", "-s", did,
        "-m", "1", "-nostdin", "-key", "salt", f"{did}{RUN}", "-timeout", "40", "-timeout_error",
        check=False,
    )  # fmt: skip
    return did, result.returncode, result.stdout[-400:] + result.stderr[-400:]


def sink_log() -> str:
    return compose("exec", "-T", "sink", "sh", "-c", "cat /logs/sink.log 2>/dev/null || true", check=False).stdout


def send_webhook(secret: str, phone: str, message_id: str, text: str, kind: str = "text") -> int:
    message = {"from": "27000000000", "id": message_id, "type": kind}
    if kind == "text":
        message["text"] = {"body": text}
    body = json.dumps({"entry": [{"changes": [{"value": {"metadata": {"phone_number_id": phone}, "messages": [message]}}]}]}).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return httpx.post(f"{GATEWAY}/webhooks/whatsapp", content=body, headers={"X-Hub-Signature-256": signature, "Content-Type": "application/json"}, timeout=10).status_code


def main() -> int:
    env = dict(line.split("=", 1) for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith("#"))
    keys = json.loads((ROOT / "runs/tenant_keys.json").read_text(encoding="utf-8"))
    failures: list[str] = []

    print("voice: 5 concurrent calls, one per tenant")
    before = sink_log()
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        results = list(pool.map(place_call, CALLS))
    for did, code, tail in results:
        if code != 0:
            failures.append(f"call to {did} exited {code}: {tail}")
    log = sink_log()[len(before):]
    seen = set(re.findall(r"INVITE sip:([a-z0-9-]+)@", log))
    print(f"  departments reached: {sorted(seen)}")
    for did, (tenant, target) in CALLS.items():
        if target not in seen:
            failures.append(f"{tenant} ({did}) never reached {target}")
    if seen - ALL_TARGETS:
        failures.append(f"unexpected targets reached: {seen - ALL_TARGETS}")

    # The destination alone isn't proof: a tenant's fallback can be the same phone. Each call must also
    # have been routed by the engine, not sent to fallback by a failure.
    gateway_logs = compose("logs", "--no-color", "gateway", check=False).stdout
    for did, (tenant, _) in CALLS.items():
        if f"routed tenant={tenant} " not in gateway_logs:
            failures.append(f"{tenant} ({did}) has no routed decision: it reached a phone by fallback")
    if "call failed, sending to fallback" in gateway_logs:
        failures.append("at least one call failed and went to fallback")

    print("messages: signature, duplicates, unsupported types, tenancy")
    secret = env["WHATSAPP_APP_SECRET"]
    if send_webhook("wrong-secret", "wa-quillfibre-01", f"m-forged-{RUN}", "hello") != 401:
        failures.append("a forged signature was accepted")
    for tenant, (phone, canary) in CANARIES.items():
        for _ in range(2):  # A duplicate delivery, which must produce one reply.
            if send_webhook(secret, phone, f"m-{tenant}-{RUN}", f"my bill is wrong {canary}") != 200:
                failures.append(f"webhook rejected for {tenant}")
    send_webhook(secret, "wa-quillfibre-01", f"m-voice-{RUN}", "", kind="audio")
    send_webhook(secret, "unknown-phone", f"m-unknown-{RUN}", "CANARY-NOBODY")
    time.sleep(4)
    replies = httpx.get(f"{GATEWAY}/mock/outbox/quillfibre", timeout=10).json()
    print(f"  quillfibre replies: {[r['text'][:50] for r in replies]}")
    routed = [r for r in replies if r.get("department") == "billing_team"]
    if len(routed) != 1:
        failures.append(f"expected one billing reply after a duplicate delivery, got {len(routed)}")
    if not any("text messages" in r["text"] for r in replies):
        failures.append("no defined reply for an unsupported message type")

    print("persistence and canary scan")
    time.sleep(3)
    for tenant, (_, canary) in CANARIES.items():
        other = next(c for t, (_, c) in CANARIES.items() if t != tenant)
        client = httpx.Client(base_url=API, headers={"Authorization": f"Bearer {keys[tenant]}"}, timeout=10)
        conversations = client.get("/v1/conversations").json()
        blob = json.dumps(conversations)
        for conversation in conversations:
            blob += client.get(f"/v1/conversations/{conversation['id']}/transcript").text
            blob += client.get(f"/v1/conversations/{conversation['id']}/events").text
        if canary not in blob:
            failures.append(f"{tenant}: its own canary wasn't persisted")
        if other in blob:
            failures.append(f"LEAK: {tenant} can read another tenant's canary")
        print(f"  {tenant}: {len(conversations)} conversations, own canary found: {canary in blob}")

    # No transcript text at any log level, from any service.
    logs = compose("logs", "--no-color", "api", "gateway", "worker", "inference", check=False).stdout
    for _, canary in CANARIES.values():
        if canary in logs:
            failures.append("LEAK: a canary appeared in service logs")
    for spoken in ("not breathing", "my router keeps"):
        if spoken in logs:
            failures.append("LEAK: transcript text appeared in service logs")

    # Redis: a canary may only appear in its own tenant's stream.
    for tenant, (_, canary) in CANARIES.items():
        for other, (_, _other_canary) in CANARIES.items():
            if other == tenant:
                continue
            raw = compose("exec", "-T", "redis", "redis-cli", "--raw", "XRANGE", f"sb:t:{other}:events", "-", "+", check=False).stdout
            decoded = decode_stream(raw)
            if canary in decoded:
                failures.append(f"LEAK: {tenant}'s canary is in {other}'s stream")

    timings = [float(x) for x in re.findall(r"decision_ms=([0-9.]+)", logs)]
    print(f"  decision timings captured: {len(timings)}")
    if failures:
        print("\nE2E FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nE2E PASSED")
    return 0


def decode_stream(raw: str) -> str:
    import base64

    out = []
    for token in raw.split():
        try:
            out.append(base64.b64decode(token, validate=True).decode("utf-8", "ignore"))
        except ValueError:
            continue
    return " ".join(out)


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
