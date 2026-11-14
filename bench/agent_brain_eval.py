"""Small behaviour check of agent brains on the real prompt: does each case get the right kind of answer?

Spends a few cents. Usage: python bench/agent_brain_eval.py gpt-4.1-mini gpt-5.4-mini ...
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

import yaml
from switchboard_inference.agent.brain import Department, Persona, Reply, build_system_prompt, caller_message
from switchboard_inference.agent.openai_brain import OpenAIBrain

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = 2

# (tenant, what the caller said, the action expected: "transfer:<dept>", "ask", or "answer:<words that must appear>")
CASES = [
    ("thornveld_vet", "through Turkey restarting by itself, and I need help.", "ask"),
    ("thornveld_vet", "My dog just ate rat poison, what do I do?", "transfer:on_call_vet"),
    ("thornveld_vet", "How much is a consultation?", "answer:450"),
    ("thornveld_vet", "Can I speak to someone about my invoice?", "transfer:accounts"),
    ("thornveld_vet", "Do you have an appointment at three tomorrow?", "no-invent"),
    ("quillfibre", "My internet has been down since this morning and the red light is on.", "any"),
    ("blue_aloe", "Can you tell me if my claim will be paid?", "no-invent"),
    ("sandrivier_metro", "There's a live wire hanging over the road outside my house!", "transfer:emergency_control"),
]


def persona(tenant: str) -> Persona:
    raw = yaml.safe_load((ROOT / "configs/tenants" / f"{tenant}.yaml").read_text(encoding="utf-8"))
    agent = raw["agent"]
    notes = agent.get("department_notes", {})
    departments = [Department(d["id"], notes.get(d["id"], d["id"]), True, "voicemail", bool(d.get("emergency_capable"))) for d in raw["departments"]]
    return Persona(raw["name"].replace(" (fictional)", ""), agent["name"], agent["greeting"], agent["knowledge"], departments, "Thursday 13 November 2026, 11:00")


def judge(expected: str, text: str, tools: list) -> bool:
    transfer = next((t.input.get("department") for t in tools if t.name == "transfer_to_department"), None)
    if expected.startswith("transfer:"):
        return transfer == expected.split(":", 1)[1]
    if expected == "ask":
        return transfer is None and "?" in text
    if expected.startswith("answer:"):
        words = {"450": "four hundred and fifty"}
        key = expected.split(":", 1)[1]
        return transfer is None and (key in text.replace(",", "").replace(" ", "") or words.get(key, "~") in text.lower())
    if expected == "no-invent":
        # Mustn't confirm a slot or a claim outcome: the answer has to decline, or offer to check or hand over.
        lowered = text.lower().replace("’", "'")
        return any(w in lowered for w in ("can't", "cannot", "don't", "not able", "unable", "check", "message", "put you through"))
    return True


async def main(models: list[str]) -> None:
    for model in models:
        brain = OpenAIBrain(model=model)
        passed, times = 0, []
        for tenant, said, expected in CASES * ROUNDS:
            reply = Reply()
            started = time.perf_counter()
            first = None
            text = ""
            async for piece in brain.respond(build_system_prompt(persona(tenant)), [caller_message(said)], reply):
                first = first or time.perf_counter()
                text += piece
            times.append(((first or time.perf_counter()) - started) * 1000)
            ok = judge(expected, text, reply.tool_uses)
            passed += ok
            if not ok:
                print(f"  {model} FAIL {tenant}: {said!r} -> {text[:110]!r} {[(t.name, t.input.get('department')) for t in reply.tool_uses]}")
        print(f"{model}: {passed}/{len(CASES) * ROUNDS} right, first word median {statistics.median(times):.0f} ms")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
