"""M0 Spike C: real LLM p50/p95 from this machine. Needs ANTHROPIC_API_KEY and spends real money.

Also settles the open question in configs/llm_models.yaml: whether each model accepts its thinking setting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from switchboard_inference.decision import CANONICAL_INTENTS
from switchboard_inference.llm.client import LlmClient

PROBES = ["my bill is wrong this month", "there is no water in our street", "i want to speak about my account", "ngicela usizo my router is dead"]


async def run(model: str, calls: int) -> dict:
    client = LlmClient(model=model)
    times: list[float] = []
    outcomes: dict[str, int] = {}
    cost = 0.0
    for n in range(calls):
        # A long deadline here: this measures the true distribution, not what the budget allows.
        result = await client.classify(PROBES[n % len(PROBES)], list(CANONICAL_INTENTS), deadline_s=30)
        outcomes[f"{result.outcome}:{result.detail}"] = outcomes.get(f"{result.outcome}:{result.detail}", 0) + 1
        cost += result.cost_usd
        if result.outcome in ("answered", "needs_clarification"):
            times.append(result.llm_ms)
    await client.client.close()  # Before asyncio.run closes the loop, or the pool's cleanup raises on exit.
    times.sort()
    return {
        "model": model,
        "calls": calls,
        "outcomes": outcomes,
        "p50_ms": times[len(times) // 2] if times else None,
        "p95_ms": times[int(0.95 * len(times))] if times else None,
        "fits_550ms_reserve": bool(times) and times[int(0.95 * len(times))] <= 550,
        "cost_usd": round(cost, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--models", default=None)
    args = parser.parse_args()
    models = args.models.split(",") if args.models else list(yaml.safe_load(Path("configs/llm_models.yaml").read_text(encoding="utf-8"))["models"])
    results = [asyncio.run(run(m, args.calls)) for m in models]
    Path("runs").mkdir(exist_ok=True)
    Path("runs/spike_llm_latency.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for result in results:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
