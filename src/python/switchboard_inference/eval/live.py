"""Tier T2/M4: LLM-only and cascade-with-LLM arms, with real answers cached and spend capped.

Needs ANTHROPIC_API_KEY for anything not yet cached. With no key it uses the cache and reports any misses,
and never invents an answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np

from switchboard_inference.data.splits import load_split
from switchboard_inference.decision import CostMatrix, load_lexicon, load_tenants
from switchboard_inference.eval.metrics import mcnemar_p, paired_bootstrap_diff
from switchboard_inference.eval.runner import fit_tfidf, summarise
from switchboard_inference.eval.simulate import Outcome, _finish, run_conversation, scoped
from switchboard_inference.llm.client import LlmClient, LlmResult
from switchboard_inference.llm.fusion import ConfusionFusion
from switchboard_inference.llm.recorded import UNCACHEABLE_OUTCOMES, ResponseCache, SpendLedger, cache_key


class Oracle:
    """Answers from the cache first, then live if there's a key and room in the budget."""

    def __init__(self, model: str, cache: ResponseCache, ledger: SpendLedger, client: LlmClient | None):
        self.model, self.cache, self.ledger, self.client = model, cache, ledger, client
        self.misses = 0
        self.failed_calls = 0
        # One loop for every call: the client's pooled connections belong to the loop that opened them.
        self.loop = asyncio.new_event_loop()

    def ask(self, tenant, text: str) -> LlmResult | None:
        allowed = list(tenant.intents)
        override = tenant.raw.get("prompt_override", "")
        key = cache_key(tenant.id, self.model, text, allowed, override)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if self.client is None or not self.ledger.allow():
            self.misses += 1
            return None
        customs = list(tenant.custom_intents.values())
        result = self.loop.run_until_complete(self.client.classify(text, allowed, override, customs, deadline_s=30))
        self.ledger.record(result)
        if result.outcome in UNCACHEABLE_OUTCOMES:
            self.failed_calls += 1
        self.cache.put(key, result)
        return result


def llm_only(row, tenant, matrix, oracle: Oracle) -> tuple[Outcome, LlmResult | None]:
    from datetime import datetime

    result = oracle.ask(tenant, row["text"])
    at = datetime.fromisoformat(row["at"])
    if result is None or result.intent is None:
        return _finish(row, tenant, matrix, at, row["intent"], "reception", tenant.raw["fallback"]["department"], 0.0, 0, True, ["llm_no_answer"]), result
    return _finish(row, tenant, matrix, at, row["intent"], "route", tenant.department_for(result.intent), 0.0, 0, True, ["llm"]), result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fixture")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--cap-usd", type=float, default=2.0)
    parser.add_argument("--cache", default="runs/llm_cache.jsonl")
    parser.add_argument("--out", default="runs/live_eval.json")
    args = parser.parse_args()

    root = Path(args.data)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tenants, matrix, phrases = load_tenants(), CostMatrix.load("configs/cost_matrix.yaml"), load_lexicon()
    rows = load_split(root, args.split)[: args.limit]
    model, calibration = fit_tfidf(root)
    client = LlmClient(model=args.model) if os.environ.get("ANTHROPIC_API_KEY") else None
    name = client.model if client else (args.model or "claude-haiku-5-5")
    oracle = Oracle(name, ResponseCache(args.cache), SpendLedger(args.cap_usd), client)

    only = [llm_only(r, tenants[r["tenant"]], matrix, oracle) for r in rows]
    answered = [(r, res) for r, (_, res) in zip(rows, only, strict=True) if res is not None and res.intent]
    if not answered:
        print(json.dumps({"status": "not_run", "reason": "no API key and no cached answers", "cache_misses": oracle.misses}))
        return 0
    intents = sorted({*(i for t in tenants.values() for i in t.intents)})
    fusion = ConfusionFusion.fit(((res.intent, r["intent"]) for r, res in answered), intents)

    def escalate(row, scores):
        result = oracle.ask(tenants[row["tenant"]], row["text"])
        return scoped(fusion.fuse(scores, result.intent, list(tenants[row["tenant"]].intents)), tenants[row["tenant"]]) if result and result.intent else None

    cascade = [run_conversation(r, tenants[r["tenant"]], matrix, model.predict, phrases=phrases, escalate=escalate) for r in rows]
    encoder_only = [run_conversation(r, tenants[r["tenant"]], matrix, model.predict, phrases=phrases) for r in rows]
    times = sorted(res.llm_ms for _, res in answered)
    llm_outcomes = [o for o, _ in only]
    report = {
        "reportable": bool(manifest.get("reportable")),
        "model": name,
        "fusion_fitted_on_eval_split": True,  # Replace with a separate fit before quoting anything.
        "cache_misses": oracle.misses,
        "failed_calls": oracle.failed_calls,  # Above zero, llm_only is measuring the connection, not the model.
        "ledger": oracle.ledger.as_dict(),
        "llm_latency_ms": {"p50": times[len(times) // 2], "p95": times[int(0.95 * len(times))], "n": len(times), "fastest": times[0], "within_550ms": sum(t <= 550 for t in times)},
        "arms": {"llm_only": summarise(llm_outcomes), "tfidf_only": summarise(encoder_only), "cascade": summarise(cascade)},
        "cascade_minus_llm_cost": paired_bootstrap_diff(np.array([o.cost for o in cascade]), np.array([o.cost for o in llm_outcomes])),
        "cascade_minus_tfidf_cost": paired_bootstrap_diff(np.array([o.cost for o in cascade]), np.array([o.cost for o in encoder_only])),
        "mcnemar_cascade_vs_llm": mcnemar_p(np.array([o.correct for o in cascade]), np.array([o.correct for o in llm_outcomes])),
        "calibration": calibration,
    }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if client:
        oracle.loop.run_until_complete(client.client.close())
    oracle.loop.close()
    print(json.dumps({k: report[k] for k in ("model", "failed_calls", "ledger", "llm_latency_ms")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
