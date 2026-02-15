"""Record and replay LLM answers, plus a spend ledger, so evals only pay once.

The cache key includes the tenant, because prompts differ per tenant and no response may cross tenants.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from switchboard_inference.llm.client import LlmResult


def cache_key(tenant: str, model: str, transcript: str, allowed: list[str], prompt_override: str) -> str:
    material = json.dumps([tenant, model, transcript, sorted(allowed), prompt_override], ensure_ascii=False)
    return hashlib.sha256(material.encode()).hexdigest()


# A failed call says nothing about the model, and caching one would replay the failure as an answer forever.
UNCACHEABLE_OUTCOMES = frozenset({"unavailable", "error"})


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line:
                    record = json.loads(line)
                    if record["result"].get("outcome") not in UNCACHEABLE_OUTCOMES:
                        self.entries[record["key"]] = record["result"]

    def get(self, key: str) -> LlmResult | None:
        return LlmResult(**self.entries[key]) if key in self.entries else None

    def put(self, key: str, result: LlmResult) -> None:
        if key in self.entries or result.outcome in UNCACHEABLE_OUTCOMES:
            return
        self.entries[key] = asdict(result)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"key": key, "result": asdict(result)}) + "\n")


class SpendLedger:
    """Refuses further live calls past a hard cap, so a bad loop can't run up a bill."""

    def __init__(self, cap_usd: float):
        self.cap_usd = cap_usd
        self.by_model: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    @property
    def total(self) -> float:
        return sum(self.by_model.values())

    def allow(self) -> bool:
        return self.total < self.cap_usd

    def record(self, result: LlmResult) -> None:
        self.by_model[result.model] += result.cost_usd
        self.calls[result.model] += 1

    def cost_per_1k(self, conversations: int) -> float:
        return self.total / conversations * 1000 if conversations else 0.0

    def as_dict(self) -> dict:
        return {"cap_usd": self.cap_usd, "total_usd": round(self.total, 6), "by_model": dict(self.by_model), "calls": dict(self.calls)}
