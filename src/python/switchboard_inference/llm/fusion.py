"""Turns "the LLM said k" into probabilities, using an empirical confusion table fitted on dev."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path


class ConfusionFusion:
    """P(true = i | LLM said k), with add-alpha smoothing, multiplied into the encoder's posterior."""

    def __init__(self, table: dict[str, dict[str, float]], default_precision: float = 0.7):
        self.table = table
        self.default_precision = default_precision

    @classmethod
    def fit(cls, pairs: Iterable[tuple[str, str]], intents: list[str], alpha: float = 1.0) -> ConfusionFusion:
        counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for said, true in pairs:
            counts[said][true] += 1
        table = {}
        for said in intents:
            total = sum(counts[said].values()) + alpha * len(intents)
            table[said] = {i: (counts[said][i] + alpha) / total for i in intents}
        return cls(table)

    def likelihood(self, said: str, intent: str, n: int) -> float:
        row = self.table.get(said)
        if row and intent in row:
            return row[intent]
        # A tenant's custom intent has no table row: a stated assumption, not a measured figure.
        return self.default_precision if intent == said else (1 - self.default_precision) / max(n - 1, 1)

    def fuse(self, prior: dict[str, float], said: str, intents: list[str]) -> dict[str, float]:
        n = len(intents)
        floor = 1e-4  # A confident encoder can't veto the LLM completely.
        scores = {i: (prior.get(i, 0.0) + floor) * self.likelihood(said, i, n) for i in intents}
        total = sum(scores.values())
        return {i: s / total for i, s in scores.items()}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.table, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ConfusionFusion:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
