"""Write shared decision vectors. Both the Python and .NET engines are tested against them."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from switchboard_inference.decision import CostMatrix, decide, load_tenants

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261103
COUNT = 600
TIMES = [
    "2026-11-04T08:00:00Z",  # Wednesday 10:00 SAST, most departments open
    "2026-11-04T20:30:00Z",  # Wednesday 22:30 SAST, most departments closed
    "2026-11-07T08:30:00Z",  # Saturday 10:30 SAST
    "2026-12-25T09:00:00Z",  # Holiday
    "2026-11-02T05:30:00Z",  # Monday 07:30 SAST, a window boundary
    "2026-11-02T14:30:00Z",  # Monday 16:30 SAST, a window end
]


def random_scores(rng: random.Random, intents: tuple[str, ...]) -> dict[str, float]:
    shape = rng.choice(["peaked", "two", "flat", "emergency_tail", "empty"])
    if shape == "empty":
        return {}
    if shape == "flat":
        return {i: rng.random() for i in intents}
    picks = rng.sample(list(intents), 2)
    if shape == "peaked":
        return {picks[0]: rng.uniform(0.8, 1.0), picks[1]: rng.uniform(0.0, 0.2)}
    if shape == "two":
        share = rng.uniform(0.4, 0.6)
        return {picks[0]: share, picks[1]: 1 - share}
    return {picks[0]: rng.uniform(0.7, 0.95), "emergency": rng.uniform(0.03, 0.25)}


def main() -> None:
    matrix = CostMatrix.load(ROOT / "configs/cost_matrix.yaml")
    tenants = load_tenants(ROOT / "configs/tenants")
    rng = random.Random(SEED)
    vectors = []
    for _ in range(COUNT):
        tenant = tenants[rng.choice(sorted(tenants))]
        scores = {k: round(v, 6) for k, v in random_scores(rng, tenant.intents).items()}
        if rng.random() < 0.05:
            scores["some_other_tenants_intent"] = 0.5  # must be ignored
        at_text = rng.choice(TIMES)
        at = datetime.fromisoformat(at_text.replace("Z", "+00:00")).astimezone(UTC)
        if rng.random() < 0.2:
            at += timedelta(minutes=rng.randint(0, 600))
        args = {
            "clarify_turns": rng.choice([0, 0, 1, 2]),
            "llm_contributed": rng.random() < 0.5,
            "llm_available": rng.random() < 0.8,
            "degraded": rng.random() < 0.2,
            "dtmf": rng.choice([None, None, None, None, "1", "9", "7"]),
            "lexicon_hit": rng.random() < 0.15,
        }
        result = decide(tenant, matrix, scores, at, **args)
        vectors.append(
            {
                "tenant": tenant.id,
                "at": at.isoformat().replace("+00:00", "Z"),
                "scores": scores,
                **args,
                "expect": {
                    "action": result.action,
                    "department": result.department,
                    "expected_cost": round(result.expected_cost, 9),
                    "reason": result.reason,
                    "candidates": list(result.candidates),
                    "closed": result.closed,
                },
            }
        )
    out = ROOT / "tests/golden/decision_vectors.json"
    out.write_text(json.dumps(vectors, indent=1) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for v in vectors:
        counts[v["expect"]["reason"]] = counts.get(v["expect"]["reason"], 0) + 1
    print(f"wrote {len(vectors)} vectors: {counts}")


if __name__ == "__main__":
    main()
