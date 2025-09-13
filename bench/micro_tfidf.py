"""Stage micro-benchmark for the TF-IDF baseline on one pinned thread. Trains on the fixture split."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from switchboard_inference.data.splits import load_split  # noqa: E402
from switchboard_inference.intent.baselines import TfidfModel  # noqa: E402


def main() -> None:
    data = Path("data/fixture")
    train = load_split(data, "train")
    dev = load_split(data, "dev")
    model = TfidfModel().fit([r["text"] for r in train], [r["intent"] for r in train])
    for row in dev[:50]:
        model.predict(row["text"])  # Warm-up.
    samples = []
    for _ in range(3):
        for row in dev:
            started = time.perf_counter()
            model.predict(row["text"])
            samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    report = {
        "stage": "tfidf_predict_single_thread",
        "n": len(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": samples[int(0.95 * len(samples))],
        "p99_ms": samples[int(0.99 * len(samples))],
        "budget_ms": 30,
        "pinned_core": False,
    }
    print(json.dumps(report))
    Path("runs").mkdir(exist_ok=True)
    Path("runs/micro_tfidf.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
