"""Tests the gate itself: real seeded regressions must turn it red, on real eval runs."""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

from switchboard_inference.data.splits import write_dataset
from switchboard_inference.eval.gate import check
from switchboard_inference.eval.runner import run

DATA = Path("data/fixture")
BASELINE = Path("configs/eval/baseline_fixture.json")


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems: list[str] = []
    with tempfile.TemporaryDirectory(prefix="sb-gate-") as tmp:
        root = Path(tmp)

        # Seed 1: corrupted labels in training, so the model learns a broken mapping.
        broken = root / "broken_labels"
        shutil.copytree(DATA, broken)
        rows = {
            name: [json.loads(x) for x in (broken / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x]
            for name in ("train", "calibration", "dev", "test")
        }
        for row in rows["train"]:
            if row["intent"] in {"billing", "outage_fault", "emergency"}:
                row["intent"] = "complaint"
        write_dataset(rows, broken, {"reportable": False, "provenance": "gate_selftest"})
        report = run(broken, "dev", root / "broken_labels_run")
        failures = check(report, baseline)
        print(f"seed broken_labels: gate {'RED' if failures else 'GREEN'} {failures[:2]}")
        if not failures:
            problems.append("broken labels did not trip the gate")

        # Seed 2: badly garbled input, as if ASR had collapsed. Thresholds can't be the seed here, because
        # the model is near-perfect on fixtures, so they never come into play (docs/failure-log.md, entry 8).
        garbled = root / "garbled"
        shutil.copytree(DATA, garbled)
        rows = {
            name: [json.loads(x) for x in (garbled / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x]
            for name in ("train", "calibration", "dev", "test")
        }
        rng = random.Random(0)
        for row in rows["dev"]:
            row["text"] = " ".join(w for w in row["text"].split() if rng.random() > 0.8)
            row["clarify_reply"] = ""
        write_dataset(rows, garbled, {"reportable": False, "provenance": "gate_selftest"})
        report = run(garbled, "dev", root / "garbled_run")
        failures = check(report, baseline)
        print(f"seed garbled_input: gate {'RED' if failures else 'GREEN'} {failures[:2]}")
        if not failures:
            problems.append("garbled input did not trip the gate")

    for problem in problems:
        print(f"SELF-TEST FAIL: {problem}")
    if not problems:
        print("gate self-test passed: every seeded regression turned the gate red")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
