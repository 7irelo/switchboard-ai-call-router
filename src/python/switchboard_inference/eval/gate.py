"""The CI gate: absolute floors plus regression checks against an explicit baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GATED_ARM = "tfidf_cascade_no_llm"
ACCURACY_SLACK = 0.02  # Fixture runs are deterministic, so this only absorbs intended small changes.
COST_SLACK = 0.03


def check(report: dict, baseline: dict) -> list[str]:
    failures: list[str] = []
    arm = report["arms"][GATED_ARM]
    base = baseline["arms"][GATED_ARM]

    # Absolute floors, which apply whatever the baseline says.
    if arm["emergency_to_voicemail"] != 0:
        failures.append(f"emergency to voicemail = {arm['emergency_to_voicemail']}, floor is 0")
    floor = baseline.get("emergency_recall_floor", 0.98)
    if arm["emergency_recall"] is not None and arm["emergency_recall"] < floor:
        failures.append(f"emergency recall {arm['emergency_recall']:.3f} is below floor {floor}")

    # Regressions.
    if arm["accuracy"] < base["accuracy"] - ACCURACY_SLACK:
        failures.append(f"accuracy fell from {base['accuracy']:.3f} to {arm['accuracy']:.3f}")
    if arm["mean_cost"] > base["mean_cost"] + COST_SLACK:
        failures.append(f"mean cost rose from {base['mean_cost']:.3f} to {arm['mean_cost']:.3f}")
    for tenant, stats in base["per_tenant"].items():
        now = arm["per_tenant"].get(tenant)
        if now is None:
            failures.append(f"tenant {tenant} is missing from the run")
        elif now["accuracy"] < stats["accuracy"] - ACCURACY_SLACK:
            failures.append(f"{tenant} accuracy fell from {stats['accuracy']:.3f} to {now['accuracy']:.3f}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/fixture/report.json")
    parser.add_argument("--baseline", default="configs/eval/baseline_fixture.json")
    args = parser.parse_args()
    report = json.loads(Path(args.run).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    failures = check(report, baseline)
    for failure in failures:
        print(f"GATE FAIL: {failure}")
    if not failures:
        print("gate passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
