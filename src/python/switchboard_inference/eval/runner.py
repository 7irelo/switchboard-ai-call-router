"""One command: every arm on a split, per-tenant metrics, confidence intervals, a generated report.

Arms with no real data behind them are reported as "not run". They are never simulated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from switchboard_inference.data.splits import load_split
from switchboard_inference.decision import CostMatrix, load_lexicon, load_tenants
from switchboard_inference.eval.metrics import (
    bootstrap_mean_ci,
    holm,
    mcnemar_p,
    paired_bootstrap_diff,
    rule_of_three_upper,
    wilson,
)
from switchboard_inference.eval.simulate import Outcome, run_conversation
from switchboard_inference.intent.baselines import KeywordModel, TfidfModel
from switchboard_inference.intent.calibrate import expected_calibration_error, fit_temperature

BANNER = (
    "FIXTURE RUN: synthetic templates with guessed noise. These numbers test the pipeline. "
    "They are not evidence about any model, and must not be quoted as results."
)


def summarise(outcomes: list[Outcome]) -> dict:
    n = len(outcomes)
    correct = sum(o.correct for o in outcomes)
    costs = np.array([o.cost for o in outcomes])
    emergencies = [o for o in outcomes if o.intent == "emergency"]
    reached = sum(o.reached_emergency for o in emergencies)
    voicemail = sum(o.emergency_to_voicemail for o in outcomes)
    clarified = [o for o in outcomes if o.clarify_turns > 0]
    per_tenant = {}
    for tenant in sorted({o.tenant for o in outcomes}):
        rows = [o for o in outcomes if o.tenant == tenant]
        hits = sum(o.correct for o in rows)
        per_tenant[tenant] = {
            "n": len(rows),
            "accuracy": hits / len(rows),
            "accuracy_ci": wilson(hits, len(rows)),
            "mean_cost": float(np.mean([o.cost for o in rows])),
        }
    per_mix = {}
    for mix in sorted({o.mix for o in outcomes}):
        rows = [o for o in outcomes if o.mix == mix]
        per_mix[mix] = {"n": len(rows), "accuracy": sum(o.correct for o in rows) / len(rows)}
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "accuracy_ci": wilson(correct, n),
        "mean_cost": float(costs.mean()) if n else 0.0,
        "mean_cost_ci": bootstrap_mean_ci(costs),
        "emergency_n": len(emergencies),
        "emergency_recall": reached / len(emergencies) if emergencies else None,
        "emergency_recall_ci": wilson(reached, len(emergencies)),
        "emergency_to_voicemail": voicemail,
        "emergency_to_voicemail_upper": rule_of_three_upper(len(emergencies)) if voicemail == 0 else None,
        "clarification_rate": len(clarified) / n if n else 0.0,
        "clarification_success": sum(o.correct for o in clarified) / len(clarified) if clarified else None,
        "escalation_rate": sum(o.escalated for o in outcomes) / n if n else 0.0,
        "per_tenant": per_tenant,
        "per_mix": per_mix,
    }


def fit_tfidf(data: Path) -> tuple[TfidfModel, dict]:
    train = load_split(data, "train")
    calibration = load_split(data, "calibration")
    model = TfidfModel().fit([r["text"] for r in train], [r["intent"] for r in train])
    index = {c: i for i, c in enumerate(model.classes)}
    usable = [r for r in calibration if r["intent"] in index]
    log_probs = model.log_probs([r["text"] for r in usable])
    y = np.array([index[r["intent"]] for r in usable])
    before = expected_calibration_error(np.exp(log_probs), y)
    model.temperature = fit_temperature(log_probs, y)
    scaled = log_probs / model.temperature
    probs = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return model, {
        "temperature": model.temperature,
        "ece_before": before,
        "ece_after": expected_calibration_error(probs, y),
        "calibration_rows": len(usable),
        "dropped_unseen_labels": len(calibration) - len(usable),
    }


def run(data: Path, split: str, out: Path, *, unseal: bool = False, reason: str = "", matrix_path: str = "configs/cost_matrix.yaml") -> dict:
    tenants = load_tenants()
    matrix = CostMatrix.load(matrix_path)
    phrases = load_lexicon()
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rows = load_split(data, split, unseal=unseal, reason=reason)
    model, calibration = fit_tfidf(data)
    keyword = KeywordModel()
    cache: dict[str, dict[str, float]] = {}

    def tfidf_score(text: str) -> dict[str, float]:
        if text not in cache:
            cache[text] = model.predict(text)
        return cache[text]

    def conversation_arm(score, **options) -> list[Outcome]:
        return [run_conversation(r, tenants[r["tenant"]], matrix, score, phrases=phrases, **options) for r in rows]

    arms = {
        "keyword_top1": conversation_arm(keyword.predict, gate=False),
        "tfidf_top1": conversation_arm(tfidf_score, gate=False),
        "keyword_gated": conversation_arm(keyword.predict),
        "tfidf_gated_no_clarify": conversation_arm(tfidf_score, allow_clarify=False),
        "tfidf_cascade_no_llm": conversation_arm(tfidf_score),
    }

    report = {
        "banner": BANNER if not manifest.get("reportable") else None,
        "reportable": bool(manifest.get("reportable")),
        "provenance": manifest.get("provenance"),
        "split": split,
        "dataset_sha256": manifest["splits"][split]["sha256"],
        "calibration": calibration,
        "arms": {name: summarise(outcomes) for name, outcomes in arms.items()},
        "not_run": {
            "llm_only": "needs ANTHROPIC_API_KEY and recorded responses",
            "cascade_with_llm": "needs ANTHROPIC_API_KEY and recorded responses",
            "encoder": "needs a trained ONNX model (M5)",
            "t_route": "needs the live stack: see bench/ and docs/REPORT.md",
        },
    }

    base = arms["tfidf_top1"]
    challenger = arms["tfidf_cascade_no_llm"]
    report["comparison"] = {
        "cascade_no_llm_minus_top1_cost": paired_bootstrap_diff(
            np.array([o.cost for o in challenger]), np.array([o.cost for o in base])
        ),
        "holm_p": holm(
            {
                "cascade_vs_tfidf_top1": mcnemar_p(np.array([o.correct for o in challenger]), np.array([o.correct for o in base])),
                "cascade_vs_keyword_top1": mcnemar_p(
                    np.array([o.correct for o in challenger]), np.array([o.correct for o in arms["keyword_top1"]])
                ),
            }
        ),
    }
    report["sensitivity"] = sensitivity(rows, tenants, matrix, tfidf_score, keyword.predict, phrases)

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "outcomes.jsonl").write_text(
        "".join(json.dumps(o.__dict__) + "\n" for name, outcomes in arms.items() if name == "tfidf_cascade_no_llm" for o in outcomes),
        encoding="utf-8",
    )
    (out / "report.md").write_text(render(report), encoding="utf-8")
    return report


GROUPS = {
    "misroute": ["adjacent_department", "unrelated_department", "voicemail_while_open"],
    "clarify": ["clarify", "reception"],
    "emergency": ["emergency_to_wrong_live_department", "emergency_clarify", "emergency_to_voicemail", "non_emergency_to_emergency_route"],
}


def sensitivity(rows, tenants, matrix, score, keyword_score, phrases) -> dict:
    """Vary each cost group by x0.5 and x2. A winner that changes isn't reported as a finding."""
    results = {}
    for group, keys in GROUPS.items():
        for factor in (0.5, 2.0):
            costs = dict(matrix.costs)
            for k in keys:
                costs[k] *= factor
            varied = CostMatrix(
                costs, matrix.adjacency, matrix.protected, matrix.tau_llm, matrix.tau_clarify,
                matrix.latency_penalty_per_second, matrix.latency_budget_ms, matrix.voicemail_guard,
            )  # fmt: skip
            cost_of = {
                "tfidf_cascade_no_llm": np.mean([run_conversation(r, tenants[r["tenant"]], varied, score, phrases=phrases).cost for r in rows]),
                "tfidf_top1": np.mean([run_conversation(r, tenants[r["tenant"]], varied, score, phrases=phrases, gate=False).cost for r in rows]),
                "keyword_top1": np.mean([run_conversation(r, tenants[r["tenant"]], varied, keyword_score, phrases=phrases, gate=False).cost for r in rows]),
            }
            winner = min(cost_of, key=lambda k: cost_of[k])
            results[f"{group}x{factor}"] = {"winner": winner, "costs": {k: float(v) for k, v in cost_of.items()}}
    winners = {v["winner"] for v in results.values()}
    return {"stable": len(winners) == 1, "winners": sorted(winners), "cells": results}


def render(report: dict) -> str:
    lines = ["# Evaluation report (generated)", ""]
    if report["banner"]:
        lines += [f"> **{report['banner']}**", ""]
    lines += [
        f"Split `{report['split']}`, dataset sha256 `{report['dataset_sha256'][:16]}`.", "",
        "| Arm | n | Accuracy | Mean cost | Emergency recall | Emergency to voicemail | Clarify rate | Clarify success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]  # fmt: skip
    for name, a in report["arms"].items():
        recall = "n/a" if a["emergency_recall"] is None else f"{a['emergency_recall']:.3f}"
        success = "n/a" if a["clarification_success"] is None else f"{a['clarification_success']:.3f}"
        lines.append(
            f"| {name} | {a['n']} | {a['accuracy']:.3f} | {a['mean_cost']:.3f} | {recall} | {a['emergency_to_voicemail']} | {a['clarification_rate']:.3f} | {success} |"
        )
    lines += ["", "## Not run", ""]
    lines += [f"- `{k}`: {v}" for k, v in report["not_run"].items()]
    s = report["sensitivity"]
    lines += ["", f"## Sensitivity: {'winner stable' if s['stable'] else 'winner CHANGES with costs, so no finding is reported'} ({', '.join(s['winners'])})", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fixture")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--out", default="runs/fixture")
    parser.add_argument("--unseal", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    report = run(Path(args.data), args.split, Path(args.out), unseal=args.unseal, reason=args.reason)
    print(render(report))


if __name__ == "__main__":
    main()
