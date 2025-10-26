"""Offline ablations: C3 (which gate), threshold fitting, E1 (train size), E2 (noise), E3 (prefixes), and earliness.

Every arm runs on real code over a chosen dataset. Whether the numbers mean anything depends on the dataset,
and the report carries the dataset's `reportable` flag so fixture runs can't pass as results.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from switchboard_inference.data.noise import GUESSED_PROFILE, NoiseProfile, inject
from switchboard_inference.data.splits import load_split
from switchboard_inference.decision import (
    CostMatrix,
    Tenant,
    load_lexicon,
    load_tenants,
    reception_cost,
    route_cost,
)
from switchboard_inference.eval.simulate import reaches_voicemail, run_conversation, scoped
from switchboard_inference.intent.baselines import TfidfModel
from switchboard_inference.intent.calibrate import (
    conformal_threshold,
    expected_calibration_error,
    fit_temperature,
    prediction_set,
)
from switchboard_inference.train.train import prefix_rows


def train_model(rows: list[dict], calibration: list[dict], text_key: str = "text") -> TfidfModel:
    model = TfidfModel().fit([r[text_key] for r in rows], [r["intent"] for r in rows])
    index = {c: i for i, c in enumerate(model.classes)}
    usable = [r for r in calibration if r["intent"] in index]
    model.temperature = fit_temperature(model.log_probs([r["text"] for r in usable]), np.array([index[r["intent"]] for r in usable]))
    return model


def outcome_cost(t: Tenant, m: CostMatrix, row: dict, action: str, dept: str | None, at: datetime, turns: int) -> tuple[float, bool, bool]:
    """Returns cost, whether the destination was right, and whether an emergency reached voicemail."""
    intent = row["intent"]
    clarify = m.costs["clarify"] * turns if intent != "emergency" else m.costs["emergency_clarify"] * turns
    final = reception_cost(m, intent) if action == "reception" else route_cost(t, m, intent, dept, at)
    return clarify + final, dept == t.department_for(intent), intent == "emergency" and reaches_voicemail(t, dept, at)


def threshold_gate(threshold: float, by_margin: bool = False) -> Callable:
    """The baselines the expected-cost gate must beat: top-1 confidence, or the margin between the top two."""

    def gate(t: Tenant, m: CostMatrix, row: dict, score, phrases, at) -> tuple[str, str | None, int]:
        turns = 0
        text = row["text"]
        while True:
            scores = scoped(score(text), t)
            ranked = sorted(scores.values(), reverse=True) or [0.0]
            top = ranked[0]
            confidence = top - (ranked[1] if len(ranked) > 1 else 0.0) if by_margin else top
            if confidence >= threshold or turns == 2:
                if not scores:
                    return "reception", t.raw["fallback"]["department"], turns
                best = max(sorted(scores), key=lambda k: scores[k])
                return "route", t.department_for(best), turns
            turns += 1
            text = f"{text} {row['clarify_reply']}"

    return gate


def conformal_gate(q: float) -> Callable:
    """Routes when the prediction set names one department. Otherwise it asks, using the set as the options."""

    def gate(t: Tenant, m: CostMatrix, row: dict, score, phrases, at) -> tuple[str, str | None, int]:
        turns = 0
        text = row["text"]
        while True:
            scores = scoped(score(text), t)
            departments = {t.department_for(i) for i in prediction_set(scores, q)}
            if len(departments) == 1:
                return "route", departments.pop(), turns
            if turns == 2:
                if not scores:
                    return "reception", t.raw["fallback"]["department"], turns
                return "route", t.department_for(max(sorted(scores), key=lambda k: scores[k])), turns
            turns += 1
            text = f"{text} {row['clarify_reply']}"

    return gate


def summarise(costs: list[float], correct: list[bool], voicemail: list[bool], turns: list[int], rows: list[dict]) -> dict:
    emergencies = [c for c, r in zip(correct, rows, strict=True) if r["intent"] == "emergency"]
    return {
        "n": len(rows),
        "mean_cost": float(np.mean(costs)) if costs else None,
        "accuracy": float(np.mean(correct)) if correct else None,
        "emergency_recall": float(np.mean(emergencies)) if emergencies else None,
        "emergency_to_voicemail": int(sum(voicemail)),
        "clarification_rate": float(np.mean([x > 0 for x in turns])) if turns else None,
    }


def run_gate(gate, rows, tenants, matrix, score, phrases) -> dict:
    costs, correct, voicemail, turns = [], [], [], []
    for row in rows:
        t = tenants[row["tenant"]]
        at = datetime.fromisoformat(row["at"])
        action, dept, n = gate(t, matrix, row, score, phrases, at)
        cost, right, vm = outcome_cost(t, matrix, row, action, dept, at, n)
        costs.append(cost)
        correct.append(right)
        voicemail.append(vm)
        turns.append(n)
    return summarise(costs, correct, voicemail, turns, rows)


def run_expected_cost(rows, tenants, matrix, score, phrases) -> dict:
    outcomes = [run_conversation(r, tenants[r["tenant"]], matrix, score, phrases=phrases) for r in rows]
    return summarise([o.cost for o in outcomes], [o.correct for o in outcomes], [o.emergency_to_voicemail for o in outcomes], [o.clarify_turns for o in outcomes], rows)


def fit_thresholds(rows, tenants, matrix: CostMatrix, score, phrases) -> dict:
    """Grid search on calibration data for the pair with the lowest mean cost."""
    best: dict = {"mean_cost": float("inf")}
    for tau_llm in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0):
        for tau_clarify in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0):
            varied = CostMatrix(
                matrix.costs, matrix.adjacency, matrix.protected, tau_llm, tau_clarify,
                matrix.latency_penalty_per_second, matrix.latency_budget_ms, matrix.voicemail_guard,
            )  # fmt: skip
            cost = float(np.mean([run_conversation(r, tenants[r["tenant"]], varied, score, phrases=phrases).cost for r in rows]))
            if cost < best["mean_cost"] - 1e-12:
                best = {"tau_llm": tau_llm, "tau_clarify": tau_clarify, "mean_cost": cost}
    return best


def calibration_by_length(model: TfidfModel, rows: list[dict]) -> dict:
    """ECE for partials against finals: a model calibrated on finals tends to be overconfident on prefixes."""
    index = {c: i for i, c in enumerate(model.classes)}
    usable = [r for r in rows if r["intent"] in index]
    out = {}
    for fraction in (0.25, 0.5, 0.75, 1.0):
        texts = [" ".join(r["text"].split()[: max(1, int(len(r["text"].split()) * fraction))]) for r in usable]
        scored = model.predict_batch(texts)
        probs = np.array([[s[c] for c in model.classes] for s in scored])
        y = np.array([index[r["intent"]] for r in usable])
        out[str(fraction)] = {"ece": expected_calibration_error(probs, y), "accuracy": float((probs.argmax(1) == y).mean())}
    return out


def earliness(model: TfidfModel, rows: list[dict]) -> dict:
    """Share of the utterance heard at the first stable correct prediction, and flips per turn."""
    shares, flips, never = [], [], 0
    for row in rows:
        words = row["text"].split()
        previous, changed, first = None, 0, None
        for n in range(1, len(words) + 1):
            scores = model.predict(" ".join(words[:n]))
            top = max(scores, key=lambda k: scores[k])
            if previous is not None and top != previous:
                changed += 1
                first = None
            if top == row["intent"] and first is None:
                first = n
            if top != row["intent"]:
                first = None
            previous = top
        flips.append(changed)
        if first is None:
            never += 1
        else:
            shares.append(first / len(words))
    return {
        "median_share_heard": float(np.median(shares)) if shares else None,
        "never_correct_at_end": never,
        "mean_flips_per_turn": float(np.mean(flips)) if flips else None,
    }


def with_noise(rows: list[dict], profile: NoiseProfile | None, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    base = "clean_text"
    return [{**r, "text": inject(r[base], profile, rng) if profile else r[base]} for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fixture")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--measured-noise", default="runs/whisper_roundtrip/summary.json")
    parser.add_argument("--out", default="runs/ablations.json")
    args = parser.parse_args()

    root = Path(args.data)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    tenants, matrix, phrases = load_tenants(), CostMatrix.load("configs/cost_matrix.yaml"), load_lexicon()
    train, calibration, rows = load_split(root, "train"), load_split(root, "calibration"), load_split(root, args.split)
    # Harden the eval text to the measured profile when one exists: fixtures are otherwise too easy to separate arms.
    measured = None
    path = Path(args.measured_noise)
    if path.exists():
        o = json.loads(path.read_text(encoding="utf-8"))["overall"]
        measured = NoiseProfile(o["substitution"], o["deletion"], o["insertion"], "measured:whisper+espeak")
    stress = measured or GUESSED_PROFILE
    hard = with_noise(rows, stress, seed=1)
    hard_calibration = with_noise(calibration, stress, seed=2)

    model = train_model(train, calibration)
    cache: dict[str, dict[str, float]] = {}

    def score(text: str) -> dict[str, float]:
        if text not in cache:
            cache[text] = model.predict(text)
        return cache[text]

    report: dict = {
        "reportable": bool(manifest.get("reportable")),
        "provenance": manifest.get("provenance"),
        "stress_noise": stress.source,
        "stress_wer": stress.wer,
        "split": args.split,
    }
    fitted = fit_thresholds(hard_calibration, tenants, matrix, score, phrases)
    report["fitted_thresholds"] = fitted
    tuned = CostMatrix(
        matrix.costs, matrix.adjacency, matrix.protected, fitted["tau_llm"], fitted["tau_clarify"],
        matrix.latency_penalty_per_second, matrix.latency_budget_ms, matrix.voicemail_guard,
    )  # fmt: skip

    # C3: which gate?
    index = {c: i for i, c in enumerate(model.classes)}
    usable = [r for r in hard_calibration if r["intent"] in index]
    probs = np.array([[score(r["text"])[c] for c in model.classes] for r in usable])
    q = conformal_threshold(probs, np.array([index[r["intent"]] for r in usable]), alpha=0.1)
    report["C3_gate"] = {
        "top1_threshold_0.8": run_gate(threshold_gate(0.8), hard, tenants, matrix, score, phrases),
        "margin_0.3": run_gate(threshold_gate(0.3, by_margin=True), hard, tenants, matrix, score, phrases),
        "conformal_alpha_0.1": run_gate(conformal_gate(q), hard, tenants, matrix, score, phrases),
        "expected_cost_fitted": run_expected_cost(hard, tenants, tuned, score, phrases),
    }

    # E1: where does the learning curve flatten?
    report["E1_train_size"] = {}
    rng = random.Random(0)
    for size in (100, 250, 500, 1000, len(train)):
        sample = rng.sample(train, min(size, len(train)))
        if len({r["intent"] for r in sample}) < 5:
            continue
        m = train_model(sample, calibration)
        report["E1_train_size"][str(len(sample))] = run_expected_cost(hard, tenants, tuned, m.predict, phrases)

    # E2: does noise in training matter?
    report["E2_noise"] = {}
    arms = {"none": None, "guessed": GUESSED_PROFILE}
    if measured:
        arms["measured"] = measured
    for name, profile in arms.items():
        m = train_model(with_noise(train, profile, seed=3), calibration)
        report["E2_noise"][name] = run_expected_cost(hard, tenants, tuned, m.predict, phrases)

    # E3: does training on prefixes help partials, and at what cost to finals?
    report["E3_prefix"] = {}
    for name, extra in {"off": [], "on": prefix_rows(train)}.items():
        m = train_model(train + extra, calibration)
        report["E3_prefix"][name] = {"by_share_heard": calibration_by_length(m, hard), "final": run_expected_cost(hard, tenants, tuned, m.predict, phrases)}

    report["earliness"] = earliness(model, hard[:200])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("reportable", "stress_noise", "stress_wer", "fitted_thresholds", "C3_gate")}, indent=1))



if __name__ == "__main__":
    main()
