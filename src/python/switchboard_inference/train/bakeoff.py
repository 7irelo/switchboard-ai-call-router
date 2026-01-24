"""Encoder bake-off against TF-IDF, scored on data the models didn't train on.

Training rows are template fixtures, so this result is NOT reportable. It shows the procedure working and gives
a first signal on real Whisper transcripts and unseen edge cases. The stop rule applies to the reportable rerun.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from switchboard_inference.data.splits import load_split
from switchboard_inference.decision import CostMatrix, load_lexicon, load_tenants
from switchboard_inference.eval.ablate import edge_case_rows, real_asr_rows, run_expected_cost
from switchboard_inference.intent.baselines import TfidfModel
from switchboard_inference.intent.calibrate import fit_temperature


class TorchScorer:
    def __init__(self, directory: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
        self.labels = json.loads((directory.parent / "labels.json").read_text(encoding="utf-8"))
        self.temperature = 1.0
        self.cache: dict[str, np.ndarray] = {}

    def logits(self, text: str) -> np.ndarray:
        if text not in self.cache:
            with self.torch.no_grad():
                encoded = self.tokenizer([text], truncation=True, max_length=64, return_tensors="pt")
                self.cache[text] = self.model(**encoded).logits[0].numpy()
        return self.cache[text]

    def predict(self, text: str) -> dict[str, float]:
        scaled = self.logits(text) / self.temperature
        probs = np.exp(scaled - scaled.max())
        probs /= probs.sum()
        return dict(zip(self.labels, probs.tolist(), strict=True))

    def calibrate(self, rows: list[dict]) -> None:
        index = {c: i for i, c in enumerate(self.labels)}
        usable = [r for r in rows if r["intent"] in index]
        raw = np.stack([self.logits(r["text"]) for r in usable])
        log_probs = raw - np.logaddexp.reduce(raw, axis=1, keepdims=True)
        self.temperature = fit_temperature(log_probs, np.array([index[r["intent"]] for r in usable]))


def split_half(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2 :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="minilm,e5-small,afro-mini")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--data", default="data/fixture")
    parser.add_argument("--out", default="runs/bakeoff.json")
    args = parser.parse_args()

    root = Path(args.data)
    tenants, matrix, phrases = load_tenants(), CostMatrix.load("configs/cost_matrix.yaml"), load_lexicon()
    asr_cal, asr_test = split_half(real_asr_rows("runs/whisper_roundtrip/rows.jsonl"), 0)
    edge_cal, edge_test = split_half(edge_case_rows(tenants), 1)
    calibration = asr_cal + edge_cal
    report: dict = {"reportable": False, "train_rows": args.limit, "epochs": args.epochs, "device": "cpu", "pinned_core": False, "arms": {}}

    def evaluate(name: str, scorer, extra: dict) -> None:
        arm = {
            "real_asr": run_expected_cost(asr_test, tenants, matrix, scorer.predict, phrases),
            "edge_cases": run_expected_cost(edge_test, tenants, matrix, scorer.predict, phrases),
            **extra,
        }
        samples = []
        for row in asr_test[:40]:
            started = time.perf_counter()
            scorer.predict(row["text"] + " ")  # A new string, so the scorer's cache can't flatter it.
            samples.append((time.perf_counter() - started) * 1000)
        arm["predict_p50_ms"] = sorted(samples)[len(samples) // 2]
        report["arms"][name] = arm
        print(name, json.dumps({k: arm[k]["mean_cost"] for k in ("real_asr", "edge_cases")}), f"p50={arm['predict_p50_ms']:.1f}ms", flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")

    train = random.Random(0).sample(load_split(root, "train"), args.limit)
    baseline = TfidfModel().fit([r["text"] for r in train], [r["intent"] for r in train])
    index = {c: i for i, c in enumerate(baseline.classes)}
    usable = [r for r in calibration if r["intent"] in index]
    baseline.temperature = fit_temperature(baseline.log_probs([r["text"] for r in usable]), np.array([index[r["intent"]] for r in usable]))
    evaluate("tfidf", baseline, {"temperature": baseline.temperature})

    for candidate in args.candidates.split(","):
        out = Path(f"runs/bakeoff-{candidate}")
        result = subprocess.run(
            [sys.executable, "-m", "switchboard_inference.train.train", "--candidate", candidate, "--data", str(root), "--out", str(out),
             "--epochs", str(args.epochs), "--limit", str(args.limit), "--allow-fixture"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )  # fmt: skip
        if result.returncode != 0 or not (out / "best").exists():
            report["arms"][candidate] = {"failed": result.stderr[-400:]}
            print(candidate, "FAILED", result.stderr[-300:], flush=True)
            continue
        scorer = TorchScorer(out / "best")
        scorer.calibrate(calibration)
        evaluate(candidate, scorer, {"temperature": scorer.temperature})

    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
