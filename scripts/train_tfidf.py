"""Trains the TF-IDF baseline for serving. Output is gitignored and mounted into the inference container.

Temperature comes from real ASR transcripts when they exist. Fixture text is so separable that fitting on
it pushes temperature to its lower bound, leaving a model wildly overconfident on anything real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from switchboard_inference.eval.ablate import real_asr_rows
from switchboard_inference.eval.runner import fit_tfidf
from switchboard_inference.intent.calibrate import expected_calibration_error, fit_temperature

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model, calibration = fit_tfidf(ROOT / "data/fixture")
    report = {"fixture_temperature": calibration["temperature"], "source": "fixture"}
    rows = real_asr_rows(ROOT / "runs/whisper_roundtrip/rows.jsonl")
    index = {c: i for i, c in enumerate(model.classes)}
    usable = [r for r in rows if r["intent"] in index and r["text"].strip()]
    if usable:
        log_probs = model.log_probs([r["text"] for r in usable])
        y = np.array([index[r["intent"]] for r in usable])
        model.temperature = fit_temperature(log_probs, y)
        scaled = log_probs / model.temperature
        probs = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        report |= {"source": "real_asr", "rows": len(usable), "ece_after": expected_calibration_error(probs, y)}
        report["bucket_temperatures"] = model.fit_prefix_calibration(usable)
    report["temperature"] = model.temperature
    model.save(ROOT / "deploy/models/tfidf.pkl")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
