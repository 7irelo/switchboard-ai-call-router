"""Runs only where the optional ML extras and an exported model exist: see docs/MODEL_CARD.md."""

import json
from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

MODEL = Path("runs/encoder-code-check")
pytestmark = pytest.mark.skipif(not (MODEL / "model.onnx").exists(), reason="no exported model here")


def test_onnx_encoder_returns_a_distribution_over_its_labels():
    from switchboard_inference.intent.encoder import OnnxEncoder

    encoder = OnnxEncoder(MODEL)
    scores = encoder.predict("my bill is wrong this month")
    assert set(scores) == set(json.loads((MODEL / "labels.json").read_text(encoding="utf-8")))
    assert sum(scores.values()) == pytest.approx(1)


def test_temperature_flattens_the_distribution():
    from switchboard_inference.intent.encoder import OnnxEncoder

    hot = OnnxEncoder(MODEL, temperature=100.0).predict("my bill is wrong")
    cold = OnnxEncoder(MODEL, temperature=0.5).predict("my bill is wrong")
    assert max(hot.values()) < max(cold.values())
