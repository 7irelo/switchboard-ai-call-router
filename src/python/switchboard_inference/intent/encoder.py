"""Serves an exported encoder through ONNX Runtime. Exercised on a code-check export: no real model is trained yet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class OnnxEncoder:
    """Expects model.onnx, tokenizer.json and labels.json. Temperature comes from calibration."""

    def __init__(self, directory: str | Path, threads: int = 1, temperature: float = 1.0, max_length: int = 64):
        import onnxruntime as ort  # Optional extra.
        from tokenizers import Tokenizer

        root = Path(directory)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(root / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.labels: list[str] = json.loads((root / "labels.json").read_text(encoding="utf-8"))
        self.temperature = temperature
        self.version = f"onnx-{root.name}"
        self.inputs = {i.name for i in self.session.get_inputs()}

    def logits(self, text: str) -> np.ndarray:
        encoded = self.tokenizer.encode(text)
        feed = {
            "input_ids": np.array([encoded.ids], dtype=np.int64),
            "attention_mask": np.array([encoded.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([encoded.type_ids], dtype=np.int64)
        return self.session.run(None, feed)[0][0]

    def predict(self, text: str) -> dict[str, float]:
        scaled = self.logits(text) / self.temperature
        probs = np.exp(scaled - scaled.max())
        probs /= probs.sum()
        return dict(zip(self.labels, probs.tolist(), strict=True))
