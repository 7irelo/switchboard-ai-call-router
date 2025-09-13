"""Temperature scaling and calibration metrics. Fit on noise that matches serving, not on clean text."""

from __future__ import annotations

import numpy as np


def nll(log_probs: np.ndarray, y: np.ndarray, temperature: float) -> float:
    scaled = log_probs / temperature
    scaled = scaled - np.logaddexp.reduce(scaled, axis=1, keepdims=True)
    return float(-scaled[np.arange(len(y)), y].mean())


def fit_temperature(log_probs: np.ndarray, y: np.ndarray, low: float = 0.05, high: float = 10.0) -> float:
    """Golden-section search. Negative log-likelihood is unimodal in temperature here."""
    ratio = (5**0.5 - 1) / 2
    a, b = np.log(low), np.log(high)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(log_probs, y, np.exp(c)), nll(log_probs, y, np.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = nll(log_probs, y, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = nll(log_probs, y, np.exp(d))
    return float(np.exp((a + b) / 2))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def conformal_threshold(probs: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> float:
    """Split conformal on 1 - p(true). Prediction sets also give the clarifying question its options."""
    scores = 1 - probs[np.arange(len(y)), y]
    n = len(scores)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q, method="higher"))


def prediction_set(row: dict[str, float], threshold: float) -> list[str]:
    return [i for i, p in sorted(row.items(), key=lambda kv: -kv[1]) if 1 - p <= threshold]
