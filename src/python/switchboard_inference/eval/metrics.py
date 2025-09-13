"""Statistics for the pre-registered metrics (docs/METRICS.md)."""

from __future__ import annotations

import math

import numpy as np


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rule_of_three_upper(n: int) -> float:
    """95% upper bound on a rate when zero events were seen in n trials."""
    return 1.0 if n == 0 else min(1.0, 3 / n)


def bootstrap_mean_ci(values: np.ndarray, resamples: int = 10000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[picks].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return (float(low), float(high))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, resamples: int = 10000, seed: int = 0) -> dict[str, float]:
    """Difference in means (a - b), paired by conversation."""
    if len(a) != len(b):
        raise ValueError("paired arrays must have the same length")
    diff = a - b
    low, high = bootstrap_mean_ci(diff, resamples, seed)
    return {"mean_diff": float(diff.mean()) if len(diff) else float("nan"), "ci_low": low, "ci_high": high}


def mcnemar_p(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """Exact two-sided McNemar test on discordant pairs."""
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values across the primary comparisons."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def percentile(values: list[float] | np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")
