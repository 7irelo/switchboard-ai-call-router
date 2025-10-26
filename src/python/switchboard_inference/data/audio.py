"""Telephone audio: 8 kHz G.711 with noise and packet loss, plus exact end-of-speech times."""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LINEAR = np.zeros(256, dtype=np.int32)
for _v in range(256):
    _u = ~_v & 0xFF
    _t = (((_u & 0x0F) << 3) + 0x84) << ((_u & 0x70) >> 4)
    _LINEAR[_v] = (0x84 - _t) if (_u & 0x80) else (_t - 0x84)


def ulaw_decode(data: bytes) -> np.ndarray:
    return _LINEAR[np.frombuffer(data, dtype=np.uint8)].astype(np.float64)


def ulaw_encode(samples: np.ndarray) -> bytes:
    s = np.clip(samples, -32635, 32635).astype(np.int64)
    sign = np.where(s < 0, 0x80, 0)
    magnitude = np.abs(s) + 0x84
    exponent = np.zeros_like(magnitude)
    for e in range(7, 0, -1):
        mask = (exponent == 0) & (magnitude >= (1 << (e + 7)))
        exponent[mask] = e
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def resample(samples: np.ndarray, source_hz: int, target_hz: int = 8000) -> np.ndarray:
    if source_hz == target_hz or len(samples) == 0:
        return samples
    # A band-limit before decimating: averaging keeps aliasing out of the telephone band.
    if source_hz > target_hz:
        width = max(1, round(source_hz / target_hz))
        samples = np.convolve(samples, np.ones(width) / width, mode="same")
    count = int(len(samples) * target_hz / source_hz)
    return np.interp(np.linspace(0, len(samples) - 1, count), np.arange(len(samples)), samples)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
        return data, w.getframerate()


@dataclass(frozen=True)
class Degrade:
    snr_db: float = 20.0
    packet_loss: float = 0.0
    trailing_silence_ms: int = 1500
    seed: int = 0


def speech_end_ms(samples: np.ndarray, hz: int = 8000, threshold_ratio: float = 0.02) -> float:
    """Where speech ends in the CLEAN signal: the ground truth for t_eos."""
    if len(samples) == 0:
        return 0.0
    frame = hz // 50
    energy = np.array([np.abs(samples[i : i + frame]).mean() for i in range(0, len(samples), frame)])
    above = np.nonzero(energy >= energy.max() * threshold_ratio)[0]
    return float((above[-1] + 1) * 20) if len(above) else 0.0


def degrade(samples: np.ndarray, hz: int, options: Degrade) -> tuple[bytes, float]:
    """Returns mu-law audio and the true end of speech in ms, measured before any noise is added."""
    rng = np.random.default_rng(options.seed)
    clean = resample(samples, hz)
    peak = np.abs(clean).max() or 1.0
    clean = clean / peak * 12000
    t_eos = speech_end_ms(clean)
    audio = np.concatenate([clean, np.zeros(options.trailing_silence_ms * 8)])
    power = np.mean(clean**2) if len(clean) else 1.0
    noise_power = power / (10 ** (options.snr_db / 10))
    audio = audio + rng.normal(0, np.sqrt(noise_power), len(audio))
    if options.packet_loss > 0:
        for start in range(0, len(audio), 160):
            if rng.random() < options.packet_loss:
                audio[start : start + 160] = 0  # What a jitter buffer without concealment leaves.
    return ulaw_encode(audio), t_eos


def audio_key(audio: bytes, engine: str, config: str) -> str:
    """Cache key for an ASR response, so evals can replay without paying again."""
    return hashlib.sha256(audio + b"|" + engine.encode() + b"|" + config.encode()).hexdigest()


def align_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Levenshtein alignment with substitution, deletion and insertion counts."""
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    rows = len(ref) + 1
    cols = len(hyp) + 1
    cost = np.zeros((rows, cols), dtype=np.int32)
    cost[:, 0] = np.arange(rows)
    cost[0, :] = np.arange(cols)
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i, j] = min(cost[i - 1, j] + 1, cost[i, j - 1] + 1, cost[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j = len(ref), len(hyp)
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i, j] == cost[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {"substitutions": int(subs), "deletions": int(dels), "insertions": int(ins), "reference_words": len(ref)}
