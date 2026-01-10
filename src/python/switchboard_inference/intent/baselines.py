"""The baselines the encoder has to beat: keyword rules and char-n-gram TF-IDF."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from switchboard_inference.decision import CANONICAL_INTENTS
from switchboard_inference.intent.calibrate import fit_temperature

KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": ("bill", "invoice", "statement", "charged", "charge", "imali engaka"),
    "payments_arrears": ("pay", "arrears", "overdue", "behind on", "payment plan", "ukukhokha"),
    "sales_new": ("sign up", "new service", "enrol", "enrolment", "quote", "interested in"),
    "sales_upgrade": ("upgrade", "faster", "bigger package"),
    "cancel": ("cancel", "terminate", "end my contract", "kanselleer"),
    "tech_support": ("router", "password", "not working on my", "my device", "setting up"),
    "outage_fault": ("outage", "no water", "no electricity", "whole area", "no power", "off since", "akukho manzi"),
    "appointment_book": ("book", "appointment", "make a time"),
    "appointment_change": ("reschedule", "move my appointment", "change my appointment"),
    "order_status": ("status", "where is my", "application", "still waiting"),
    "complaint": ("complain", "complaint", "unhappy", "disappointed", "disaster"),
    "account_admin": ("update my details", "change my address", "my details", "certificate"),
    "hours_location": ("open", "hours", "where are you", "address", "close"),
    "speak_to_human": ("human", "real person", "agent", "speak to someone"),
    "emergency": ("emergency", "not breathing", "on fire", "bleeding", "gas leak", "live wire", "burst"),
    "wrong_number": ("wrong number", "dialled by mistake", "who is this"),
    "out_of_scope": ("pizza", "weather", "lottery"),
}


class KeywordModel:
    version = "keyword-v1"

    def predict(self, text: str) -> dict[str, float]:
        lowered = text.lower()
        hits = {i: sum(1.0 for k in kws if k in lowered) for i, kws in KEYWORDS.items()}
        total = sum(hits.values())
        if total == 0:
            return {}  # No evidence: the decision layer treats this as a uniform prior.
        return {i: h / total for i, h in hits.items() if h > 0}


class TfidfModel:
    """Char n-grams hold up against ASR noise and agglutination. Temperature is fitted separately."""

    version = "tfidf-char-v1"

    # Word-count buckets. Partials are much less reliable than their probabilities suggest.
    BUCKETS = ((1, 3), (4, 6), (7, 10_000))

    def __init__(self, temperature: float = 1.0):
        self.bucket_temperatures: dict[int, float] = {}
        self.pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True),
            LogisticRegression(C=20.0, max_iter=2000, class_weight="balanced"),
        )
        self.temperature = temperature

    def fit(self, texts: list[str], labels: list[str]) -> TfidfModel:
        self.pipeline.fit(texts, labels)
        return self

    def log_probs(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_log_proba(texts)

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_)

    def temperature_for(self, text: str) -> float:
        # An older pickled model has no such attribute, and falls back to one temperature.
        buckets = getattr(self, "bucket_temperatures", {})
        words = len(text.split())
        for index, (low, high) in enumerate(self.BUCKETS):
            if low <= words <= high and index in buckets:
                return buckets[index]
        return self.temperature

    def fit_prefix_calibration(self, rows: list[dict], min_rows: int = 30) -> dict[int, float]:
        """Fits one temperature per length bucket, on truncated text from noisy transcripts."""
        index = {c: i for i, c in enumerate(self.classes)}
        samples: dict[int, tuple[list[str], list[int]]] = {i: ([], []) for i in range(len(self.BUCKETS))}
        for row in rows:
            if row["intent"] not in index:
                continue
            words = row["text"].split()
            for fraction in (0.25, 0.5, 0.75, 1.0):
                text = " ".join(words[: max(1, int(len(words) * fraction))])
                count = len(text.split())
                for bucket, (low, high) in enumerate(self.BUCKETS):
                    if low <= count <= high:
                        samples[bucket][0].append(text)
                        samples[bucket][1].append(index[row["intent"]])
        fitted = {}
        for bucket, (texts, labels) in samples.items():
            if len(texts) >= min_rows:
                fitted[bucket] = fit_temperature(self.log_probs(texts), np.array(labels))
        self.bucket_temperatures = fitted
        return fitted

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        temperatures = np.array([[self.temperature_for(t)] for t in texts])
        scaled = self.log_probs(texts) / temperatures
        probs = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return [dict(zip(self.classes, row.tolist(), strict=True)) for row in probs]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> TfidfModel:
        with open(path, "rb") as f:
            return pickle.load(f)


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
assert set(KEYWORDS) == set(CANONICAL_INTENTS)
