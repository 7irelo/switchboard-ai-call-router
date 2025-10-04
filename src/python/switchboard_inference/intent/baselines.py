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

    def __init__(self, temperature: float = 1.0):
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

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        scaled = self.log_probs(texts) / self.temperature
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
