"""Text noise that imitates ASR errors.

GUESSED_PROFILE is arm E2's "guessed" level. Reported results must use a profile
measured from a real ASR round trip (profile_from_alignments), which needs Deepgram audio runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseProfile:
    substitution: float
    deletion: float
    insertion: float
    source: str

    @property
    def wer(self) -> float:
        return self.substitution + self.deletion + self.insertion


GUESSED_PROFILE = NoiseProfile(substitution=0.08, deletion=0.04, insertion=0.02, source="guessed")
CONFUSIONS = {"bill": "bell", "pay": "play", "water": "what a", "router": "rooter", "cancel": "counsel", "book": "look"}
FILLERS = ["uh", "um", "ja", "you know"]


def inject(text: str, profile: NoiseProfile, rng: random.Random) -> str:
    out: list[str] = []
    for word in text.split():
        roll = rng.random()
        if roll < profile.deletion:
            continue
        if roll < profile.deletion + profile.substitution:
            out.append(CONFUSIONS.get(word, word[:-1] if len(word) > 3 else word))
            continue
        out.append(word)
        if rng.random() < profile.insertion:
            out.append(rng.choice(FILLERS))
    return " ".join(out)


def profile_from_alignments(substitutions: int, deletions: int, insertions: int, reference_words: int, source: str) -> NoiseProfile:
    """Builds a profile from measured counts, such as `jiwer` output on a real ASR round trip."""
    if reference_words <= 0:
        raise ValueError("reference_words must be positive")
    return NoiseProfile(substitutions / reference_words, deletions / reference_words, insertions / reference_words, source)
