"""Energy VAD over G.711 mu-law: the local watchdog for Deepgram's end-of-speech signals."""

from __future__ import annotations

from dataclasses import dataclass

FRAME_MS = 20
_ULAW_TO_LINEAR: list[int] = []
for _value in range(256):
    _u = ~_value & 0xFF
    _t = ((_u & 0x0F) << 3) + 0x84
    _t <<= (_u & 0x70) >> 4
    _ULAW_TO_LINEAR.append((0x84 - _t) if (_u & 0x80) else (_t - 0x84))


def frame_energy(ulaw: bytes) -> float:
    if not ulaw:
        return 0.0
    return sum(abs(_ULAW_TO_LINEAR[b]) for b in ulaw) / len(ulaw)


@dataclass
class EnergyVad:
    """Fires once after speech has started and silence has held for `hangover_ms`."""

    threshold: float = 300.0
    hangover_ms: int = 700
    min_speech_ms: int = 100
    speech_ms: int = 0
    silence_ms: int = 0
    fired: bool = False

    def feed(self, ulaw: bytes) -> bool:
        duration = len(ulaw) / 8  # 8 samples per millisecond.
        if frame_energy(ulaw) >= self.threshold:
            self.speech_ms += duration
            self.silence_ms = 0
            return False
        if self.speech_ms >= self.min_speech_ms:
            self.silence_ms += duration
            if self.silence_ms >= self.hangover_ms and not self.fired:
                self.fired = True
                return True
        return False

    @property
    def heard_speech(self) -> bool:
        return self.speech_ms >= self.min_speech_ms
