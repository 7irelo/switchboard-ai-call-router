"""The ASR contract: audio frames in, transcript and end-of-speech events out."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AsrEvent:
    kind: str  # partial | final | end_of_speech | degraded
    text: str = ""
    confidence: float = 0.0
    language: str = ""
    source: str = ""  # End-of-speech source, or the reason for degradation.


class AsrSession(Protocol):
    async def send_audio(self, ulaw: bytes) -> None: ...

    async def finish(self) -> None: ...

    def events(self) -> AsyncIterator[AsrEvent]: ...

    async def close(self) -> None: ...


class AsrUnavailable(RuntimeError):
    """Raised by a session that has failed, so failover can take over."""
