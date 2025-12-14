"""Self-hosted Whisper fallback, a DECLARED DEGRADED MODE.

Whisper's language table has `af` but no `zu`, `xh`, `st` or `tn`, so isiZulu callers can't get parity.
Not run in this repo's default tests: it needs the optional `whisper` extra and a model download.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from switchboard_inference.asr.base import AsrEvent
from switchboard_inference.asr.vad import _ULAW_TO_LINEAR, EnergyVad

_DECODE = np.array(_ULAW_TO_LINEAR, dtype=np.float32) / 32768.0
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")


def ulaw_to_16k_float(ulaw: bytes) -> np.ndarray:
    audio = _DECODE[np.frombuffer(ulaw, dtype=np.uint8)]
    if len(audio) == 0:
        return audio
    # Linear interpolation is enough: there's no content above 4 kHz to preserve.
    positions = np.linspace(0, len(audio) - 1, len(audio) * 2)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


class WhisperSession:
    """Buffers to end of speech, then transcribes once. There are no partials in degraded mode."""

    def __init__(self, model, vad: EnergyVad | None = None, max_seconds: float = 15.0):
        self.model = model
        self.vad = vad or EnergyVad()
        self.buffer = bytearray()
        self.max_bytes = int(max_seconds * 8000)
        self.queue: asyncio.Queue[AsrEvent | None] = asyncio.Queue()
        self.done = False

    async def send_audio(self, ulaw: bytes) -> None:
        if self.done:
            return
        self.buffer.extend(ulaw)
        if self.vad.feed(ulaw) or len(self.buffer) >= self.max_bytes:
            await self._transcribe("vad_watchdog")

    async def finish(self) -> None:
        await self._transcribe("stream_end")

    async def _transcribe(self, source: str) -> None:
        if self.done:
            return
        self.done = True
        # No speech heard means an empty result. Whisper can hallucinate on silence, so it isn't asked.
        text = ""
        if self.vad.heard_speech:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(_executor, self._run, bytes(self.buffer))
        await self.queue.put(AsrEvent("final", text, 0.0, ""))
        await self.queue.put(AsrEvent("end_of_speech", source=source))
        await self.queue.put(None)

    def _run(self, ulaw: bytes) -> str:
        # One pass only. Retries at higher temperatures can multiply the time, and a caller is waiting.
        segments, _ = self.model.transcribe(
            ulaw_to_16k_float(ulaw), vad_filter=True, beam_size=1, temperature=0.0,
            condition_on_previous_text=False, without_timestamps=True,
        )  # fmt: skip
        return " ".join(s.text.strip() for s in segments)

    async def events(self) -> AsyncIterator[AsrEvent]:
        while (event := await self.queue.get()) is not None:
            yield event

    async def close(self) -> None:
        self.done = True


def load_whisper(size: str = "base", threads: int | None = None):
    from faster_whisper import WhisperModel  # Optional extra, so it's imported only here.

    # Using every core oversubscribes a shared machine and made it slower here, not faster.
    return WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads or int(os.environ.get("WHISPER_THREADS", "4")))
