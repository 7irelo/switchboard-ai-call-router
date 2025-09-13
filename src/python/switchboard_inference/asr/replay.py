"""Scripted ASR for tests and E2E runs: free, stable, and driven by real incoming audio.

The transcript comes from a script, but timing doesn't. Partials and end of speech depend on frames
actually arriving, so a broken media path still fails the test.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import yaml

from switchboard_inference.asr.base import AsrEvent

FRAME_BYTES = 160
DEFAULT_SPEECH_FRAMES = 25  # 500 ms of audio before the transcript is released.


class ReplayScripts:
    def __init__(self, scripts: dict[str, list[str]]):
        self.scripts = scripts

    @classmethod
    def load(cls, path: str | Path) -> ReplayScripts:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def turns(self, key: str) -> list[str]:
        return self.scripts.get(key, [""])


class ReplaySession:
    def __init__(self, text: str, speech_frames: int = DEFAULT_SPEECH_FRAMES):
        self.words = text.split()
        self.speech_frames = speech_frames
        self.frames = 0
        self.released = 0
        self.done = False
        self.queue: asyncio.Queue[AsrEvent | None] = asyncio.Queue()

    async def send_audio(self, ulaw: bytes) -> None:
        if self.done:
            return
        self.frames += len(ulaw) / FRAME_BYTES
        target = min(len(self.words), int(len(self.words) * self.frames / self.speech_frames))
        if target > self.released:
            self.released = target
            if target < len(self.words):
                await self.queue.put(AsrEvent("partial", " ".join(self.words[:target]), 0.8, "en"))
        if self.frames >= self.speech_frames:
            await self._complete()

    async def _complete(self) -> None:
        if not self.done:
            self.done = True
            await self.queue.put(AsrEvent("final", " ".join(self.words), 0.95, "en"))
            await self.queue.put(AsrEvent("end_of_speech", source="replay"))
            await self.queue.put(None)

    async def finish(self) -> None:
        await self._complete()

    async def events(self) -> AsyncIterator[AsrEvent]:
        while (event := await self.queue.get()) is not None:
            yield event

    async def close(self) -> None:
        self.done = True
