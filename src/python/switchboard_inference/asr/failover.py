"""Primary ASR with a ring buffer. On failure, replays held audio into the fallback and says so."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable

from switchboard_inference.asr.base import AsrEvent, AsrSession, AsrUnavailable

RING_FRAMES = 500  # 10 seconds of 20 ms frames.
SessionFactory = Callable[[], Awaitable[AsrSession]]


class FailoverSession:
    def __init__(
        self,
        primary: SessionFactory,
        fallback: SessionFactory | None,
        no_result_deadline_s: float = 3.0,
        ring_frames: int = RING_FRAMES,
    ):
        self.primary_factory = primary
        self.fallback_factory = fallback
        self.no_result_deadline_s = no_result_deadline_s
        self.ring: deque[bytes] = deque(maxlen=ring_frames)
        self.dropped_frames = 0
        self.active: AsrSession | None = None
        self.queue: asyncio.Queue[AsrEvent | None] = asyncio.Queue()
        self.failed_over = False
        self.finished = False
        self.lock = asyncio.Lock()
        self.pump: asyncio.Task | None = None

    async def start(self) -> None:
        try:
            self.active = await self.primary_factory()
        except AsrUnavailable as err:
            await self._failover(str(err))
            return
        self.pump = asyncio.create_task(self._pump(self.active))

    async def _pump(self, session: AsrSession) -> None:
        got_any = asyncio.Event()

        async def watchdog() -> None:
            await asyncio.sleep(self.no_result_deadline_s)
            if not got_any.is_set() and session is self.active and not self.failed_over:
                await self._failover("no result within deadline")

        timer = asyncio.create_task(watchdog())
        try:
            async for event in session.events():
                got_any.set()
                await self.queue.put(event)
                if event.kind == "end_of_speech":
                    break
            if session is self.active:
                await self.queue.put(None)
        except AsrUnavailable as err:
            if session is self.active and not self.failed_over:
                await self._failover(str(err))
        finally:
            timer.cancel()

    async def _failover(self, reason: str) -> None:
        async with self.lock:
            if self.failed_over:
                return
            self.failed_over = True
            await self.queue.put(AsrEvent("degraded", source=reason))
            if self.fallback_factory is None:
                # The gateway's next step: the DTMF menu.
                self.active = None
                await self.queue.put(None)
                return
            try:
                self.active = await self.fallback_factory()
            except AsrUnavailable as err:
                self.active = None
                await self.queue.put(AsrEvent("degraded", source=f"fallback failed: {err}"))
                await self.queue.put(None)
                return
            for frame in list(self.ring):
                await self.active.send_audio(frame)
            self.pump = asyncio.create_task(self._pump(self.active))
            if self.finished:
                await self.active.finish()

    async def send_audio(self, ulaw: bytes) -> None:
        if len(self.ring) == self.ring.maxlen:
            self.dropped_frames += 1  # Counted, and audio older than the ring can't be recovered.
        self.ring.append(ulaw)
        async with self.lock:
            active = self.active
        if active is None:
            return
        try:
            await active.send_audio(ulaw)
        except AsrUnavailable as err:
            await self._failover(str(err))

    async def finish(self) -> None:
        self.finished = True
        if self.active is not None:
            try:
                await self.active.finish()
            except AsrUnavailable as err:
                await self._failover(str(err))

    async def events(self) -> AsyncIterator[AsrEvent]:
        while (event := await self.queue.get()) is not None:
            yield event

    async def close(self) -> None:
        if self.pump:
            self.pump.cancel()
        if self.active is not None:
            await self.active.close()
