"""Turn detection with Deepgram Flux (/v2/listen).

Flux decides a caller has finished from the words and how they're said, not from a silence timer. Its
EagerEndOfTurn arrives earlier still, so the agent can start thinking before the turn is certain:
TurnResumed takes that back, EndOfTurn confirms it.
"""

from __future__ import annotations

import asyncio
import bisect
import dataclasses
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

EARLY_SECONDS = 6


class ListenerUnavailable(Exception):
    pass


@dataclass(frozen=True)
class TurnEvent:
    # start | eager_end | resumed | end
    kind: str
    transcript: str = ""
    # Seconds into the audio stream at which the caller's last word ended, when known.
    last_word_end_s: float | None = None
    # Seconds of audio sent when this event arrived: the listener's clock.
    audio_clock_s: float = 0.0
    # When the caller's last word actually reached this service (perf_counter), for honest latency.
    last_word_wall: float | None = None


@dataclass(frozen=True)
class FluxListenConfig:
    api_key: str
    host: str = "api.deepgram.com"
    model: str = "flux-general-en"
    eot_threshold: float = 0.7
    eager_eot_threshold: float | None = 0.5
    eot_timeout_ms: int = 5000
    connect_timeout_s: float = 4.0

    def url(self) -> str:
        query = {
            "model": self.model,
            "encoding": "mulaw",
            "sample_rate": 8000,
            "eot_threshold": self.eot_threshold,
            "eot_timeout_ms": self.eot_timeout_ms,
        }
        if self.eager_eot_threshold is not None:
            query["eager_eot_threshold"] = self.eager_eot_threshold
        return f"wss://{self.host}/v2/listen?{urlencode(query)}"


_KINDS = {"StartOfTurn": "start", "EagerEndOfTurn": "eager_end", "TurnResumed": "resumed", "EndOfTurn": "end"}


def parse_turn_message(raw: str, audio_clock_s: float) -> TurnEvent | None:
    """Pure, so it can be tested on recorded messages. Returns None for messages the agent doesn't act on."""
    message = json.loads(raw)
    if message.get("type") == "Error":
        raise ListenerUnavailable(f"flux error: {message.get('code') or message.get('description') or 'unknown'}")
    if message.get("type") != "TurnInfo":
        return None
    kind = _KINDS.get(message.get("event", ""))
    if kind is None:
        return None
    words = message.get("words") or []
    last = words[-1].get("end") if words else None
    return TurnEvent(kind, (message.get("transcript") or "").strip(), float(last) if last is not None else None, audio_clock_s)


class FluxListener:
    def __init__(self, config: FluxListenConfig):
        self.config = config
        self.socket = None
        self.sent_bytes = 0
        self.queue: asyncio.Queue[TurnEvent | Exception | None] = asyncio.Queue()
        self.reader: asyncio.Task | None = None
        # Callers often start talking while this connects: hold their audio and send it once connected.
        self.early = bytearray()
        # (bytes received so far, when they arrived): maps a word's position in the audio to wall time.
        # Held audio is sent in a burst, so the sending clock alone would overstate end-of-turn delay.
        self.received = 0
        self.offsets: list[int] = []
        self.arrivals: list[float] = []

    @property
    def audio_clock_s(self) -> float:
        return self.sent_bytes / 8000

    async def connect(self) -> None:
        try:
            self.socket = await asyncio.wait_for(
                websockets.connect(self.config.url(), additional_headers={"Authorization": f"Token {self.config.api_key}"}),
                timeout=self.config.connect_timeout_s,
            )
        except (TimeoutError, OSError, websockets.WebSocketException) as err:
            raise ListenerUnavailable(f"connect failed: {type(err).__name__}") from err
        self.reader = asyncio.create_task(self._read())
        if self.early:
            held, self.early = bytes(self.early), bytearray()
            for start in range(0, len(held), 3200):
                await self._send_now(held[start : start + 3200])

    async def _read(self) -> None:
        try:
            async for raw in self.socket:
                if isinstance(raw, bytes):
                    continue
                event = parse_turn_message(raw, self.audio_clock_s)
                if event is not None and event.last_word_end_s is not None:
                    event = dataclasses.replace(event, last_word_wall=self.arrived_at(event.last_word_end_s))
                if event is not None:
                    await self.queue.put(event)
            await self.queue.put(None)
        except (websockets.WebSocketException, ListenerUnavailable) as err:
            await self.queue.put(err if isinstance(err, ListenerUnavailable) else ListenerUnavailable(type(err).__name__))

    def arrived_at(self, seconds: float) -> float | None:
        """When the audio at this position in the stream reached the service."""
        index = bisect.bisect_left(self.offsets, int(seconds * 8000))
        return self.arrivals[index] if index < len(self.arrivals) else None

    async def send(self, ulaw: bytes) -> None:
        self.received += len(ulaw)
        self.offsets.append(self.received)
        self.arrivals.append(time.perf_counter())
        if self.socket is None:
            if len(self.early) < 8000 * EARLY_SECONDS:
                self.early += ulaw
            return
        await self._send_now(ulaw)

    async def _send_now(self, ulaw: bytes) -> None:
        try:
            await self.socket.send(ulaw)
        except websockets.WebSocketException as err:
            raise ListenerUnavailable(f"send failed: {type(err).__name__}") from err
        self.sent_bytes += len(ulaw)

    async def events(self) -> AsyncIterator[TurnEvent]:
        while (item := await self.queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    async def close(self) -> None:
        if self.reader:
            self.reader.cancel()
        if self.socket is not None:
            try:
                await self.socket.send(json.dumps({"type": "CloseStream"}))
            except websockets.WebSocketException:
                pass
            await self.socket.close()
