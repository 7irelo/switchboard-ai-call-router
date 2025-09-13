"""Deepgram streaming over a raw WebSocket. NOT yet run against the real API (needs a key: M0 Spike B).

End of speech (PLAN F3): `speech_final` is primary, a local VAD is the watchdog because `speech_final`
can be delayed or lost in noise, and `UtteranceEnd` is the safety net.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

from switchboard_inference.asr.base import AsrEvent, AsrUnavailable
from switchboard_inference.asr.vad import EnergyVad


@dataclass(frozen=True)
class DeepgramConfig:
    api_key: str
    host: str = "api.deepgram.com"  # Region is benchmark variable L2.
    model: str = "nova-3"
    language: str = "multi"
    endpointing_ms: int = 300  # Ablation L1.
    utterance_end_ms: int = 1000  # Deepgram's documented minimum.
    connect_timeout_s: float = 2.0

    def url(self) -> str:
        query = urlencode(
            {
                "model": self.model,
                "language": self.language,
                "encoding": "mulaw",
                "sample_rate": 8000,
                "channels": 1,
                "interim_results": "true",
                "endpointing": self.endpointing_ms,
                "utterance_end_ms": self.utterance_end_ms,
                "vad_events": "true",
                "smart_format": "false",
            }
        )
        return f"wss://{self.host}/v1/listen?{query}"


def parse_message(raw: str) -> tuple[list[AsrEvent], bool]:
    """Returns events plus whether this message ended the utterance. Kept pure so it can be tested."""
    message = json.loads(raw)
    kind = message.get("type")
    if kind == "UtteranceEnd":
        return [AsrEvent("end_of_speech", source="utterance_end")], True
    if kind != "Results":
        return [], False
    alternatives = message.get("channel", {}).get("alternatives", [])
    if not alternatives:
        return [], False
    best = alternatives[0]
    text = best.get("transcript", "")
    confidence = float(best.get("confidence", 0.0))
    languages = best.get("languages") or []
    language = languages[0] if languages else ""
    events: list[AsrEvent] = []
    is_final = bool(message.get("is_final"))
    if text:
        events.append(AsrEvent("final" if is_final else "partial", text, confidence, language))
    if message.get("speech_final"):
        events.append(AsrEvent("end_of_speech", source="speech_final"))
        return events, True
    return events, False


class DeepgramSession:
    def __init__(self, config: DeepgramConfig, vad: EnergyVad | None = None):
        self.config = config
        self.vad = vad or EnergyVad()
        self.socket = None
        self.queue: asyncio.Queue[AsrEvent | Exception | None] = asyncio.Queue()
        self.reader: asyncio.Task | None = None
        self.ended = False

    async def connect(self) -> None:
        try:
            self.socket = await asyncio.wait_for(
                websockets.connect(
                    self.config.url(), additional_headers={"Authorization": f"Token {self.config.api_key}"}, max_size=2**20
                ),
                timeout=self.config.connect_timeout_s,
            )
        except (TimeoutError, OSError, websockets.WebSocketException) as err:
            raise AsrUnavailable(f"connect failed: {type(err).__name__}") from err
        self.reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        try:
            async for raw in self.socket:
                if isinstance(raw, bytes):
                    continue
                events, ended = parse_message(raw)
                for event in events:
                    await self.queue.put(event)
                if ended:
                    self.ended = True
                    await self.queue.put(None)
                    return
            if not self.ended:
                await self.queue.put(AsrUnavailable("socket closed before end of speech"))
        except websockets.WebSocketException as err:
            await self.queue.put(AsrUnavailable(f"socket error: {type(err).__name__}"))

    async def send_audio(self, ulaw: bytes) -> None:
        if self.ended or self.socket is None:
            return
        try:
            await self.socket.send(ulaw)
        except websockets.WebSocketException as err:
            raise AsrUnavailable(f"send failed: {type(err).__name__}") from err
        if self.vad.feed(ulaw) and not self.ended:
            # The watchdog: silence is real, but no signal has come from Deepgram. Ask it to flush
            # what it holds, then end the utterance whether or not anything further arrives.
            self.ended = True
            await self.socket.send(json.dumps({"type": "Finalize"}))
            await self.queue.put(AsrEvent("end_of_speech", source="vad_watchdog"))
            await self.queue.put(None)

    async def finish(self) -> None:
        if self.socket is not None and not self.ended:
            await self.socket.send(json.dumps({"type": "CloseStream"}))

    async def events(self) -> AsyncIterator[AsrEvent]:
        while (item := await self.queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    async def close(self) -> None:
        if self.reader:
            self.reader.cancel()
        if self.socket is not None:
            await self.socket.close()
