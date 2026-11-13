"""Streaming speech with Deepgram Flux TTS (/v2/speak), straight into the phone's own format: mu-law, 8 kHz.

Text streams in as the model writes it and audio streams out while it's still writing. Connecting costs over a
second from far away, so a spare connection is always open: when a caller interrupts, the agent drops the busy
socket (and any audio still in flight on it) and switches to the spare at once.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlencode

import websockets

log = logging.getLogger("switchboard.agent")


class SpeakerUnavailable(Exception):
    pass


@dataclass(frozen=True)
class FluxSpeakConfig:
    api_key: str
    voice: str = "flux-gemma-en"
    host: str = "api.deepgram.com"
    connect_timeout_s: float = 4.0
    # A local service with the same protocol, such as Kokoro (ws://tts:8880). It gets no Deepgram key.
    base_url: str | None = None

    def url(self) -> str:
        base = self.base_url or f"wss://{self.host}"
        return f"{base}/v2/speak?{urlencode({'model': self.voice, 'encoding': 'mulaw', 'sample_rate': 8000})}"

    def headers(self) -> dict[str, str]:
        return {} if self.base_url else {"Authorization": f"Token {self.api_key}"}


async def _open(config: FluxSpeakConfig):
    try:
        return await asyncio.wait_for(
            websockets.connect(config.url(), additional_headers=config.headers()),
            timeout=config.connect_timeout_s,
        )
    except (TimeoutError, OSError, websockets.WebSocketException) as err:
        raise SpeakerUnavailable(f"connect failed: {type(err).__name__}") from err


async def synthesize(config: FluxSpeakConfig, text: str, timeout_s: float = 15.0) -> bytes:
    """One whole utterance, for audio that's worth caching, like a tenant's greeting."""
    socket = await _open(config)
    audio = bytearray()
    try:
        await socket.send(json.dumps({"type": "Speak", "text": text}))
        await socket.send(json.dumps({"type": "Flush"}))
        async with asyncio.timeout(timeout_s):
            async for message in socket:
                if isinstance(message, bytes):
                    audio += message
                elif json.loads(message).get("type") == "SpeechMetadata":
                    break
    except (TimeoutError, websockets.WebSocketException) as err:
        raise SpeakerUnavailable(f"synthesis failed: {type(err).__name__}") from err
    finally:
        await socket.close()
    return bytes(audio)


class _Line:
    """One open socket. Audio it produces is tagged with the utterance it was speaking."""

    def __init__(self, socket, queue: asyncio.Queue):
        self.socket = socket
        self.utterance = 0
        self.reader = asyncio.create_task(self._read(queue))

    async def _read(self, queue: asyncio.Queue) -> None:
        try:
            async for message in self.socket:
                if isinstance(message, bytes):
                    await queue.put((self.utterance, message))
                    continue
                kind = json.loads(message).get("type")
                if kind == "SpeechMetadata":
                    await queue.put((self.utterance, None))
                elif kind in ("Error", "Warning"):
                    log.warning("tts %s: %s", kind.lower(), message[:200])
        except websockets.WebSocketException as err:
            await queue.put((self.utterance, SpeakerUnavailable(type(err).__name__)))

    async def close(self) -> None:
        self.reader.cancel()
        try:
            await self.socket.close()
        except websockets.WebSocketException:
            pass


class FluxSpeaker:
    def __init__(self, config: FluxSpeakConfig):
        self.config = config
        self.queue: asyncio.Queue = asyncio.Queue()
        self.active: _Line | None = None
        self.spare: asyncio.Task | None = None
        self.utterance = 0
        self.closing: asyncio.Task | None = None

    async def _line(self) -> _Line:
        return _Line(await _open(self.config), self.queue)

    async def connect(self) -> None:
        self.active = await self._line()
        self.spare = asyncio.create_task(self._line())

    async def begin(self) -> int:
        self.utterance += 1
        self.active.utterance = self.utterance
        return self.utterance

    async def say(self, text: str) -> None:
        try:
            await self.active.socket.send(json.dumps({"type": "Speak", "text": text}))
        except websockets.WebSocketException as err:
            raise SpeakerUnavailable(f"send failed: {type(err).__name__}") from err

    async def finish(self) -> None:
        try:
            await self.active.socket.send(json.dumps({"type": "Flush"}))
        except websockets.WebSocketException as err:
            raise SpeakerUnavailable(f"flush failed: {type(err).__name__}") from err

    async def cancel(self) -> None:
        """Abandons the current utterance. Anything it still sends is ignored, because its socket is closed."""
        old = self.active
        self.utterance += 1
        try:
            self.active = await self.spare
        except SpeakerUnavailable:
            self.active = await self._line()
        self.spare = asyncio.create_task(self._line())
        self.closing = asyncio.create_task(old.close())

    async def audio(self) -> AsyncIterator[tuple[int, bytes | None]]:
        """(utterance, bytes) while it speaks, then (utterance, None) when that utterance is complete."""
        while True:
            utterance, item = await self.queue.get()
            if isinstance(item, Exception):
                raise item
            yield utterance, item

    async def close(self) -> None:
        if self.spare:
            self.spare.cancel()
            try:
                line = await self.spare
                await line.close()
            except (asyncio.CancelledError, SpeakerUnavailable):
                pass
        if self.active:
            await self.active.close()
