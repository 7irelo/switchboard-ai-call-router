"""Streaming speech with ElevenLabs, behind the same interface as FluxSpeaker.

ElevenLabs ends a WebSocket stream when its input ends, so each utterance gets its own socket. Opening one from
far away costs over a second, so a spare is always open and initialised, ready for the next utterance; an
interruption just closes the busy one. Text is sent a phrase at a time, ending in a space as ElevenLabs expects:
model tokens often split words, and single words voiced separately sound broken.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import websockets

from switchboard_inference.agent.speaker import SpeakerUnavailable

log = logging.getLogger("switchboard.agent")
PREFIX = "eleven-"
SPARE_MAX_AGE_S = 150  # The server drops an idle socket after inactivity_timeout (180 s).
PHRASE_ENDS = (". ", ", ", "? ", "! ", "; ", ": ")
LONG_PHRASE = 80


@dataclass(frozen=True)
class ElevenConfig:
    api_key: str
    voice: str  # eleven-<voice id>
    model: str = "eleven_flash_v2_5"
    host: str = "api.elevenlabs.io"
    connect_timeout_s: float = 4.0

    @property
    def voice_id(self) -> str:
        return self.voice.removeprefix(PREFIX)

    def url(self) -> str:
        return (
            f"wss://{self.host}/v1/text-to-speech/{self.voice_id}/stream-input"
            f"?model_id={self.model}&output_format=ulaw_8000&auto_mode=true&inactivity_timeout=180"
        )


async def _open(config: ElevenConfig):
    try:
        socket = await asyncio.wait_for(
            websockets.connect(config.url(), additional_headers={"xi-api-key": config.api_key}),
            timeout=config.connect_timeout_s,
        )
        await socket.send(json.dumps({"text": " ", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": 1.0}}))
    except (TimeoutError, OSError, websockets.WebSocketException) as err:
        raise SpeakerUnavailable(f"connect failed: {type(err).__name__}") from err
    return socket, time.monotonic()


# Plans cap concurrent requests (four on the free plan) and each live call holds two sockets, so cached
# greetings render two at a time.
_render_slots = asyncio.Semaphore(2)


async def synthesize_eleven(config: ElevenConfig, text: str, timeout_s: float = 15.0) -> bytes:
    async with _render_slots:
        return await _synthesize(config, text, timeout_s)


async def _synthesize(config: ElevenConfig, text: str, timeout_s: float) -> bytes:
    socket, _ = await _open(config)
    audio = bytearray()
    try:
        await socket.send(json.dumps({"text": text.strip() + " "}))
        await socket.send(json.dumps({"text": ""}))
        async with asyncio.timeout(timeout_s):
            async for raw in socket:
                message = json.loads(raw)
                if message.get("audio"):
                    audio += base64.b64decode(message["audio"])
                if message.get("isFinal"):
                    break
    except (TimeoutError, websockets.WebSocketException) as err:
        raise SpeakerUnavailable(f"synthesis failed: {type(err).__name__}: {err}") from err
    finally:
        await socket.close()
    return bytes(audio)


class ElevenSpeaker:
    def __init__(self, config: ElevenConfig):
        self.config = config
        self.queue: asyncio.Queue = asyncio.Queue()
        self.spare: asyncio.Task | None = None
        self.active = None
        self.reader: asyncio.Task | None = None
        self.utterance = 0
        self.pending = ""  # Text held until a word boundary.
        self.closing: set[asyncio.Task] = set()

    async def connect(self) -> None:
        self.spare = asyncio.create_task(_open(self.config))
        await asyncio.shield(self.spare)  # Fail fast at call start if ElevenLabs can't be reached.

    async def _take_spare(self):
        try:
            socket, opened = await self.spare
        except SpeakerUnavailable:
            socket, opened = await _open(self.config)
        if time.monotonic() - opened > SPARE_MAX_AGE_S:
            self._close_later(socket)
            socket, opened = await _open(self.config)
        self.spare = asyncio.create_task(_open(self.config))
        return socket

    def _close_later(self, socket) -> None:
        task = asyncio.create_task(socket.close())
        self.closing.add(task)
        task.add_done_callback(self.closing.discard)

    async def _read(self, socket, utterance: int) -> None:
        try:
            async for raw in socket:
                message = json.loads(raw)
                if message.get("audio"):
                    await self.queue.put((utterance, base64.b64decode(message["audio"])))
                if message.get("isFinal"):
                    break
            await self.queue.put((utterance, None))
        except websockets.ConnectionClosedOK:
            await self.queue.put((utterance, None))
        except websockets.WebSocketException as err:
            if utterance == self.utterance:
                await self.queue.put((utterance, SpeakerUnavailable(f"{type(err).__name__}: {err}")))

    async def begin(self) -> int:
        self.utterance += 1
        self.pending = ""
        self.active = await self._take_spare()
        self.reader = asyncio.create_task(self._read(self.active, self.utterance))
        return self.utterance

    async def _send(self, payload: dict) -> None:
        try:
            await self.active.send(json.dumps(payload))
        except websockets.WebSocketException as err:
            raise SpeakerUnavailable(f"send failed: {type(err).__name__}") from err

    async def say(self, text: str) -> None:
        # In auto mode each message is voiced as it arrives, so single words came out with gaps between them
        # (13.4 s of audio for a 7 s sentence). Whole phrases keep the rhythm and still start in about 0.3 s.
        self.pending += text
        cut = max(self.pending.rfind(p) for p in PHRASE_ENDS)
        if cut < 0 and len(self.pending) > LONG_PHRASE:
            cut = self.pending.rfind(" ")
        if cut > 0:
            ready, self.pending = self.pending[: cut + 1], self.pending[cut + 1 :].lstrip()
            await self._send({"text": ready.strip() + " "})

    async def finish(self) -> None:
        if self.pending.strip():
            await self._send({"text": self.pending.strip() + " "})
        self.pending = ""
        await self._send({"text": ""})  # End of input: the rest is generated, then isFinal.

    async def cancel(self) -> None:
        self.utterance += 1
        if self.reader:
            self.reader.cancel()
        if self.active is not None:
            self._close_later(self.active)
            self.active = None

    async def audio(self) -> AsyncIterator[tuple[int, bytes | None]]:
        while True:
            utterance, item = await self.queue.get()
            if isinstance(item, Exception):
                raise item
            yield utterance, item

    async def close(self) -> None:
        if self.reader:
            self.reader.cancel()
        if self.active is not None:
            with contextlib.suppress(Exception):
                await self.active.close()
        if self.spare is not None:
            self.spare.cancel()
            with contextlib.suppress(BaseException):
                socket, _ = await self.spare
                await socket.close()
