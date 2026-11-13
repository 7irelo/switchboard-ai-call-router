"""A local text-to-speech service: Kokoro-82M on CPU, behind the same WebSocket protocol as Deepgram's /v2/speak.

The agent's speaker works with either, unchanged. Text streams in with Speak messages; each sentence is
synthesised as soon as it's complete, so the first audio doesn't wait for the whole answer. Audio leaves as
8 kHz mu-law, the phone's own format. Flush ends a turn: the rest is spoken, then Flushed and SpeechMetadata.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

import numpy as np
import websockets
from kokoro_onnx import Kokoro
from scipy.signal import resample_poly

log = logging.getLogger("kokoro")
MODEL = os.environ.get("KOKORO_MODEL", "/models/kokoro-v1.0.int8.onnx")
VOICES = os.environ.get("KOKORO_VOICES", "/models/voices-v1.0.bin")
DEFAULT_VOICE = "af_heart"
# A sentence ends at . ! ? followed by space; a long clause may also break at a comma.
SENTENCE = re.compile(r"(.+?[.!?])(\s+|$)", re.S)
LONG_CLAUSE = 120

kokoro = Kokoro(MODEL, VOICES)
# One synthesis at a time per CPU core group: more threads only slow each sentence down.
pool = ThreadPoolExecutor(max_workers=int(os.environ.get("KOKORO_WORKERS", "2")))


def ulaw(samples: np.ndarray) -> bytes:
    """Float samples in [-1, 1] to G.711 mu-law."""
    x = np.clip(samples, -1.0, 1.0)
    pcm = (x * 32767).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(pcm), 32635) + 0x84
    exponent = np.floor(np.log2(magnitude)).astype(np.int32) - 7
    exponent = np.clip(exponent, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def synthesize(text: str, voice: str, speed: float) -> bytes:
    samples, rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    return ulaw(resample_poly(samples, 1, rate // 8000))


FIRST_CHUNK_WORDS = int(os.environ.get("KOKORO_FIRST_CHUNK_WORDS", "5"))


def split_ready(buffer: str, first: bool = False) -> tuple[list[str], str]:
    """Complete sentences out of the buffer, and what's left over.

    The first chunk of a turn is cut short, at a comma or after a few words: synthesis time grows with length,
    so a short opening gets the caller hearing something sooner while the rest is prepared.
    """
    ready, rest = [], buffer
    if first:
        words = rest.split(" ")
        comma = rest.find(", ")
        if 0 < comma and len(rest[:comma].split()) <= FIRST_CHUNK_WORDS + 3:
            return [rest[: comma + 1].strip()], rest[comma + 2 :]
        if len(words) > FIRST_CHUNK_WORDS + 1:
            return [" ".join(words[:FIRST_CHUNK_WORDS])], " ".join(words[FIRST_CHUNK_WORDS:])
    while (m := SENTENCE.match(rest)) and m.group(2):
        ready.append(m.group(1).strip())
        rest = rest[m.end() :]
    if len(rest) > LONG_CLAUSE and "," in rest:
        cut = rest.rfind(",") + 1
        ready.append(rest[:cut].strip())
        rest = rest[cut:]
    return ready, rest


async def speak(socket, request_path: str) -> None:
    query = parse_qs(urlparse(request_path).query)
    voice = query.get("model", [DEFAULT_VOICE])[0].removeprefix("kokoro-")
    speed = float(query.get("speed", ["1.0"])[0])
    if voice not in kokoro.get_voices():
        await socket.send(json.dumps({"type": "Error", "description": f"unknown voice {voice}"}))
        return
    loop = asyncio.get_running_loop()
    buffer = ""
    chars = 0
    opened = False  # Whether this turn's first chunk has gone to synthesis.
    queue: asyncio.Queue = asyncio.Queue()

    async def say(sentences: list[str]) -> None:
        for sentence in sentences:
            if sentence:
                await queue.put(loop.run_in_executor(pool, synthesize, sentence, voice, speed))

    async def send_audio() -> None:
        # Sentences synthesise in parallel but leave in order.
        while (item := await queue.get()) is not None:
            if isinstance(item, str):
                await socket.send(item)
                continue
            started = time.perf_counter()
            audio = await item
            for i in range(0, len(audio), 8000):
                await socket.send(audio[i : i + 8000])
            log.debug("sentence %.0f ms", (time.perf_counter() - started) * 1000)

    sender = asyncio.create_task(send_audio())
    try:
        async for raw in socket:
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "Speak":
                text = message.get("text", "")
                buffer += text
                chars += len(text)
                ready, buffer = split_ready(buffer, first=not opened)
                if ready:
                    opened = True
                    more, buffer = split_ready(buffer)
                    ready += more
                await say(ready)
            elif kind == "Flush":
                await say([buffer.strip()])
                buffer = ""
                opened = False
                await queue.put(json.dumps({"type": "Flushed"}))
                await queue.put(json.dumps({"type": "SpeechMetadata", "input_character_count": chars}))
                chars = 0
            elif kind == "Close":
                break
    finally:
        await queue.put(None)
        await asyncio.wait_for(sender, 30)


async def main() -> None:
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)  # The health check's bare TCP probe.
    logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')
    synthesize("Warming up.", DEFAULT_VOICE, 1.0)  # The first run builds the ONNX session's caches.
    async with websockets.serve(lambda s: speak(s, s.request.path), "0.0.0.0", 8880, max_size=2**20):
        log.info("kokoro ready voices=%d model=%s", len(kokoro.get_voices()), os.path.basename(MODEL))
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
