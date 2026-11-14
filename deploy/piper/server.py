"""A local text-to-speech service: Piper on CPU, behind the same WebSocket protocol as Deepgram's /v2/speak.

The agent's speaker works with either, unchanged. Text streams in with Speak messages; each sentence is voiced as
soon as it's complete, and Piper is fast enough on a CPU that it's ready well before the previous one finishes
playing. Audio leaves as 8 kHz mu-law, the phone's own format. Flush ends a turn: Flushed, then SpeechMetadata.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import websockets
from piper import PiperVoice
from scipy.signal import resample_poly

log = logging.getLogger("piper")
MODELS = Path(os.environ.get("PIPER_MODELS", "/models"))
SENTENCE = re.compile(r"(.+?[.!?])(\s+|$)", re.S)
LONG_CLAUSE = 120

voices = {path.stem: PiperVoice.load(str(path)) for path in sorted(MODELS.glob("*.onnx"))}
pool = ThreadPoolExecutor(max_workers=int(os.environ.get("PIPER_WORKERS", "2")))


def ulaw(samples: np.ndarray) -> bytes:
    """Float samples in [-1, 1] to G.711 mu-law."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(pcm), 32635) + 0x84
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def synthesize(text: str, voice: str) -> bytes:
    model = voices[voice]
    parts = [chunk.audio_float_array for chunk in model.synthesize(text)]
    if not parts:
        return b""
    rate = model.config.sample_rate
    step = gcd(8000, rate)
    return ulaw(resample_poly(np.concatenate(parts), 8000 // step, rate // step))


def split_ready(buffer: str) -> tuple[list[str], str]:
    """Complete sentences out of the buffer, and what's left over."""
    ready, rest = [], buffer
    while (m := SENTENCE.match(rest)) and m.group(2):
        ready.append(m.group(1).strip())
        rest = rest[m.end() :]
    if len(rest) > LONG_CLAUSE and "," in rest:
        cut = rest.rfind(",") + 1
        ready.append(rest[:cut].strip())
        rest = rest[cut:]
    return ready, rest


async def speak(socket, request_path: str) -> None:
    voice = parse_qs(urlparse(request_path).query).get("model", [""])[0].removeprefix("piper-")
    if voice not in voices:
        await socket.send(json.dumps({"type": "Error", "description": f"unknown voice {voice}"}))
        return
    loop = asyncio.get_running_loop()
    buffer, chars = "", 0
    queue: asyncio.Queue = asyncio.Queue()

    async def say(sentences: list[str]) -> None:
        for sentence in sentences:
            if sentence:
                await queue.put(loop.run_in_executor(pool, synthesize, sentence, voice))

    async def send_audio() -> None:
        # Sentences voice in parallel but leave in order.
        while (item := await queue.get()) is not None:
            if isinstance(item, str):
                await socket.send(item)
                continue
            audio = await item
            for i in range(0, len(audio), 8000):
                await socket.send(audio[i : i + 8000])

    sender = asyncio.create_task(send_audio())
    try:
        async for raw in socket:
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "Speak":
                text = message.get("text", "")
                buffer += text
                chars += len(text)
                ready, buffer = split_ready(buffer)
                await say(ready)
            elif kind == "Flush":
                await say([buffer.strip()])
                buffer = ""
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
    for name in voices:
        synthesize("Warming up.", name)
    async with websockets.serve(lambda s: speak(s, s.request.path), "0.0.0.0", 8880, max_size=2**20):
        log.info("piper ready voices=%s", ",".join(voices))
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
