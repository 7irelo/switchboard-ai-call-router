"""Compares speech engines the way the agent uses them: text streamed in word by word, as a model writes it.

Reports time to first audio after the first word was sent, and after the reply finished arriving, plus how
much faster than real time the whole reply was produced. Usage:
  python bench/tts_compare.py ws://tts:8880 kokoro-bf_emma
  python bench/tts_compare.py wss://api.deepgram.com flux-gemma-en   (needs DEEPGRAM_API_KEY)
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time

import websockets

REPLIES = [
    "Sure, a standard consultation is four hundred and fifty rand. Would you like me to book one for you?",
    "We're open until six this evening. Is there anything else I can help with?",
    "I'm sorry to hear that. Is he putting any weight on the leg at all?",
    "No problem. Can I take your name and the best number to reach you on?",
    "Let me put you through to the vet on call right now.",
]
TOKENS_PER_SECOND = 60  # Roughly how fast a small model streams.


async def one(base: str, voice: str, text: str, headers: dict) -> tuple[float, float, float]:
    async with websockets.connect(f"{base}/v2/speak?model={voice}&encoding=mulaw&sample_rate=8000", additional_headers=headers) as ws:
        words = text.split(" ")
        start = time.perf_counter()
        first_audio = None
        audio = 0

        async def send() -> float:
            for i, word in enumerate(words):
                await ws.send(json.dumps({"type": "Speak", "text": word + (" " if i < len(words) - 1 else "")}))
                await asyncio.sleep(1 / TOKENS_PER_SECOND)
            await ws.send(json.dumps({"type": "Flush"}))
            return time.perf_counter()

        sender = asyncio.create_task(send())
        async for message in ws:
            if isinstance(message, bytes):
                if first_audio is None:
                    first_audio = time.perf_counter()
                audio += len(message)
            elif json.loads(message).get("type") == "SpeechMetadata":
                break
        done = time.perf_counter()
        sent = await sender
        seconds = audio / 8000
        return (first_audio - start) * 1000, max(0.0, first_audio - sent) * 1000, seconds / (done - start)


async def main() -> None:
    base, voice = sys.argv[1], sys.argv[2]
    headers = {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"} if "deepgram" in base else {}
    rows = [await one(base, voice, text, headers) for text in REPLIES for _ in range(2)]
    first = [r[0] for r in rows]
    print(f"{voice}: first audio after first word p50 {statistics.median(first):.0f} ms, max {max(first):.0f} ms; "
          f"speed {statistics.median(r[2] for r in rows):.1f}x real time")


if __name__ == "__main__":
    asyncio.run(main())
