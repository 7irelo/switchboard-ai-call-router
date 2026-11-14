"""Probes the OpenAI Realtime API the way the agent would use it: 8 kHz mu-law in and out, a tool, a spoken question.

Prints the event types it sees and the time from the end of the caller's speech to the first audio back.
Usage: python bench/realtime_probe.py [model]   (needs OPENAI_API_KEY and DEEPGRAM_API_KEY, to voice the question)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from collections import Counter

import websockets

TURN = json.loads(os.environ.get("TURN") or '{"type": "semantic_vad", "eagerness": "high"}')
QUESTION = "Hi there. How much is a standard consultation, and are you open on Saturday?"


async def caller_audio() -> bytes:
    url = "wss://api.deepgram.com/v2/speak?model=flux-hannah-en&encoding=mulaw&sample_rate=8000"
    async with websockets.connect(url, additional_headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"}) as ws:
        await ws.send(json.dumps({"type": "Speak", "text": QUESTION}))
        await ws.send(json.dumps({"type": "Flush"}))
        audio = bytearray()
        async for m in ws:
            if isinstance(m, bytes):
                audio += m
            elif json.loads(m).get("type") == "SpeechMetadata":
                return bytes(audio)
    return bytes(audio)


async def main(model: str) -> None:
    speech = await caller_audio()
    url = f"wss://api.openai.com/v1/realtime?model={model}"
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, max_size=2**24) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "You are Gemma, receptionist at Thornveld Animal Hospital. A standard consultation is 450 rand. "
                                "Open Saturdays 08:00 to 12:00. Answer in one or two short sentences.",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "transcription": {"model": "gpt-4o-mini-transcribe"},
                        "turn_detection": TURN,
                    },
                    "output": {"format": {"type": "audio/pcmu"}, "voice": "marin"},
                },
                "tools": [{"type": "function", "name": "transfer_to_department", "description": "Transfer the caller.",
                           "parameters": {"type": "object", "properties": {"department": {"type": "string"}}, "required": ["department"]}}],
                "tool_choice": "auto",
            },
        }))
        seen: Counter = Counter()
        speech_end = None
        first_audio = None
        audio = 0

        async def send() -> None:
            nonlocal speech_end
            padded = b"\xff" * 4000 + speech + b"\xff" * 24000  # Half a second before, three seconds of silence after.
            start = time.perf_counter()
            for i in range(0, len(padded), 160):
                await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(padded[i : i + 160]).decode()}))
                if i == 4000 + len(speech) - len(speech) % 160:
                    speech_end = time.perf_counter()
                await asyncio.sleep(max(0, start + (i + 160) / 8000 - time.perf_counter()))

        sender = asyncio.create_task(send())
        try:
            async with asyncio.timeout(30):
                async for raw in ws:
                    event = json.loads(raw)
                    kind = event["type"]
                    seen[kind] += 1
                    if kind == "error":
                        print("ERROR", json.dumps(event)[:400])
                    elif kind == "session.updated":
                        s = event["session"]
                        print("session:", s.get("model"), s["audio"]["input"]["format"], s["audio"]["output"]["format"], s["audio"]["output"].get("voice"))
                    elif kind == "response.output_audio.delta":
                        first_audio = first_audio or time.perf_counter()
                        audio += len(base64.b64decode(event["delta"]))
                    elif kind == "conversation.item.input_audio_transcription.completed":
                        print("heard:", event.get("transcript"))
                    elif kind == "response.output_audio_transcript.done":
                        print("said:", event.get("transcript"))
                    elif kind == "response.done":
                        usage = event["response"].get("usage")
                        print("usage:", json.dumps(usage)[:300])
                        break
        except TimeoutError:
            print("timed out")
        sender.cancel()
        print("events:", dict(seen))
        if first_audio and speech_end:
            print(f"{model}: first audio {1000 * (first_audio - speech_end):.0f} ms after the caller stopped; {audio / 8000:.1f} s of audio")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "gpt-realtime-mini"))
