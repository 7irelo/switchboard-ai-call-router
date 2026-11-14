"""Lists ElevenLabs voices on the account and times first audio over the streaming WebSocket, in 8 kHz mu-law.

Text goes in word by word, as the agent's model writes it. Usage: python bench/elevenlabs_probe.py [model_id]
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import statistics
import sys
import time

import httpx
import websockets

KEY = os.environ["ELEVENLABS_API_KEY"]
PHRASES = False
TEXT = "Oh no, poor thing. A standard consultation is four hundred and fifty rand. Would you like me to book her in for tomorrow morning?"


def voices() -> list[dict]:
    r = httpx.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": KEY}, timeout=20)
    r.raise_for_status()
    return r.json()["voices"]


async def first_audio(voice_id: str, model: str, schedule: list[int], auto: bool) -> tuple[float, float, int]:
    url = (f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model}"
           f"&output_format=ulaw_8000&auto_mode={'true' if auto else 'false'}")
    started = time.perf_counter()
    async with websockets.connect(url, additional_headers={"xi-api-key": KEY}) as ws:
        connected = time.perf_counter()
        init = {"text": " ", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        if not auto:
            init["generation_config"] = {"chunk_length_schedule": schedule}
        await ws.send(json.dumps(init))
        t0 = time.perf_counter()

        async def send() -> None:
            phrase = ""
            for word in TEXT.split(" "):
                phrase += word + " "
                await asyncio.sleep(1 / 60)  # Words arrive at a model's pace either way.
                if not PHRASES or word.endswith((",", ".", "?", "!")):
                    await ws.send(json.dumps({"text": phrase}))
                    phrase = ""
            if phrase:
                await ws.send(json.dumps({"text": phrase}))
            await ws.send(json.dumps({"text": "", "flush": True}))
            await ws.send(json.dumps({"text": ""}))

        sender = asyncio.create_task(send())
        first, total = None, 0
        async for raw in ws:
            message = json.loads(raw)
            if message.get("audio"):
                first = first or time.perf_counter()
                total += len(base64.b64decode(message["audio"]))
            if message.get("isFinal"):
                break
        await sender
        return (connected - started) * 1000, (first - t0) * 1000, total


async def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "eleven_flash_v2_5"
    found = voices()
    print("voices:", [(v["name"], v["voice_id"], (v.get("labels") or {}).get("accent")) for v in found][:30])
    voice = next(v for v in found if v["name"].startswith(("Sarah", "Jessica", "Alice", "Matilda", "Laura")))
    global PHRASES
    for label, schedule, auto, phrases in [("auto_mode, words", [], True, False), ("auto_mode, phrases", [], True, True),
                                           ("schedule 50, words", [50, 80, 120, 150], False, False)]:
        PHRASES = phrases
        rows = [await first_audio(voice["voice_id"], model, schedule, auto) for _ in range(4)]
        print(f"{model} {voice['name']} {label}: connect p50 {statistics.median(r[0] for r in rows):.0f} ms, "
              f"first audio after first word p50 {statistics.median(r[1] for r in rows):.0f} ms "
              f"({', '.join(f'{r[1]:.0f}' for r in rows)}), audio {rows[0][2] / 8000:.1f} s")


if __name__ == "__main__":
    asyncio.run(main())
