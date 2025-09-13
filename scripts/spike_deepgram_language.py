"""M0 Spike B: does Nova-3 `multi` handle isiZulu and code-switching on 8 kHz telephone audio?

Needs DEEPGRAM_API_KEY and a folder of .ulaw clips (8 kHz G.711), each with a same-named .txt reference.
Read every transcript yourself: the WER is a summary, and the decision about isiZulu scope is yours.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from switchboard_inference.asr.deepgram import DeepgramConfig, DeepgramSession


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / max(len(ref), 1)


async def transcribe(config: DeepgramConfig, clip: Path) -> dict:
    session = DeepgramSession(config)
    await session.connect()
    audio = clip.read_bytes()
    final: list[str] = []
    first_final_ms = None
    source = ""
    started = time.perf_counter()

    async def send() -> None:
        for i in range(0, len(audio), 160):
            await session.send_audio(audio[i : i + 160])
            await asyncio.sleep(0.02)  # Real time, as a phone call would be.
        await session.send_audio(bytes([0xFF]) * 16000)  # Two seconds of silence for endpointing.
        await session.finish()

    sender = asyncio.create_task(send())
    async for event in session.events():
        if event.kind == "final":
            final.append(event.text)
            first_final_ms = first_final_ms or (time.perf_counter() - started) * 1000
        elif event.kind == "end_of_speech":
            source = event.source
            break
    sender.cancel()
    await session.close()
    reference = clip.with_suffix(".txt").read_text(encoding="utf-8").strip()
    hypothesis = " ".join(final)
    return {"clip": clip.name, "reference": reference, "hypothesis": hypothesis, "wer": word_error_rate(reference, hypothesis), "eos_source": source}


async def run(args) -> None:
    results = []
    for host in args.hosts.split(","):
        config = DeepgramConfig(api_key=os.environ["DEEPGRAM_API_KEY"], host=host, endpointing_ms=args.endpointing)
        for clip in sorted(Path(args.clips).glob("*.ulaw")):
            result = {"host": host, **await transcribe(config, clip)}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            results.append(result)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/spike_deepgram_language.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True)
    parser.add_argument("--hosts", default="api.deepgram.com,api.eu.deepgram.com")
    parser.add_argument("--endpointing", type=int, default=300)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
