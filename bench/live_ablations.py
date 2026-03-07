"""Ablations L1 and L2 against real Deepgram, in real time. L5 (other engines) is not implemented here.

Clips are .ulaw files with a .json sidecar holding {"t_eos_ms": ..., "text": ...}, from the audio pipeline.
For each setting it measures how long after TRUE end of speech the end-of-speech signal arrives, which is the
largest controllable item in the latency budget, and how often it fires before the caller has finished.

It also measures the head start a speculative LLM call could get. The gateway keeps a speculative answer only
if the partial it was based on exactly equals the final transcript, so the head start is measured from the
first such partial. Stability isn't applied, so this is an upper bound.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from switchboard_inference.asr.base import AsrUnavailable
from switchboard_inference.asr.deepgram import DeepgramConfig, DeepgramSession
from switchboard_inference.asr.vad import EnergyVad
from switchboard_inference.data.audio import align_counts

COST_PER_MINUTE_USD = 0.0092  # Nova-3 multi streaming, list price checked 2026-11-03.


async def measure(config: DeepgramConfig, clip: Path, watchdog: bool = True) -> dict:
    meta = json.loads(clip.with_suffix(".json").read_text(encoding="utf-8"))
    audio = clip.read_bytes()
    session = DeepgramSession(config, None if watchdog else EnergyVad(hangover_ms=10**9))
    connecting = time.perf_counter()
    try:
        await session.connect()
    except AsrUnavailable as err:
        # On a live call this is a failover to Whisper, so it's a result to count, not a reason to stop.
        return {"clip": clip.name, "connect_failed": str(err)}
    connect_ms = (time.perf_counter() - connecting) * 1000
    finals: list[str] = []
    partials: list[tuple[float, str]] = []
    started = time.perf_counter()

    async def send() -> None:
        for i in range(0, len(audio), 160):
            await session.send_audio(audio[i : i + 160])
            # Keep real time against the clock, not against sleep's drift.
            await asyncio.sleep(max(0.0, started + (i + 160) / 8000 - time.perf_counter()))
        await session.finish()

    sender = asyncio.create_task(send())
    eos_ms, source = None, None
    try:
        async with asyncio.timeout(len(audio) / 8000 + 10):
            async for event in session.events():
                if event.kind == "partial":
                    partials.append(((time.perf_counter() - started) * 1000, event.text))
                elif event.kind == "final":
                    finals.append(event.text)
                elif event.kind == "end_of_speech":
                    eos_ms, source = (time.perf_counter() - started) * 1000, event.source
                    break
    except TimeoutError:
        source = "timeout"
    sender.cancel()
    await session.close()
    transcript = " ".join(finals)
    counts = align_counts(meta["text"], transcript)
    matching = [at for at, text in partials if text == transcript]
    return {
        "clip": clip.name,
        "connect_ms": connect_ms,
        "t_eos_ms": meta["t_eos_ms"],
        "signal_after_true_eos_ms": None if eos_ms is None else eos_ms - meta["t_eos_ms"],
        "premature": eos_ms is not None and eos_ms < meta["t_eos_ms"],
        "source": source,
        "transcript": transcript,
        "partials": len(partials),
        "speculation_head_start_ms": eos_ms - matching[0] if matching and eos_ms is not None else None,
        **counts,
    }


def summarise(all_rows: list[dict]) -> dict:
    rows = [r for r in all_rows if "connect_failed" not in r]
    connects = sorted(r["connect_ms"] for r in rows)
    delays = sorted(r["signal_after_true_eos_ms"] for r in rows if r["signal_after_true_eos_ms"] is not None)
    words = sum(r["reference_words"] for r in rows)
    errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rows)
    sources: dict[str, int] = {}
    for r in rows:
        sources[str(r["source"])] = sources.get(str(r["source"]), 0) + 1
    head_starts = sorted(r["speculation_head_start_ms"] for r in rows if r["speculation_head_start_ms"] is not None)
    return {
        "n": len(rows),
        "connect_failures": len(all_rows) - len(rows),
        "connect_p50_ms": connects[len(connects) // 2] if connects else None,
        "connect_max_ms": connects[-1] if connects else None,
        "empty_transcripts": sum(not r["transcript"] for r in rows),
        "speculation_usable_rate": len(head_starts) / len(rows) if rows else None,
        "speculation_head_start_p50_ms": head_starts[len(head_starts) // 2] if head_starts else None,
        "speculation_head_start_max_ms": head_starts[-1] if head_starts else None,
        "delay_p50_ms": delays[len(delays) // 2] if delays else None,
        "delay_p95_ms": delays[int(0.95 * len(delays))] if delays else None,
        "premature_rate": sum(r["premature"] for r in rows) / len(rows) if rows else None,
        "wer": errors / words if words else None,
        "sources": sources,
    }


async def run(args) -> dict:
    key = os.environ["DEEPGRAM_API_KEY"]
    clips = sorted(Path(args.clips).glob("*.ulaw"))[: args.limit]
    minutes = sum(c.stat().st_size for c in clips) / 8000 / 60
    arms = [("L1", f"endpointing={e}", DeepgramConfig(api_key=key, endpointing_ms=e), True) for e in (100, 200, 300, 500)]
    arms += [("L1", f"endpointing={e} no_watchdog", DeepgramConfig(api_key=key, endpointing_ms=e), False) for e in (100, 300)]
    arms += [("L2", f"host={h}", DeepgramConfig(api_key=key, host=h), True) for h in ("api.deepgram.com", "api.eu.deepgram.com")]
    cost = minutes * COST_PER_MINUTE_USD * len(arms)
    if cost > args.cap_usd:
        raise SystemExit(f"estimated ${cost:.2f} exceeds the ${args.cap_usd:.2f} cap")
    report: dict = {"estimated_cost_usd": round(cost, 3), "reportable": False, "clips": len(clips), "arms": {}, "rows": {}}
    for group, name, config, watchdog in arms:
        rows = [await measure(config, clip, watchdog) for clip in clips]
        report["arms"][f"{group} {name}"] = summarise(rows)
        report["rows"][f"{group} {name}"] = rows
        print(name, json.dumps(report["arms"][f"{group} {name}"]), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")  # A crash keeps the finished arms.
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--cap-usd", type=float, default=5.0)
    parser.add_argument("--out", default="runs/live_ablations.json")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
