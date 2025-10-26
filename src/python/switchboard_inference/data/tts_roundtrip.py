"""Real ASR round trip with no paid services: espeak-ng speech, telephone degradation, then faster-whisper.

The voice is robotic, so absolute error rates don't predict real calls. What this gives is a measured
error profile and proof that the audio path and Whisper fallback really work. Whisper also has no isiZulu,
so utterances are grouped by mix and the isiZulu damage shows up as a number rather than an assumption.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

from switchboard_inference.asr.whisper import ulaw_to_16k_float
from switchboard_inference.data.audio import Degrade, align_counts, degrade, read_wav
from switchboard_inference.data.noise import profile_from_alignments
from switchboard_inference.data.splits import load_split

IMAGE = "switchboard/tools"
VOICES = {"en": "en-gb", "en+zu": "en-gb", "en+af": "en-gb", "en+zu+af": "en-gb"}


def speak_all(rows: list[dict], directory: Path) -> None:
    """Runs one container for all clips: starting one per clip would take minutes."""
    script = directory / "speak.sh"
    lines = ["#!/bin/sh", "set -e"]
    for row in rows:
        (directory / f"{row['id']}.txt").write_text(row["clean_text"], encoding="utf-8")
        lines.append(f"espeak-ng -v {VOICES[row['mix']]} -s 150 -f '/work/{row['id']}.txt' -w '/work/{row['id']}.wav'")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    subprocess.run(["docker", "run", "--rm", "-v", f"{directory}:/work", IMAGE, "sh", "/work/speak.sh"], check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fixture")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--whisper", default="base")
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--packet-loss", type=float, default=0.01)
    parser.add_argument("--out", default="runs/whisper_roundtrip")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    rows = load_split(args.data, args.split)
    random.Random(0).shuffle(rows)
    rows = rows[: args.count]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(args.whisper, device="cpu", compute_type="int8", cpu_threads=4)
    results = []
    with tempfile.TemporaryDirectory(prefix="sb-tts-", dir=Path.cwd() / "runs") as tmp:
        directory = Path(tmp).resolve()
        speak_all(rows, directory)
        for index, row in enumerate(rows):
            samples, hz = read_wav(directory / f"{row['id']}.wav")
            ulaw, t_eos = degrade(samples, hz, Degrade(args.snr, args.packet_loss, seed=index))
            started = time.perf_counter()
            segments, info = model.transcribe(ulaw_to_16k_float(ulaw), vad_filter=True, beam_size=1)
            text = " ".join(s.text.strip() for s in segments)
            elapsed = (time.perf_counter() - started) * 1000
            cleaned = "".join(c for c in text.lower() if c.isalnum() or c.isspace() or c == "'")
            results.append(
                {
                    **{k: row[k] for k in ("id", "tenant", "intent", "mix", "at", "group", "clarify_reply", "provenance")},
                    "clean_text": row["clean_text"], "text": cleaned, "language": info.language,
                    "t_eos_ms": t_eos, "audio_ms": len(ulaw) / 8, "asr_ms": elapsed,
                    **align_counts(row["clean_text"], cleaned),
                }
            )  # fmt: skip
            if index % 20 == 0:
                print(f"{index}/{len(rows)}", flush=True)

    summary: dict = {"whisper": args.whisper, "snr_db": args.snr, "packet_loss": args.packet_loss, "voice": "espeak-ng en-gb", "rows": len(results), "per_mix": {}}
    for mix in sorted({r["mix"] for r in results}):
        part = [r for r in results if r["mix"] == mix]
        summary["per_mix"][mix] = wer_of(part)
    summary["overall"] = wer_of(results)
    ordered = sorted(r["asr_ms"] for r in results)
    summary["asr_ms"] = {"p50": ordered[len(ordered) // 2], "p95": ordered[int(0.95 * len(ordered))]}
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def wer_of(rows: list[dict]) -> dict:
    s = sum(r["substitutions"] for r in rows)
    d = sum(r["deletions"] for r in rows)
    i = sum(r["insertions"] for r in rows)
    n = sum(r["reference_words"] for r in rows)
    profile = profile_from_alignments(s, d, i, n, "measured")
    return {"rows": len(rows), "wer": profile.wer, "substitution": profile.substitution, "deletion": profile.deletion, "insertion": profile.insertion}


if __name__ == "__main__":
    main()
