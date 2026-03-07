"""Synthetic telephone clips for latency work against real ASR, using the voices built into Windows.

They give true end-of-speech timing and clean references, so endpointing and region can be compared. They say
nothing about accents, noise or isiZulu, so accuracy from these clips must not be quoted.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

from switchboard_inference.data.audio import Degrade, degrade, read_wav
from switchboard_inference.data.splits import load_split

VOICES = ("Microsoft David Desktop", "Microsoft Zira Desktop")


def speak(items: list[tuple[str, str, Path]]) -> None:
    """One PowerShell process for all clips: starting one per clip would take minutes."""
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "items.json"
        manifest.write_text(json.dumps([{"voice": v, "text": t, "path": str(p)} for v, t, p in items]), encoding="utf-8")
        script = (
            "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"foreach ($i in (Get-Content -Raw -Encoding UTF8 '{manifest}' | ConvertFrom-Json)) "
            "{ $s.SelectVoice($i.voice); $s.SetOutputToWaveFile($i.path); $s.Speak($i.text); $s.SetOutputToNull() }"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fixture")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--packet-loss", type=float, default=0.01)
    parser.add_argument("--out", default="runs/sapi_clips")
    args = parser.parse_args()

    # English only: these voices can't pronounce isiZulu or Afrikaans, and pretending otherwise would mislead.
    rows = [r for r in load_split(args.data, args.split) if r["mix"] == "en"]
    random.Random(0).shuffle(rows)
    rows = rows[: args.count]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    speak([(VOICES[i % len(VOICES)], r["clean_text"], out / f"{r['id']}.wav") for i, r in enumerate(rows)])
    for index, row in enumerate(rows):
        wav = out / f"{row['id']}.wav"
        samples, hz = read_wav(wav)
        ulaw, t_eos = degrade(samples, hz, Degrade(args.snr, args.packet_loss, seed=index))
        (out / f"{row['id']}.ulaw").write_bytes(ulaw)
        meta = {"t_eos_ms": t_eos, "text": row["clean_text"], "voice": VOICES[index % len(VOICES)], "provenance": "template_fixture", "reportable": False}
        (out / f"{row['id']}.json").write_text(json.dumps(meta), encoding="utf-8")
        wav.unlink()
    print(f"{len(rows)} clips in {out}")


if __name__ == "__main__":
    main()
