"""Pre-renders each tenant's prompts to 8 kHz mu-law. Output is gitignored: no audio in the repo.

Uses espeak-ng as a stand-in voice. Swap in a licensed zu-ZA and af-ZA TTS before any real demo. isiZulu
uses espeak's English voice if no Zulu voice exists, so it will sound wrong until then.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import yaml
from switchboard_inference.data.audio import Degrade, degrade, read_wav
from switchboard_inference.decision import load_tenants

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy/prompts"
IMAGE = "switchboard/tools"
VOICES = {"en": "en-gb", "af": "af", "zu": "en-gb"}


def main() -> None:
    prompts = yaml.safe_load((ROOT / "configs/prompts.yaml").read_text(encoding="utf-8"))
    languages = sorted({lang for t in load_tenants(ROOT / "configs/tenants").values() for lang in t.raw.get("languages", [])})
    with tempfile.TemporaryDirectory(prefix="sb-prompts-", dir=ROOT / "runs") as tmp:
        work = Path(tmp).resolve()
        lines = ["#!/bin/sh", "set -e"]
        for lang in languages:
            for name, text in prompts[lang].items():
                (work / f"{lang}_{name}.txt").write_text(text, encoding="utf-8")
                lines.append(f"espeak-ng -v {VOICES[lang]} -s 150 -f /work/{lang}_{name}.txt -w /work/{lang}_{name}.wav")
        (work / "speak.sh").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        subprocess.run(["docker", "run", "--rm", "-v", f"{work}:/work", IMAGE, "sh", "/work/speak.sh"], check=True, capture_output=True)
        count = 0
        for tenant in load_tenants(ROOT / "configs/tenants").values():
            for lang in tenant.raw.get("languages", []):
                target = OUT / tenant.id / lang
                target.mkdir(parents=True, exist_ok=True)
                for name in prompts[lang]:
                    samples, hz = read_wav(work / f"{lang}_{name}.wav")
                    # Clean prompt audio: high SNR, no packet loss, and no padding that would delay listening.
                    ulaw, _ = degrade(samples, hz, Degrade(snr_db=60, trailing_silence_ms=0, seed=1))
                    ulaw = ulaw[: len(ulaw) - len(ulaw) % 160]
                    (target / f"{name}.ulaw").write_bytes(ulaw)
                    count += 1
    print(f"rendered {count} prompts to {OUT}")


if __name__ == "__main__":
    main()
