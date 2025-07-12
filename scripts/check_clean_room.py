"""Fail if any tracked file contains a term from the local, gitignored deny-list."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENYLIST = ROOT / ".clean-room-denylist"


def main() -> int:
    if not DENYLIST.exists():
        print("clean-room: no local .clean-room-denylist; add employer and customer terms to enable this check")
        return 0
    terms = [
        t.strip().lower()
        for t in DENYLIST.read_text(encoding="utf-8").splitlines()
        if t.strip() and not t.startswith("#")
    ]
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    # Term names are never printed, so CI logs can't leak them.
    hits = []
    for name in filter(None, files):
        try:
            text = (ROOT / name).read_text(encoding="utf-8").lower()
        except (UnicodeDecodeError, OSError):
            continue
        if any(t in text for t in terms):
            hits.append(name)
    for name in hits:
        print(f"clean-room: deny-listed term found in {name}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
