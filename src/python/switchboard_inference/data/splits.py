"""Group-aware splits and the hash lock on the sealed test set."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

SPLITS = {"train": 0.6, "calibration": 0.13, "dev": 0.13, "test": 0.14}
SEAL_LOG = "SEAL_OPENED.json"


class SealedTestError(RuntimeError):
    pass


def split_of(group: str, salt: str = "switchboard-v1") -> str:
    """Deterministic by group, so paraphrases of one template never cross splits."""
    digest = hashlib.sha256(f"{salt}:{group}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64
    edge = 0.0
    for name, share in SPLITS.items():
        edge += share
        if point < edge:
            return name
    return "train"


def assign(rows: Iterable[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {name: [] for name in SPLITS}
    for row in rows:
        out[split_of(row["group"])].append(row)
    return out


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_dataset(rows_by_split: dict[str, list[dict]], out: str | Path, extra: dict | None = None) -> dict:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"splits": {}, **(extra or {})}
    for name, rows in rows_by_split.items():
        path = root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        manifest["splits"][name] = {"rows": len(rows), "sha256": file_hash(path)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def load_split(root: str | Path, split: str, *, unseal: bool = False, reason: str = "") -> list[dict]:
    """The test split opens once. Opening it needs a stated reason and leaves a record."""
    base = Path(root)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    path = base / f"{split}.jsonl"
    if file_hash(path) != manifest["splits"][split]["sha256"]:
        raise SealedTestError(f"{split} no longer matches its manifest hash")
    if split == "test":
        log = base / SEAL_LOG
        if not unseal:
            raise SealedTestError("the test split is sealed; it opens once, at the final evaluation")
        if log.exists():
            raise SealedTestError(f"the test split was already opened: see {log}. Further runs are post-hoc.")
        if not reason:
            raise SealedTestError("opening the test split needs a reason")
        log.write_text(json.dumps({"reason": reason, "sha256": manifest["splits"][split]["sha256"]}), encoding="utf-8")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
