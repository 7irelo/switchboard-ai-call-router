"""Builds the template fixture dataset. Not evidence for any performance claim (docs/adr/0001)."""

from __future__ import annotations

import argparse
import random

from switchboard_inference.data.noise import GUESSED_PROFILE, inject
from switchboard_inference.data.specs import PROVENANCE, generate
from switchboard_inference.data.splits import assign, write_dataset
from switchboard_inference.decision import load_tenants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/fixture")
    parser.add_argument("--per-intent-tenant", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()

    tenants = load_tenants()
    rng = random.Random(args.seed)
    rows = []
    for spec in generate(tenants, args.per_intent_tenant, args.seed):
        row = spec.to_row()
        row["clean_text"] = row["text"]
        row["text"] = inject(row["text"], GUESSED_PROFILE, rng)
        rows.append(row)
    splits = assign(rows)
    manifest = write_dataset(
        splits,
        args.out,
        {"provenance": PROVENANCE, "seed": args.seed, "noise_profile": GUESSED_PROFILE.source, "reportable": False},
    )
    print({k: v["rows"] for k, v in manifest["splits"].items()})


if __name__ == "__main__":
    main()
