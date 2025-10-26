"""Generates reportable utterances from seeded specs, using an open-weight model behind any
OpenAI-compatible endpoint, then checks each one with a second independent model.

Labels are planted by the spec. The verifier only rejects rows whose text no longer says what the spec asked.
Needs GENERATOR_URL, GENERATOR_KEY, GENERATOR_MODEL and VERIFIER_MODEL. Closed APIs are not used here (docs/adr/0001).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx2 as httpx

from switchboard_inference.data.splits import assign, write_dataset
from switchboard_inference.decision import CANONICAL_INTENTS, load_tenants

MIXES = {"en": 0.40, "en+zu": 0.30, "en+af": 0.20, "en+zu+af": 0.10}
STYLES = ["single-word insertion", "mid-sentence switch", "whole-turn switch", "register change when upset"]
HARD_CASES = {
    "indirect": "State the need indirectly, without naming it.",
    "two_intents": "Mention a second, lesser concern as well, after the main one.",
    "calm_emergency": "Describe real, immediate danger in a calm and unhurried voice.",
    "dramatic_non_emergency": "Use dramatic words like disaster or nightmare about something that is not dangerous.",
    "none": "",
}
BUSINESS = {
    "thornveld_vet": "an animal hospital",
    "sandrivier_metro": "a municipal water and electricity contact centre",
    "quillfibre": "a fibre internet provider",
    "marula_ridge": "a private school",
    "blue_aloe": "a short-term insurance broker",
}


@dataclass(frozen=True)
class GenSpec:
    id: str
    group: str
    tenant: str
    intent: str
    mix: str
    style: str
    hard_case: str
    length: str
    at: str


def sample_specs(tenants: dict, count: int, seed: int, times: list[str]) -> list[GenSpec]:
    rng = random.Random(seed)
    mixes, weights = zip(*MIXES.items(), strict=True)
    specs = []
    for n in range(count):
        tenant_id = rng.choice(sorted(tenants))
        pool = [*CANONICAL_INTENTS, *sorted(tenants[tenant_id].custom_intents)]
        intent = rng.choice(pool)
        if intent == "emergency":
            hard = rng.choice(["calm_emergency", "none"])
        elif intent == "complaint":
            hard = rng.choice(["dramatic_non_emergency", "none"])
        else:
            hard = rng.choice(["indirect", "two_intents", "none", "none"])
        spec_id = f"s{seed}-{n}"
        specs.append(
            GenSpec(
                id=spec_id, group=spec_id, tenant=tenant_id, intent=intent, mix=rng.choices(mixes, weights)[0],
                style=rng.choice(STYLES), hard_case=hard, length=rng.choice(["short", "medium", "long"]), at=rng.choice(times),
            )
        )  # fmt: skip
    return specs


def generation_prompt(spec: GenSpec, rubric: str) -> str:
    languages = {"en": "English", "en+zu": "South African English mixed with isiZulu", "en+af": "South African English mixed with Afrikaans", "en+zu+af": "South African English mixed with isiZulu and Afrikaans"}
    return (
        f"Write what one caller says to {BUSINESS.get(spec.tenant, 'a business')} on the phone.\n"
        f"Their need, by this rubric, is `{spec.intent}`:\n{rubric}\n"
        f"Language: {languages[spec.mix]}. Code-switching style: {spec.style}.\n"
        f"Length: {spec.length}. {HARD_CASES[spec.hard_case]}\n"
        "Natural spoken language, with no names of real companies or people. Output only what they say."
    )


def verify_prompt(text: str, intents: list[str], rubric: str) -> str:
    return (
        f"Rubric:\n{rubric}\n\nCaller said:\n{text}\n\nWhich one intent applies? Choose from: {', '.join(intents)}. "
        "Reply with the intent only."
    )


class Client:
    def __init__(self, url: str, key: str, http: httpx.AsyncClient | None = None):
        self.url = url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {key}"}
        self.http = http or httpx.AsyncClient(timeout=120)

    async def complete(self, model: str, prompt: str, max_tokens: int = 200) -> str:
        response = await self.http.post(
            self.url, headers=self.headers,
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        )  # fmt: skip
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


async def generate_rows(specs: list[GenSpec], client: Client, generator: str, verifier: str, rubric: str, tenants: dict, concurrency: int = 8) -> tuple[list[dict], list[dict]]:
    gate = asyncio.Semaphore(concurrency)
    kept: list[dict] = []
    rejected: list[dict] = []

    async def one(spec: GenSpec) -> None:
        async with gate:
            text = await client.complete(generator, generation_prompt(spec, rubric))
            answer = await client.complete(verifier, verify_prompt(text, list(tenants[spec.tenant].intents), rubric), max_tokens=20)
        row = {**asdict(spec), "text": text, "clean_text": text, "clarify_reply": "", "provenance": f"generated:{generator}", "verifier": verifier}
        if answer.strip().strip("`.").lower() == spec.intent:
            kept.append(row)
        else:
            rejected.append({**row, "verifier_said": answer})

    await asyncio.gather(*(one(s) for s in specs))
    return sorted(kept, key=lambda r: r["id"]), sorted(rejected, key=lambda r: r["id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50, help="start with the 50-row pilot for native reviewers")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="data/generated")
    args = parser.parse_args()

    from switchboard_inference.data.specs import TIMES

    tenants = load_tenants()
    rubric = Path("docs/RUBRIC.md").read_text(encoding="utf-8")
    specs = sample_specs(tenants, args.count, args.seed, TIMES)
    generator, verifier = os.environ["GENERATOR_MODEL"], os.environ["VERIFIER_MODEL"]
    client = Client(os.environ["GENERATOR_URL"], os.environ["GENERATOR_KEY"])
    kept, rejected = asyncio.run(generate_rows(specs, client, generator, verifier, rubric, tenants))
    manifest = write_dataset(
        assign(kept), args.out,
        {
            "provenance": f"generated:{generator}", "verifier": verifier, "seed": args.seed,
            "rejected": len(rejected), "prompt_sha256": hashlib.sha256(rubric.encode()).hexdigest(),
            # Reportable only once a human flips this, after the review gates in docs/DATA_CARD.md.
            "reportable": False,
        },
    )  # fmt: skip
    Path(args.out, "rejected.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
    print({k: v["rows"] for k, v in manifest["splits"].items()}, "rejected:", len(rejected))


if __name__ == "__main__":
    main()
