# ADR 0001: Licensing and training data

Status: proposed. Needs the owner's decision. This is an engineering reading, not legal advice.

## Context

Anthropic's Commercial Terms restrict using the service "to train competing AI models". A small intent classifier may well not be that, but this is a public repo and the call belongs to the owner.

## Decision

- Closed APIs are evaluation baselines only. Their outputs never enter a training set.
- Reportable training data comes from an open-weight model, recorded in the data card with its licence. The baseline LLM must be from a different family than the generator, so it can't simply recognise a sibling's writing.
- Until the generator runs, the repo uses a seeded template generator. Its rows carry `provenance: template_fixture` and are for pipeline tests and regression gates only. No claim may be made from them.

## To check before release

| Item | Status |
|---|---|
| Generator model licence | Not chosen |
| Encoder licences and revision hashes | Not pinned |
| TTS voice terms | Not chosen |
