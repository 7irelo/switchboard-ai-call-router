# ADR 0002: The escalation gate compares the best route, not the best action

Status: accepted. Refines PLAN section 8.

## Context

The plan's rule is `a* = argmin_a E[C(i, a)]` over routes, clarify and reception, with an escalation gate when "the best action's expected cost" exceeds `τ_llm`.

Taken literally, that gate can almost never fire. Clarify costs 0.5 on non-emergencies, so the best action's expected cost is capped near 0.5 whenever emergency probability is low. Any `τ_llm` above that means the LLM never runs.

## Decision

The gate reads the expected cost of the best **terminal** action (a route or reception), written `EC*`:

1. `EC* ≤ τ_llm`: take it.
2. Otherwise, if the LLM hasn't contributed and is available: escalate.
3. Otherwise, if `EC* > τ_clarify`, clarify is cheaper than `EC*`, and turns remain: clarify.
4. Otherwise: take the best terminal action. It never loops, and a caller always ends up somewhere.

Clarifying an emergency is allowed once. After the first clarifying turn, its cost on the emergency class rises from 5 to the wrong-live-department cost of 20.

## Consequences

Python's offline implementation and .NET's production implementation both follow this, and shared vectors in `tests/golden/decision_vectors.json` hold them to the same answers.
