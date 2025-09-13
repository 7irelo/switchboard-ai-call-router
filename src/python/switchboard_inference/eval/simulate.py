"""Runs a conversation through the cascade offline: score, decide, optionally clarify, then cost it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from switchboard_inference.decision import (
    EMERGENCY,
    MAX_CLARIFY_TURNS,
    CostMatrix,
    Tenant,
    clarify_cost,
    decide,
    lexicon_matches,
    reception_cost,
    route_cost,
)

Scorer = Callable[[str], dict[str, float]]
# Returns updated scores, or None when the LLM gave no usable answer.
Escalator = Callable[[dict, dict[str, float]], "dict[str, float] | None"]


@dataclass
class Outcome:
    id: str
    tenant: str
    intent: str
    mix: str
    department: str | None
    action: str
    correct: bool
    cost: float
    clarify_turns: int
    escalated: bool
    emergency_to_voicemail: bool
    reached_emergency: bool
    trace: list[str] = field(default_factory=list)


def scoped(scores: dict[str, float], tenant: Tenant) -> dict[str, float]:
    return {k: v for k, v in scores.items() if k in tenant.intents}


def reaches_voicemail(t: Tenant, department: str, at: datetime) -> bool:
    return not t.is_open(department, at) and t.departments[department]["closed_action"] == "voicemail"


def run_conversation(
    row: dict,
    tenant: Tenant,
    matrix: CostMatrix,
    score: Scorer,
    *,
    phrases: list[str],
    escalate: Escalator | None = None,
    allow_clarify: bool = True,
    gate: bool = True,
) -> Outcome:
    """`gate=False` is the "no cascade" baseline: top-1 routes immediately, whatever the expected cost."""
    at = datetime.fromisoformat(row["at"])
    intent = row["intent"]
    text = row["text"]
    scores = scoped(score(text), tenant)
    llm_done = False
    escalated = False
    turns = 0
    cost = 0.0
    trace: list[str] = []

    if not gate:
        # Baseline: route straight to the top intent's department, with no rules and no gate.
        if scores:
            top = max(sorted(scores), key=lambda k: scores[k])
            dept = tenant.department_for(top)
        else:
            dept = tenant.raw["fallback"]["department"]
        return _finish(row, tenant, matrix, at, intent, "route", dept, cost, turns, escalated, ["top1"])

    while True:
        d = decide(
            tenant, matrix, scores, at,
            clarify_turns=turns if allow_clarify else MAX_CLARIFY_TURNS,
            llm_contributed=llm_done,
            llm_available=escalate is not None,
            lexicon_hit=lexicon_matches(text, phrases),
        )  # fmt: skip
        trace.append(f"{d.action}:{d.reason}")
        if d.action == "escalate":
            llm_done = True
            escalated = True
            updated = escalate(row, scores) if escalate else None
            if updated:
                scores = updated
            continue
        if d.action == "clarify":
            cost += clarify_cost(matrix, intent, turns)
            turns += 1
            # The reply joins the earlier words, so context isn't lost.
            text = f"{text} {row['clarify_reply']}"
            scores = scoped(score(text), tenant)
            llm_done = False
            continue
        return _finish(row, tenant, matrix, at, intent, d.action, d.department, cost, turns, escalated, trace)


def _finish(row, tenant, matrix, at, intent, action, dept, cost, turns, escalated, trace) -> Outcome:
    final = reception_cost(matrix, intent) if action == "reception" else route_cost(tenant, matrix, intent, dept, at)
    true_dept = tenant.department_for(intent)
    return Outcome(
        id=row["id"],
        tenant=row["tenant"],
        intent=intent,
        mix=row["mix"],
        department=dept,
        action=action,
        correct=dept == true_dept,
        cost=cost + final,
        clarify_turns=turns,
        escalated=escalated,
        emergency_to_voicemail=intent == EMERGENCY and reaches_voicemail(tenant, dept, at),
        reached_emergency=intent == EMERGENCY and dept == true_dept,
        trace=trace,
    )
