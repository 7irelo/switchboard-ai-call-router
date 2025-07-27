"""Offline twin of Switchboard.Domain's decision engine, for evals and threshold fitting.

.NET is the production decision-maker. Both implementations must agree on
tests/golden/decision_vectors.json. See docs/adr/0002.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

CANONICAL_INTENTS = (
    "billing", "payments_arrears", "sales_new", "sales_upgrade", "cancel", "tech_support", "outage_fault",
    "appointment_book", "appointment_change", "order_status", "complaint", "account_admin", "hours_location",
    "speak_to_human", "emergency", "wrong_number", "out_of_scope",
)  # fmt: skip
EMERGENCY = "emergency"
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
MAX_CLARIFY_TURNS = 2
DEGRADED_THRESHOLD_FACTOR = 0.5
EPS = 1e-12


@dataclass(frozen=True)
class CostMatrix:
    costs: dict[str, float]
    adjacency: tuple[frozenset[str], ...]
    protected: frozenset[str]
    tau_llm: float
    tau_clarify: float
    latency_penalty_per_second: float
    latency_budget_ms: float
    voicemail_guard: float

    @classmethod
    def load(cls, path: str | Path) -> CostMatrix:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            costs={k: float(v) for k, v in raw["costs"].items()},
            adjacency=tuple(frozenset(g) for g in raw["adjacency_groups"]),
            protected=frozenset(raw["protected_costs"]),
            tau_llm=float(raw["thresholds"]["tau_llm"]),
            tau_clarify=float(raw["thresholds"]["tau_clarify"]),
            latency_penalty_per_second=float(raw["latency_penalty_per_second_over_budget"]),
            latency_budget_ms=float(raw["latency_budget_ms"]),
            voicemail_guard=float(raw["emergency_voicemail_guard"]),
        )

    def with_overrides(self, overrides: dict[str, float]) -> CostMatrix:
        """A tenant may change costs, but can never lower a protected emergency penalty."""
        for key, value in overrides.items():
            if key not in self.costs:
                raise ValueError(f"unknown cost {key}")
            if key in self.protected and value < self.costs[key]:
                raise ValueError(f"override may not lower protected cost {key}")
        return CostMatrix(
            {**self.costs, **overrides}, self.adjacency, self.protected, self.tau_llm, self.tau_clarify,
            self.latency_penalty_per_second, self.latency_budget_ms, self.voicemail_guard,
        )  # fmt: skip

    def adjacent(self, a: str, b: str) -> bool:
        return any(a in group and b in group for group in self.adjacency)


@dataclass(frozen=True)
class Tenant:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> Tenant:
        tenant = cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
        tenant.validate()
        return tenant

    @property
    def id(self) -> str:
        return self.raw["tenant"]

    @property
    def departments(self) -> dict[str, dict[str, Any]]:
        return {d["id"]: d for d in self.raw["departments"]}

    @property
    def custom_intents(self) -> dict[str, dict[str, Any]]:
        return {c["id"]: c for c in self.raw.get("custom_intents", [])}

    @property
    def intents(self) -> tuple[str, ...]:
        return (*CANONICAL_INTENTS, *sorted(self.custom_intents))

    def validate(self) -> None:
        depts = self.departments
        for intent in CANONICAL_INTENTS:
            if self.raw["intent_map"].get(intent) not in depts:
                raise ValueError(f"{self.id}: {intent} maps to no department")
        for custom in self.custom_intents.values():
            if custom["department"] not in depts:
                raise ValueError(f"{self.id}: custom intent {custom['id']} maps to no department")
        emergency = depts[self.raw["intent_map"][EMERGENCY]]
        if emergency.get("hours") != "always" or not emergency.get("emergency_capable"):
            raise ValueError(f"{self.id}: the emergency department must be emergency-capable and always open")
        if self.raw["fallback"]["department"] not in depts:
            raise ValueError(f"{self.id}: fallback department unknown")

    def department_for(self, intent: str) -> str:
        if intent in self.custom_intents:
            return self.custom_intents[intent]["department"]
        return self.raw["intent_map"][intent]

    def intents_of(self, department: str) -> list[str]:
        return [i for i in self.intents if self.department_for(i) == department]

    def is_open(self, department: str, at: datetime) -> bool:
        hours = self.departments[department]["hours"]
        if hours == "always":
            return True
        local = at.astimezone(ZoneInfo(self.raw["timezone"]))
        if local.date().isoformat() in self.raw.get("holidays", []):
            return False
        now = local.strftime("%H:%M")
        windows = self.raw["schedules"][hours].get(DAYS[local.weekday()], [])
        return any(start <= now < end for start, end in windows)


@dataclass(frozen=True)
class Decision:
    action: str  # route | clarify | reception | escalate
    department: str | None
    expected_cost: float
    reason: str
    candidates: tuple[str, ...] = ()
    closed: bool = False  # route target closed: the closed_action applies


def route_cost(t: Tenant, m: CostMatrix, intent: str, dept: str, at: datetime) -> float:
    c = m.costs
    is_emergency = intent == EMERGENCY
    true_dept = t.department_for(intent)
    if not t.is_open(dept, at):
        if t.departments[dept]["closed_action"] == "voicemail":
            if is_emergency:
                return c["emergency_to_voicemail"]
            if true_dept == dept:
                return c["correct_route"]
            return c["voicemail_while_open"] if t.is_open(true_dept, at) else c["unrelated_department"]
        # A closed department that hands to reception reaches a person.
        if is_emergency:
            return c["emergency_to_wrong_live_department"]
        return c["correct_route"] if true_dept == dept else c["reception"]
    if true_dept == dept:
        return c["correct_route"]
    if is_emergency:
        return c["emergency_to_wrong_live_department"]
    if dept == t.department_for(EMERGENCY):
        return c["non_emergency_to_emergency_route"]
    if any(m.adjacent(intent, other) for other in t.intents_of(dept)):
        return c["adjacent_department"]
    return c["unrelated_department"]


def reception_cost(m: CostMatrix, intent: str) -> float:
    return m.costs["emergency_to_wrong_live_department"] if intent == EMERGENCY else m.costs["reception"]


def clarify_cost(m: CostMatrix, intent: str, turns: int) -> float:
    if intent != EMERGENCY:
        return m.costs["clarify"]
    return m.costs["emergency_clarify"] if turns == 0 else m.costs["emergency_to_wrong_live_department"]


def normalise(t: Tenant, scores: dict[str, float]) -> dict[str, float]:
    """Keep this tenant's intents only. No usable scores becomes a uniform prior."""
    probs = {i: max(0.0, float(scores.get(i, 0.0))) for i in t.intents}
    total = sum(probs.values())
    return {i: p / total for i, p in probs.items()} if total > EPS else {i: 1 / len(probs) for i in probs}


def decide(
    t: Tenant,
    m: CostMatrix,
    scores: dict[str, float],
    at: datetime,
    *,
    clarify_turns: int = 0,
    llm_contributed: bool = False,
    llm_available: bool = True,
    degraded: bool = False,
    did: str | None = None,
    dtmf: str | None = None,
    lexicon_hit: bool = False,
) -> Decision:
    # Layer 0: deterministic rules.
    if dtmf is not None:
        target = t.raw["fallback"].get("dtmf_menu", {}).get(dtmf)
        if target is not None:
            return Decision("route", target, 0.0, "dtmf", closed=not t.is_open(target, at))
    if did is not None:
        for rule in t.raw.get("rules", []):
            if rule.get("when", {}).get("did") == did:
                target = rule["action"]["route"]
                return Decision("route", target, 0.0, "did_rule", closed=not t.is_open(target, at))

    if lexicon_hit:
        # Someone who says "not breathing" is routed, not questioned, even if inference is down.
        return Decision("route", t.department_for(EMERGENCY), 0.0, "emergency_lexicon")

    probs = normalise(t, scores)
    factor = DEGRADED_THRESHOLD_FACTOR if degraded else 1.0
    tau_llm, tau_clarify = m.tau_llm * factor, m.tau_clarify * factor

    def expected(cost_of) -> float:
        return sum(p * cost_of(i) for i, p in probs.items())

    # Terminal actions, in a fixed order so ties resolve the same way in both languages.
    terminal: list[tuple[float, str, str | None]] = [
        (expected(lambda i, d=d: route_cost(t, m, i, d, at)), "route", d)
        for d in sorted(t.departments, key=_ordinal)
        if not (probs[EMERGENCY] >= m.voicemail_guard and _reaches_voicemail(t, d, at))
    ]
    fallback = t.raw["fallback"]["department"]
    if t.is_open(fallback, at):
        terminal.append((expected(lambda i: reception_cost(m, i)), "reception", fallback))
    best_cost, best_action, best_dept = min(terminal, key=lambda x: x[0])
    closed = best_action == "route" and not t.is_open(best_dept, at)

    if best_cost <= tau_llm + EPS:
        return Decision(best_action, best_dept, best_cost, "confident", closed=closed)
    if not llm_contributed and llm_available:
        return Decision("escalate", None, best_cost, "above_tau_llm")
    if clarify_turns < MAX_CLARIFY_TURNS and best_cost > tau_clarify + EPS:
        clarify = expected(lambda i: clarify_cost(m, i, clarify_turns))
        if clarify < best_cost:
            return Decision("clarify", None, clarify, "above_tau_clarify", candidates=_candidates(t, probs))
    return Decision(best_action, best_dept, best_cost, "best_available", closed=closed)


def _reaches_voicemail(t: Tenant, department: str, at: datetime) -> bool:
    return not t.is_open(department, at) and t.departments[department]["closed_action"] == "voicemail"


def _ordinal(value: str) -> list[int]:
    return [ord(ch) for ch in value]


def _candidates(t: Tenant, probs: dict[str, float]) -> tuple[str, ...]:
    """Top two intents in different departments: asking about one department tells us nothing."""
    ranked = sorted(probs.items(), key=lambda kv: (-kv[1], _ordinal(kv[0])))
    picked: list[str] = []
    seen: set[str] = set()
    for intent, _ in ranked:
        dept = t.department_for(intent)
        if dept not in seen:
            picked.append(intent)
            seen.add(dept)
        if len(picked) == 2:
            break
    return tuple(picked)


def load_tenants(directory: str | Path = "configs/tenants") -> dict[str, Tenant]:
    tenants = [Tenant.load(p) for p in sorted(Path(directory).glob("*.yaml"))]
    return {t.id: t for t in tenants}


def load_lexicon(path: str | Path = "configs/emergency_lexicon.yaml") -> list[str]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [p.lower() for p in raw["phrases"]]


def lexicon_matches(text: str, phrases: list[str]) -> bool:
    lowered = " ".join(text.lower().split())
    return any(p in lowered for p in phrases)
