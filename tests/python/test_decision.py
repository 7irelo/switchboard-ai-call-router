import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from switchboard_inference.decision import CANONICAL_INTENTS, CostMatrix, Tenant, decide, load_tenants

ROOT = Path(__file__).resolve().parents[2]
MATRIX = CostMatrix.load(ROOT / "configs/cost_matrix.yaml")
TENANTS = load_tenants(ROOT / "configs/tenants")
OPEN = datetime(2026, 11, 4, 8, 0, tzinfo=UTC)  # Wednesday 10:00 SAST
NIGHT = datetime(2026, 11, 4, 20, 30, tzinfo=UTC)  # Wednesday 22:30 SAST
VET = TENANTS["thornveld_vet"]


def test_five_tenants_load_and_validate():
    assert len(TENANTS) == 5


def test_confident_route_goes_to_mapped_department():
    d = decide(VET, MATRIX, {"billing": 0.98, "complaint": 0.02}, OPEN)
    assert (d.action, d.department) == ("route", "accounts")


def test_uncertain_escalates_before_it_clarifies():
    flat = {i: 1.0 for i in CANONICAL_INTENTS}
    assert decide(TENANTS["quillfibre"], MATRIX, flat, OPEN).action == "escalate"


def test_clarifies_after_llm_when_still_uncertain():
    flat = {i: 1.0 for i in CANONICAL_INTENTS}
    d = decide(TENANTS["quillfibre"], MATRIX, flat, OPEN, llm_contributed=True)
    assert d.action == "clarify"
    assert len(d.candidates) == 2


def test_never_loops_clarifying():
    flat = {i: 1.0 for i in CANONICAL_INTENTS}
    d = decide(TENANTS["quillfibre"], MATRIX, flat, OPEN, llm_contributed=True, clarify_turns=2)
    assert d.action in {"route", "reception"}


def test_emergency_tail_beats_top1_confidence():
    """The motivating case: 90% on a queue, 10% emergency, does not just route."""
    d = decide(VET, MATRIX, {"appointment_book": 0.9, "emergency": 0.1}, OPEN)
    assert d.action == "escalate"


def test_lexicon_hit_routes_to_emergency_when_inference_is_down():
    d = decide(VET, MATRIX, {}, NIGHT, llm_available=False, lexicon_hit=True)
    assert (d.action, d.department, d.reason) == ("route", "on_call_vet", "emergency_lexicon")


def test_ignores_intents_outside_the_tenant():
    d = decide(VET, MATRIX, {"claims_new": 1.0, "billing": 0.01}, OPEN)
    assert d.department == "accounts"


def test_custom_intent_routes_only_for_its_tenant():
    d = decide(TENANTS["blue_aloe"], MATRIX, {"claims_new": 1.0}, OPEN)
    assert d.department == "claims"


def test_degraded_mode_is_more_cautious():
    scores = {"billing": 0.72, "complaint": 0.28}
    tenant = TENANTS["quillfibre"]
    normal = decide(tenant, MATRIX, scores, OPEN, llm_available=False)
    degraded = decide(tenant, MATRIX, scores, OPEN, llm_available=False, degraded=True)
    assert normal.action == "route"
    assert degraded.action == "clarify"


def test_override_cannot_lower_emergency_penalties():
    for key in MATRIX.protected:
        with pytest.raises(ValueError):
            MATRIX.with_overrides({key: MATRIX.costs[key] - 1})
    assert MATRIX.with_overrides({"emergency_to_voicemail": 200}).costs["emergency_to_voicemail"] == 200


def test_validation_rejects_emergency_department_with_hours():
    raw = json.loads(json.dumps(VET.raw))
    raw["departments"][2]["hours"] = "extended"
    with pytest.raises(ValueError):
        Tenant(raw).validate()


score_maps = st.dictionaries(st.sampled_from(CANONICAL_INTENTS), st.floats(0, 1), max_size=6)
moments = st.datetimes(datetime(2026, 1, 1), datetime(2027, 1, 1), timezones=st.just(UTC))


@settings(max_examples=150, deadline=None)
@given(tenant=st.sampled_from(sorted(TENANTS)), scores=score_maps, at=moments, turns=st.integers(0, 2), share=st.floats(0.02, 1))
def test_emergency_never_goes_to_voicemail(tenant, scores, at, turns, share):
    """Hard invariant: with real emergency probability, no decision sends the caller to a closed voicemail."""
    t = TENANTS[tenant]
    rest = {k: v for k, v in scores.items() if k != "emergency"}
    total = sum(rest.values()) or 1
    mix = {k: (1 - share) * v / total for k, v in rest.items()} | {"emergency": share}
    d = decide(t, MATRIX, mix, at, clarify_turns=turns, llm_contributed=True)
    if d.action == "route" and d.closed:
        assert t.departments[d.department]["closed_action"] != "voicemail"


@settings(max_examples=150, deadline=None)
@given(tenant=st.sampled_from(sorted(TENANTS)), scores=score_maps, at=moments, turns=st.integers(0, 2))
def test_decision_stays_in_tenant_scope_and_terminates(tenant, scores, at, turns):
    t = TENANTS[tenant]
    d = decide(t, MATRIX, scores, at, clarify_turns=turns, llm_contributed=True)
    assert d.action != "escalate"
    if d.department is not None:
        assert d.department in t.departments
    if turns >= 2:
        assert d.action != "clarify"


def test_golden_vectors_match():
    vectors = json.loads((ROOT / "tests/golden/decision_vectors.json").read_text(encoding="utf-8"))
    assert len(vectors) >= 500
    for v in vectors:
        at = datetime.fromisoformat(v["at"].replace("Z", "+00:00"))
        got = decide(
            TENANTS[v["tenant"]], MATRIX, v["scores"], at,
            clarify_turns=v["clarify_turns"], llm_contributed=v["llm_contributed"],
            llm_available=v["llm_available"], degraded=v["degraded"], dtmf=v["dtmf"],
            lexicon_hit=v["lexicon_hit"],
        )  # fmt: skip
        e = v["expect"]
        assert (got.action, got.department, got.reason, list(got.candidates), got.closed) == (
            e["action"], e["department"], e["reason"], e["candidates"], e["closed"],
        )  # fmt: skip
        assert got.expected_cost == pytest.approx(e["expected_cost"], abs=1e-8)
