import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from switchboard_inference.decision import CANONICAL_INTENTS, CostMatrix, load_lexicon, load_tenants
from switchboard_inference.eval.ablate import (
    conformal_gate,
    outcome_cost,
    run_expected_cost,
    run_gate,
    threshold_gate,
)
from switchboard_inference.eval.simulate import run_conversation
from switchboard_inference.intent.baselines import KeywordModel
from switchboard_inference.llm.client import LlmResult
from switchboard_inference.llm.recorded import ResponseCache, SpendLedger, cache_key

ROOT = Path(__file__).resolve().parents[2]
TENANTS = load_tenants(ROOT / "configs/tenants")
MATRIX = CostMatrix.load(ROOT / "configs/cost_matrix.yaml")
PHRASES = load_lexicon(ROOT / "configs/emergency_lexicon.yaml")
AT = "2026-11-04T08:00:00+00:00"


def row(intent, text, reply="it is about my bill", tenant="quillfibre"):
    return {"id": "r", "tenant": tenant, "intent": intent, "mix": "en", "at": AT, "text": text, "clarify_reply": reply}


def test_edge_cases_use_real_intents_and_cover_the_hard_pairs():
    cases = [json.loads(x) for x in (ROOT / "tests/golden/edge_cases.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert len(cases) == 50
    assert {c["intent"] for c in cases} == set(CANONICAL_INTENTS)


def test_keyword_baseline_runs_on_every_edge_case_without_crashing():
    cases = [json.loads(x) for x in (ROOT / "tests/golden/edge_cases.jsonl").read_text(encoding="utf-8").splitlines() if x]
    model = KeywordModel()
    for case in cases:
        outcome = run_conversation(row(case["intent"], case["text"], ""), TENANTS["quillfibre"], MATRIX, model.predict, phrases=PHRASES)
        assert outcome.department in TENANTS["quillfibre"].departments


def test_threshold_gate_asks_until_confident():
    gate = threshold_gate(0.9)
    scores = {"a b": {"billing": 0.5, "complaint": 0.5}, "a b it is about my bill": {"billing": 0.95, "complaint": 0.05}}
    action, dept, turns = gate(TENANTS["quillfibre"], MATRIX, row("billing", "a b"), scores.__getitem__, PHRASES, None)
    assert (action, dept, turns) == ("route", "billing_team", 1)


def test_threshold_gate_stops_after_two_turns():
    gate = threshold_gate(0.99)
    _, _, turns = gate(TENANTS["quillfibre"], MATRIX, row("billing", "x"), lambda t: {"billing": 0.5, "complaint": 0.5}, PHRASES, None)
    assert turns == 2


def test_conformal_gate_routes_only_on_a_single_department():
    # billing and payments_arrears share a department, so a set holding both still routes.
    gate = conformal_gate(0.6)
    action, dept, turns = gate(TENANTS["quillfibre"], MATRIX, row("billing", "x"), lambda t: {"billing": 0.5, "payments_arrears": 0.45}, PHRASES, None)
    assert (action, dept, turns) == ("route", "billing_team", 0)


def test_outcome_cost_charges_each_clarifying_turn():
    t = TENANTS["quillfibre"]
    at = datetime(2026, 11, 4, 8, tzinfo=UTC)
    assert outcome_cost(t, MATRIX, row("billing", "x"), "route", "billing_team", at, 2) == (1.0, True, False)
    assert outcome_cost(t, MATRIX, row("emergency", "x"), "route", "noc", at, 1) == (5.0, True, False)


def test_gate_comparison_reports_the_same_shape_for_every_arm():
    rows = [row("billing", "my bill is wrong"), row("emergency", "there is a gas leak")]
    model = KeywordModel()
    a = run_gate(threshold_gate(0.5), rows, TENANTS, MATRIX, model.predict, PHRASES)
    b = run_expected_cost(rows, TENANTS, MATRIX, model.predict, PHRASES)
    assert set(a) == set(b)
    assert b["emergency_recall"] == 1.0


def test_prefix_calibration_gives_short_text_its_own_temperature():
    from switchboard_inference.intent.baselines import TfidfModel

    texts = {"billing": "my bill is far too high this month again", "cancel": "please end my contract at the end of the month"}
    rows = [{"text": t, "intent": i} for i, t in texts.items() for _ in range(40)]
    model = TfidfModel().fit([r["text"] for r in rows], [r["intent"] for r in rows])
    # Short prefixes share opening words, or say nothing, so they must be trusted less than full sentences.
    noisy = [{"text": "my", "intent": "cancel"}] * 20 + [{"text": "my", "intent": "billing"}] * 20 + rows
    fitted = model.fit_prefix_calibration(noisy, min_rows=10)
    assert fitted
    assert model.temperature_for("my") == fitted[0]
    assert model.temperature_for("one two three four five") == fitted.get(1, model.temperature)
    legacy = TfidfModel()
    del legacy.bucket_temperatures
    assert legacy.temperature_for("my") == legacy.temperature


def test_cache_keeps_tenants_apart_and_survives_reload(tmp_path):
    a = cache_key("tenant_a", "m", "hello", ["x"], "")
    b = cache_key("tenant_b", "m", "hello", ["x"], "")
    assert a != b
    assert a != cache_key("tenant_a", "other", "hello", ["x"], "")
    path = tmp_path / "c.jsonl"
    cache = ResponseCache(path)
    assert cache.get(a) is None
    result = LlmResult("answered", intent="billing", model="m", cost_usd=0.1)
    cache.put(a, result)
    assert ResponseCache(path).get(a) == result
    assert ResponseCache(path).get(b) is None


def test_ledger_stops_spend_at_the_cap():
    ledger = SpendLedger(cap_usd=0.2)
    assert ledger.allow()
    ledger.record(LlmResult("answered", model="m", cost_usd=0.15))
    assert ledger.allow()
    ledger.record(LlmResult("answered", model="m", cost_usd=0.1))
    assert not ledger.allow()
    assert ledger.cost_per_1k(10) == pytest.approx(25)
