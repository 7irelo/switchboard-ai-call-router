import asyncio
import json

import httpx2 as httpx
import numpy as np
import pytest
from switchboard_inference.data.audio import (
    Degrade,
    align_counts,
    audio_key,
    degrade,
    resample,
    speech_end_ms,
    ulaw_decode,
    ulaw_encode,
)
from switchboard_inference.data.generate import Client, generate_rows, generation_prompt, sample_specs
from switchboard_inference.decision import load_tenants
from switchboard_inference.train.train import prefix_rows

TENANTS = load_tenants("configs/tenants")
TIMES = ["2026-11-04T08:00:00+00:00"]


def test_ulaw_round_trip_is_close():
    samples = 8000 * np.sin(np.linspace(0, 200, 4000))
    decoded = ulaw_decode(ulaw_encode(samples))
    assert np.abs(decoded - samples).max() < 300  # mu-law's largest quantisation step at this level.
    assert len(ulaw_encode(samples)) == 4000


def test_silence_encodes_to_the_ulaw_zero_byte():
    assert set(ulaw_encode(np.zeros(10))) == {0xFF}


def test_resample_changes_length_only_as_expected():
    assert len(resample(np.ones(16000), 16000)) == 8000
    assert len(resample(np.ones(8000), 8000)) == 8000


def test_true_end_of_speech_is_measured_before_noise():
    speech = 8000 * np.sin(np.linspace(0, 2000, 8000))  # One second at 8 kHz.
    silence = np.zeros(8000)
    assert speech_end_ms(np.concatenate([speech, silence])) == pytest.approx(1000, abs=40)
    _, loud = degrade(np.concatenate([speech, silence]), 8000, Degrade(snr_db=5))
    _, quiet = degrade(np.concatenate([speech, silence]), 8000, Degrade(snr_db=40))
    assert loud == quiet


def test_degradation_is_seeded_and_adds_trailing_silence():
    speech = 8000 * np.sin(np.linspace(0, 2000, 8000))
    a, _ = degrade(speech, 8000, Degrade(seed=3, trailing_silence_ms=500))
    b, _ = degrade(speech, 8000, Degrade(seed=3, trailing_silence_ms=500))
    c, _ = degrade(speech, 8000, Degrade(seed=4, trailing_silence_ms=500))
    assert a == b and a != c
    assert len(a) == 8000 + 4000


def test_packet_loss_blanks_whole_frames():
    speech = 8000 * np.sin(np.linspace(0, 2000, 80000))
    clean, _ = degrade(speech, 8000, Degrade(snr_db=60, packet_loss=0, seed=1))
    lossy, _ = degrade(speech, 8000, Degrade(snr_db=60, packet_loss=0.2, seed=1))
    assert np.abs(ulaw_decode(lossy)).sum() < np.abs(ulaw_decode(clean)).sum() * 0.9


def test_asr_cache_key_depends_on_audio_engine_and_config():
    keys = {audio_key(b"a", "dg", "x"), audio_key(b"b", "dg", "x"), audio_key(b"a", "wh", "x"), audio_key(b"a", "dg", "y")}
    assert len(keys) == 4


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("a b c", "a b c", (0, 0, 0)),
        ("a b c", "a x c", (1, 0, 0)),
        ("a b c", "a c", (0, 1, 0)),
        ("a c", "a b c", (0, 0, 1)),
        ("A B", "a b", (0, 0, 0)),
        ("a b", "", (0, 2, 0)),
    ],
)
def test_alignment_counts(reference, hypothesis, expected):
    c = align_counts(reference, hypothesis)
    assert (c["substitutions"], c["deletions"], c["insertions"]) == expected


def test_specs_are_seeded_with_one_group_per_spec():
    a = sample_specs(TENANTS, 50, 1, TIMES)
    assert a == sample_specs(TENANTS, 50, 1, TIMES)
    assert len({s.group for s in a}) == 50
    assert {s.tenant for s in a if s.intent == "claims_new"} <= {"blue_aloe"}


def test_hard_cases_match_their_intent():
    specs = sample_specs(TENANTS, 500, 2, TIMES)
    assert all(s.intent == "emergency" for s in specs if s.hard_case == "calm_emergency")
    assert all(s.intent == "complaint" for s in specs if s.hard_case == "dramatic_non_emergency")


def test_generation_prompt_carries_the_plant():
    spec = sample_specs(TENANTS, 1, 3, TIMES)[0]
    prompt = generation_prompt(spec, "RUBRIC-TEXT")
    assert spec.intent in prompt and "RUBRIC-TEXT" in prompt


def test_verifier_rejects_rows_that_lost_their_label():
    specs = sample_specs(TENANTS, 12, 4, TIMES)

    def handler(request):
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if prompt.startswith("Write what"):
            return httpx.Response(200, json={"choices": [{"message": {"content": "generated line"}}]})
        # This verifier disagrees with every spec except those it can't contradict.
        return httpx.Response(200, json={"choices": [{"message": {"content": "`billing`."}}]})

    client = Client("http://generator.test/v1", "k", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    kept, rejected = asyncio.run(generate_rows(specs, client, "gen", "ver", "rubric", TENANTS))
    assert len(kept) + len(rejected) == 12
    assert all(r["intent"] == "billing" for r in kept)
    assert all(r["intent"] != "billing" and r["verifier_said"] for r in rejected)
    assert all(r["provenance"] == "generated:gen" for r in kept + rejected)


def test_prefix_rows_keep_the_label_and_shorten_the_text():
    row = {"text": "one two three four", "intent": "billing"}
    rows = prefix_rows([row])
    assert {r["text"] for r in rows} == {"one two", "one two three"}
    assert all(r["intent"] == "billing" for r in rows)
    assert prefix_rows([{"text": "one", "intent": "x"}]) == []
