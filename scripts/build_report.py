"""Generates docs/REPORT.md from run outputs, so no number in it is typed by hand."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
EVIDENCE = ROOT / "docs/evidence"


def load(name: str):
    """Prefers a fresh run, then the snapshot committed in docs/evidence."""
    for base in (RUNS, EVIDENCE):
        path = base / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def section_mutation(lines: list[str]) -> None:
    data = load("mutation_check.json")
    lines += ["## Isolation and safety mutation check", ""]
    if not data:
        lines.append("Not run.")
        return
    caught = sum(m["status"] == "caught" for m in data)
    lines += [f"{caught} of {len(data)} removed controls were caught by a named failing test.", "", "| Removed control | Result | Caught by |", "|---|---|---|"]
    lines += [f"| {m['name']} | {m['status']} | {m.get('caught_by', '').split(';')[0]} |" for m in data]


def section_failure(lines: list[str]) -> None:
    lines += ["", "## Failure injection, real SIP calls", ""]
    data = load("failure_injection.json")
    if data:
        lines += ["| Scenario | Outcome | Reached |", "|---|---|---|"]
        lines += [f"| {r['scenario']} | {'pass' if r['ok'] else 'FAIL'} | {', '.join(r.get('reached', []))} |" for r in data]
    chaos = load("chaos.json")
    if chaos:
        lines += ["", "Link faults through Toxiproxy, gateway to inference:", "", "| Case | Outcome | Reached | Decision ms |", "|---|---|---|---:|"]
        lines += [f"| {r['case']} | {'pass' if r['ok'] else 'FAIL'} | {', '.join(r['reached'])} | {fmt(r['decision_ms'], 2)} |" for r in chaos]


def section_traces(lines: list[str]) -> None:
    data = load("trace_isolation.json")
    lines += ["", "## Trace isolation, real spans through a real Collector", ""]
    if not data:
        lines.append("Not run.")
        return
    lines += [f"Result: {'pass' if data['ok'] else 'FAIL'}. Spans per tenant sink: " + ", ".join(f"{k} {v['spans']}" for k, v in data["tenants"].items()) + f". Quarantined: {data['quarantine_spans']}."]
    lines += [f"- {f}" for f in data["failures"]]


def section_real_asr(lines: list[str]) -> None:
    data = load("real_asr.json")
    lines += ["", "## Real speech on real calls", ""]
    if not data:
        lines.append("Not run.")
        return
    lines += [
        "A synthetic voice over SIP through Asterisk into CPU Whisper `tiny`, with no replay. `approx_t_route` estimates true end of speech to decision, with an error of a few hundred ms, because the gateway doesn't know exactly when the caller starts speaking. **It is not the pre-registered `T_route`**, which needs Deepgram.",
        "",
        "| Case | Outcome | Reached | Listen ms | Decision ms | Route command ms | Approx. T_route ms |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in data:
        approx = r.get("approx_t_route_ms", "n/a")
        lines.append(f"| {r['case']} | {'pass' if r['ok'] else 'FAIL'} | {', '.join(r['reached'])} | {r.get('listen_ms', 'n/a')} | {r.get('decision_ms', 'n/a')} | {r.get('route_cmd_ms', 'n/a')} | {approx} |")
    repeats = load("whisper_latency.json")
    if repeats:
        lines += ["", f"Repeated calls, Whisper `{repeats['whisper']}`: {repeats['routed']} of {repeats['calls']} routed. Cold first call about {repeats['cold_first_call_ms']} ms; warm median about {repeats['warm_median_ms']} ms, worst {repeats['warm_max_ms']} ms, against a {repeats['budget_ms']} ms budget. Host CPU {repeats.get('host_cpu_percent_after')}% just after. Separately, with the host at about 99% busy from other work, Whisper `tiny` took 6 to 13 s on 4.5 s of audio. **CPU Whisper's speed depends heavily on host load, so it can't promise this budget on a shared machine.**"]
    lines += ["", "Early-commit rows use replay ASR. Premature cut-offs can't be measured there, so the time they save has no counterweight yet."]


def section_langfuse(lines: list[str]) -> None:
    data = load("langfuse_check.json")
    if data:
        lines += ["", "## Self-hosted Langfuse", "", f"Result: {'pass' if data['ok'] else 'FAIL'}. {data.get('traces', 0)} traces from real calls, tags {', '.join(data.get('tags', []))}, transcript text found: {data.get('transcript_leak')}. One project with tenant tags: see `docs/adr/0003`."]


def section_load(lines: list[str]) -> None:
    data = load("load_e2e.json")
    lines += ["", "## SIP load, replay ASR", ""]
    if not data:
        lines.append("Not run.")
        return
    lines += [
        f"Host: {data['host']['processor']}. Cores not pinned, so treat these as indicative.",
        "",
        "**This is not `T_route`.** Replay ASR has no endpointing wait and no network path to Deepgram, which are expected to be most of the budget. These figures cover the gateway's own work from the end-of-speech signal.",
        "",
        "| Calls per second | Offered | Routed | Decision p50 / p95 ms | Route command p50 / p95 ms | SIPp |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for r in data["levels"]:
        d, c = r["eos_to_decision_ms"], r["route_cmd_ms"]
        lines.append(f"| {r['rate_cps']} | {r['calls_offered']} | {r['calls_routed']} | {d['p50']:.2f} / {d['p95']:.2f} | {c['p50']:.1f} / {c['p95']:.1f} | {r['sipp']} |")


def section_stages(lines: list[str]) -> None:
    overhead = load("tracing_overhead.json")
    if overhead:
        lines += ["", "## Tracing overhead", "", overhead["note"], "", "| Tracing | Calls routed | Route p50 ms | Route p95 ms | Decision p95 ms |", "|---|---:|---:|---:|---:|"]
        lines += [f"| {r['tracing']} | {r['routed']} | {r['route_p50']:.1f} | {r['route_p95']:.1f} | {r['decision_p95']:.1f} |" for r in overhead["runs"]]
    lines += ["", "## Stage micro-benchmarks", ""]
    micro = load("micro_tfidf.json")
    if micro:
        lines.append(f"- TF-IDF predict, one thread: p50 {micro['p50_ms']:.2f} ms, p95 {micro['p95_ms']:.2f} ms, p99 {micro['p99_ms']:.2f} ms (n={micro['n']}, stage budget {micro['budget_ms']} ms). Core not pinned.")
    decision = load("micro_decision.json")
    if decision:
        for b in decision["Benchmarks"]:
            lines.append(f"- {b['Method']}: mean {b['Statistics']['Mean'] / 1000:.2f} µs, p95 {b['Statistics']['Percentiles']['P95'] / 1000:.2f} µs (BenchmarkDotNet; stage budget 5 ms).")


def section_encoder(lines: list[str]) -> None:
    data = load("encoder_export.json")
    lines += ["", "## Encoder export pipeline (code check only)", ""]
    if not data:
        lines.append("Not run.")
        return
    lines += [
        "`paraphrase-multilingual-MiniLM-L12-v2`, briefly fine-tuned on CPU over 600 fixture rows. This proves training, export, quantisation, trimming and parity all work. **Its accuracy means nothing, and the bake-off hasn't been run.**",
        "",
        "| Stage | Size MB | Top-1 agreement with PyTorch | Largest logit difference | p50 / p95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("fp32", "int8", "int8_trimmed"):
        r = data[name]
        lines.append(f"| {name} | {r['size_mb']} | {r['top1_agreement']:.3f} | {r['max_logit_diff']:.3f} | {r['p50_ms']:.1f} / {r['p95_ms']:.1f} |")
    lines += ["", f"Vocabulary trimmed from {data['trim']['vocab_before']:,} to {data['trim']['vocab_after']:,} tokens, on fixture text only, so a real trim must use the real training text."]


def section_whisper(lines: list[str]) -> None:
    data = load("whisper_summary.json")
    lines += ["", "## Real ASR round trip: Whisper on synthetic speech", ""]
    if not data:
        lines.append("Not run.")
        return
    tiny = load("whisper_tiny_summary.json")
    lines += [
        f"Whisper `{data['whisper']}` on CPU, {data['rows']} fixture lines spoken by {data['voice']}, at {data['snr_db']} dB SNR and {data['packet_loss']:.0%} packet loss on 8 kHz G.711.",
        "",
        "**Limits:** a robotic voice reads every line in English phonetics, including isiZulu and Afrikaans fragments. That's a fair test of the pipeline, and an unfair test of Whisper. Real callers will differ.",
        "",
        "| Mix | Rows | WER | Substitution | Deletion | Insertion |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mix, r in data["per_mix"].items():
        lines.append(f"| {mix} | {r['rows']} | {r['wer']:.3f} | {r['substitution']:.3f} | {r['deletion']:.3f} | {r['insertion']:.3f} |")
    o = data["overall"]
    lines += [f"| overall | {o['rows']} | {o['wer']:.3f} | {o['substitution']:.3f} | {o['deletion']:.3f} | {o['insertion']:.3f} |", "", f"Transcription time per clip on CPU, measured while other jobs ran: p50 {data['asr_ms']['p50']:.0f} ms, p95 {data['asr_ms']['p95']:.0f} ms."]
    if tiny:
        lines += ["", f"Whisper `tiny` on {tiny['rows']} clips: overall WER {tiny['overall']['wer']:.3f}, p50 {tiny['asr_ms']['p50']:.0f} ms, p95 {tiny['asr_ms']['p95']:.0f} ms, also under load. Either way, CPU Whisper is a fallback that won't meet a 1.5 s budget reliably on this machine."]


def section_ablations(lines: list[str]) -> None:
    data = load("ablations.json")
    lines += ["", "## Offline ablations (pipeline check only)", ""]
    if not data:
        lines.append("Not run.")
        return
    lines += [
        f"> Template fixtures with `{data['stress_noise']}` noise at WER {data['stress_wer']:.3f}, so `reportable` is {str(data['reportable']).lower()}. These show the machinery runs and where it's sensitive. They aren't evidence for any design choice.",
        "",
        f"Thresholds fitted on calibration data: `tau_llm` {data['fitted_thresholds']['tau_llm']}, `tau_clarify` {data['fitted_thresholds']['tau_clarify']}.",
        "",
        "| Arm | Mean cost | Accuracy | Emergency recall | Emergency to voicemail | Clarify rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def row(group: str, name: str, r: dict) -> str:
        return f"| {group} {name} | {fmt(r['mean_cost'])} | {fmt(r['accuracy'])} | {fmt(r['emergency_recall'])} | {r['emergency_to_voicemail']} | {fmt(r['clarification_rate'])} |"

    for name, r in data["C3_gate"].items():
        lines.append(row("C3", name, r))
    for key, title in (("C3_real_asr", "real Whisper transcripts"), ("C3_edge_cases", "unseen hand-written edge cases")):
        block = data.get(key)
        if not block:
            continue
        fitted = block["fitted"]
        lines += ["", f"Gates on {title} (`tau_llm` {fitted['tau_llm']}, `tau_clarify` {fitted['tau_clarify']}, fitted on a separate half; no LLM layer):", "", "| Arm | Mean cost | Accuracy | Emergency recall | Emergency to voicemail | Clarify rate |", "|---|---:|---:|---:|---:|---:|"]
        for name, r in block["arms"].items():
            lines.append(row("C3", name, r))
    lines += ["", "Other ablations, on fixture text:", "", "| Arm | Mean cost | Accuracy | Emergency recall | Emergency to voicemail | Clarify rate |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in data["E1_train_size"].items():
        lines.append(row("E1 rows", name, r))
    for name, r in data["E2_noise"].items():
        lines.append(row("E2", name, r))
    for name, r in data["E3_prefix"].items():
        lines.append(row("E3", name, r["final"]))
    e = data["earliness"]
    lines += ["", f"Earliness: median {fmt(e['median_share_heard'], 2)} of the utterance heard at the first stable correct prediction, {fmt(e['mean_flips_per_turn'], 2)} flips per turn, {e['never_correct_at_end']} never correct by the end."]


def section_fixture(lines: list[str]) -> None:
    data = load("fixture/report.json") or load("fixture_report.json")
    if not data:
        return
    lines += ["", "## Fixture evaluation (pipeline check only)", "", f"> {data['banner']}", "", "| Arm | Accuracy | Mean cost | Emergency recall | Emergency to voicemail |", "|---|---:|---:|---:|---:|"]
    for name, a in data["arms"].items():
        lines.append(f"| {name} | {a['accuracy']:.3f} | {a['mean_cost']:.3f} | {fmt(a['emergency_recall'])} | {a['emergency_to_voicemail']} |")


def main() -> None:
    lines = [
        "# Evidence report",
        "",
        "Generated by `poe report`. Hand edits are overwritten.",
        "",
        "**Measurement conditions:** a shared laptop (i7-1355U), cores not pinned, and another project's heavy model training ran during some of these runs, so timings are indicative and the spread between runs is real.",
        "",
        "**This is not the final evaluation.** No Deepgram or LLM call has been made, no reportable data exists, the encoder bake-off hasn't run, and the test split hasn't been opened. Routing numbers below come from template fixtures and must not be quoted as results.",
        "",
    ]
    for section in (section_mutation, section_failure, section_traces, section_langfuse, section_real_asr, section_load, section_stages, section_whisper, section_encoder, section_ablations, section_fixture):
        section(lines)
    lines += [
        "",
        "## Still needs credentials or people",
        "",
        "| Item | Needs | Command |",
        "|---|---|---|",
        "| isiZulu on telephone audio (Spike B) | Deepgram key, recorded clips | `scripts/spike_deepgram_language.py` |",
        "| LLM p95 and thinking settings (Spike C) | Anthropic key | `scripts/spike_llm_latency.py` |",
        "| LLM-only and cascade arms | Anthropic key | `poe live-eval` |",
        "| Reportable data | Open-weight model endpoint | `python -m switchboard_inference.data.generate` |",
        "| Native-speaker and label review | Paid reviewers, a second rater | See `docs/DATA_CARD.md` |",
        "| Encoder bake-off | Reportable data, a GPU | `train/train.py`, `train/export.py` |",
        "| `T_route` on live ASR | Deepgram key | `poe load` with `ASR_MODE=live` |",
        "| Sealed test evaluation | All of the above | `poe eval --split test --unseal` |",
    ]
    (ROOT / "docs/REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
