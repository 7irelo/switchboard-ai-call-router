"""Copies the small result files out of gitignored runs/ into docs/evidence, then rebuilds the report."""

from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

from build_report import main as build_report

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "docs/evidence"
FILES = {
    "mutation_check.json": "mutation_check.json",
    "failure_injection.json": "failure_injection.json",
    "chaos.json": "chaos.json",
    "trace_isolation.json": "trace_isolation.json",
    "load_e2e.json": "load_e2e.json",
    "micro_tfidf.json": "micro_tfidf.json",
    "ablations.json": "ablations.json",
    "real_asr.json": "real_asr.json",
    "bakeoff.json": "bakeoff.json",
    "whisper_latency.json": "whisper_latency.json",
    "langfuse_check.json": "langfuse_check.json",
    "tracing_overhead.json": "tracing_overhead.json",
    "whisper_tiny/summary.json": "whisper_tiny_summary.json",
    "fixture/report.json": "fixture_report.json",
    "whisper_roundtrip/summary.json": "whisper_summary.json",
    "encoder-code-check/export_report.json": "encoder_export.json",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source, target in FILES.items():
        path = RUNS / source
        if path.exists():
            shutil.copy2(path, OUT / target)
    # BenchmarkDotNet writes a dated file name, and only its statistics are kept.
    results = glob.glob(str(ROOT / "**/BenchmarkDotNet.Artifacts/results/*-report-full.json"), recursive=True)
    if results:
        raw = json.loads(Path(max(results, key=lambda p: Path(p).stat().st_mtime)).read_text(encoding="utf-8"))
        kept = {"Benchmarks": [{"Method": b["Method"], "Statistics": {"Mean": b["Statistics"]["Mean"], "Percentiles": {"P95": b["Statistics"]["Percentiles"]["P95"]}}} for b in raw["Benchmarks"]]}
        (OUT / "micro_decision.json").write_text(json.dumps(kept, indent=1), encoding="utf-8")
    build_report()


if __name__ == "__main__":
    main()
