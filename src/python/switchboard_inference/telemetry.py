"""Optional tracing and an event-loop lag monitor. Neither may ever block the audio path."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import contextmanager

log = logging.getLogger("switchboard.inference")
_tracer = None


def configure_tracing() -> bool:
    """OTLP over HTTP, when an endpoint is set and the optional extra is installed."""
    global _tracer
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "switchboard-inference"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")))
    _tracer = provider.get_tracer("switchboard.inference")
    return True


@contextmanager
def span(name: str, tenant_id: str, conversation_id: str = "", **attributes):
    """Every span carries its tenant. Transcript text must never be passed here."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        current.set_attribute("tenant.id", tenant_id)
        current.set_attribute("langfuse.session.id", conversation_id)
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield current


class LoopLagMonitor:
    """How late the event loop wakes up. Scoring in the loop would show here as lost audio."""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.samples: list[float] = []

    async def run(self, report_every: int = 1200) -> None:
        while True:
            started = time.perf_counter()
            await asyncio.sleep(self.interval_s)
            self.samples.append((time.perf_counter() - started - self.interval_s) * 1000)
            if len(self.samples) >= report_every:
                ordered = sorted(self.samples)
                log.info("event_loop_lag_ms p50=%.2f p99=%.2f max=%.2f", ordered[len(ordered) // 2], ordered[int(0.99 * len(ordered))], ordered[-1])
                self.samples.clear()
