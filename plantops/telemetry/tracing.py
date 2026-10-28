"""OpenTelemetry setup.

ADK already emits spans for agent invocations (invoke_agent), LLM calls (call_llm,
generate_content) and tool executions (execute_tool). We add spans for retrieval, SQL and
ingestion, and send all of them to a local SQLite exporter. Set PLANTOPS_CLOUD_TRACE=1 (with
the `gcp` extra installed) to also export to Google Cloud Trace, which includes a free
monthly allotment of spans.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult

from plantops.telemetry.store import TelemetryStore

MAX_ATTR_CHARS = 4000
_configured = False


class SqliteSpanExporter(SpanExporter):
    def __init__(self, store: TelemetryStore):
        self.store = store

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows = []
        for s in spans:
            attrs = {}
            for k, v in (s.attributes or {}).items():
                v = list(v) if isinstance(v, tuple) else v
                if isinstance(v, str) and len(v) > MAX_ATTR_CHARS:
                    v = v[:MAX_ATTR_CHARS] + f"… [{len(v)} chars]"
                attrs[k] = v
            rows.append({
                "trace_id": format(s.context.trace_id, "032x"),
                "span_id": format(s.context.span_id, "016x"),
                "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
                "name": s.name,
                "start_ns": s.start_time,
                "end_ns": s.end_time,
                "status": s.status.status_code.name,
                "attributes": json.dumps(attrs, default=str),
            })
        self.store.write("spans", rows)
        return SpanExportResult.SUCCESS


def setup_tracing(store: TelemetryStore, synchronous: bool = False) -> None:
    """Install the global tracer provider (idempotent). Call before building agents."""
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "plantops"}))
    exporter = SqliteSpanExporter(store)
    provider.add_span_processor(SimpleSpanProcessor(exporter) if synchronous else BatchSpanProcessor(exporter, schedule_delay_millis=500))
    if os.getenv("PLANTOPS_CLOUD_TRACE") == "1":
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

        provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


def flush() -> None:
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None
