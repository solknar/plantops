"""LLM-native metrics as an ADK plugin: tokens/sec, TTFT, latency and cost per call/agent/request.

ADK runs each AgentTool call in its own child Runner with its own invocation_id, and the
plugins are inherited, so callbacks for one user request arrive under several invocation ids.
They all share one OpenTelemetry trace, so metrics are grouped by trace_id and a request is
finalized when its outermost run ends.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from opentelemetry import trace

from plantops.config import PRICING_USD_PER_1M
from plantops.telemetry.store import TelemetryStore
from plantops.telemetry.tracing import current_trace_id


def cost_usd(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """List-price cost. Thinking tokens are billed as output, so pass them in output_tokens."""
    p = PRICING_USD_PER_1M.get(_base_model(model))
    if p is None:
        return 0.0
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * p["input"] + cached_tokens * p["cached"] + output_tokens * p["output"]) / 1e6


def _base_model(model: str) -> str:
    return model.rsplit("/", 1)[-1]


@dataclass
class ModelCall:
    trace_id: str
    agent: str
    model: str
    started_at: float
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    cached_tokens: int = 0
    tokens_per_sec: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str = ""


@dataclass
class ToolCall:
    trace_id: str
    agent: str
    tool: str
    started_at: float
    latency_ms: float = 0.0
    error: int = 0
    result_chars: int = 0


@dataclass
class RequestMetrics:
    trace_id: str
    session_id: str
    user_id: str
    question: str
    started_at: float
    depth: int = 0
    latency_ms: float = 0.0
    tokens_saved_by_trimming: int = 0
    agent_runs: dict[str, int] = field(default_factory=dict)
    model_calls: list[ModelCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        per_agent: dict[str, dict[str, float]] = {}
        for c in self.model_calls:
            a = per_agent.setdefault(c.agent, {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0})
            a["llm_calls"] += 1
            a["input_tokens"] += c.input_tokens
            a["output_tokens"] += c.output_tokens + c.thoughts_tokens
            a["cost_usd"] = round(a["cost_usd"] + c.cost_usd, 6)
            a["latency_ms"] = round(a["latency_ms"] + c.latency_ms, 1)
        gen_ms = sum(c.latency_ms for c in self.model_calls) or 1.0
        out = sum(c.output_tokens + c.thoughts_tokens for c in self.model_calls)
        ttfts = [c.ttft_ms for c in self.model_calls if c.ttft_ms is not None]
        return {
            "trace_id": self.trace_id,
            "latency_ms": round(self.latency_ms, 1),
            "llm_calls": len(self.model_calls),
            "tool_calls": len(self.tool_calls),
            "tool_errors": sum(t.error for t in self.tool_calls),
            "input_tokens": sum(c.input_tokens for c in self.model_calls),
            "output_tokens": sum(c.output_tokens for c in self.model_calls),
            "thoughts_tokens": sum(c.thoughts_tokens for c in self.model_calls),
            "cached_tokens": sum(c.cached_tokens for c in self.model_calls),
            "cost_usd": round(sum(c.cost_usd for c in self.model_calls), 6),
            "output_tokens_per_sec": round(out / (gen_ms / 1000), 1),
            "mean_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
            "reflections": self.agent_runs.get("critic", 0),
            "agent_runs": dict(self.agent_runs),
            "tokens_saved_by_trimming": self.tokens_saved_by_trimming,
            "per_agent": per_agent,
        }


class MetricsPlugin(BasePlugin):
    def __init__(self, store: TelemetryStore | None = None, name: str = "plantops_metrics"):
        super().__init__(name=name)
        self.store = store
        self._requests: dict[str, RequestMetrics] = {}
        self._inflight_model: dict[tuple[str, str], ModelCall] = {}
        self._inflight_tool: dict[tuple[str, str, str], ToolCall] = {}
        self.completed: dict[str, dict] = {}

    # Request lifecycle -----------------------------------------------------------------
    def _trace_id(self, fallback: str) -> str:
        return current_trace_id() or fallback

    async def before_run_callback(self, *, invocation_context: InvocationContext) -> None:
        tid = self._trace_id(invocation_context.invocation_id)
        req = self._requests.get(tid)
        if req is None:
            question = ""
            if invocation_context.user_content and invocation_context.user_content.parts:
                question = "".join(p.text or "" for p in invocation_context.user_content.parts)
            req = RequestMetrics(tid, invocation_context.session.id, invocation_context.user_id, question, time.time())
            self._requests[tid] = req
        req.depth += 1
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        tid = self._trace_id(invocation_context.invocation_id)
        req = self._requests.get(tid)
        if req is None:
            return None
        req.depth -= 1
        if req.depth > 0:
            return None
        req.latency_ms = (time.time() - req.started_at) * 1000
        summary = req.summary()
        self.completed[tid] = summary
        span = trace.get_current_span()
        for k, v in summary.items():
            if isinstance(v, (int, float)) and v is not None:
                span.set_attribute(f"plantops.{k}", v)
        if self.store:
            self.store.write("requests", [{
                "trace_id": tid, "session_id": req.session_id, "user_id": req.user_id, "question": req.question,
                "started_at": req.started_at, **{k: summary[k] for k in (
                    "latency_ms", "llm_calls", "tool_calls", "tool_errors", "input_tokens", "output_tokens",
                    "thoughts_tokens", "cached_tokens", "cost_usd", "reflections", "tokens_saved_by_trimming")},
                "per_agent": json.dumps(summary["per_agent"]),
            }])
            self.store.write("model_calls", [asdict(c) for c in req.model_calls])
            self.store.write("tool_calls", [asdict(t) for t in req.tool_calls])
        del self._requests[tid]
        return None

    def record_trimming(self, tokens_saved: int) -> None:
        req = self._requests.get(current_trace_id() or "")
        if req:
            req.tokens_saved_by_trimming += tokens_saved

    async def before_agent_callback(self, *, agent, callback_context: CallbackContext) -> None:
        req = self._requests.get(self._trace_id(callback_context.invocation_id))
        if req:
            req.agent_runs[agent.name] = req.agent_runs.get(agent.name, 0) + 1
        return None

    # Model calls -------------------------------------------------------------------------
    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        tid = self._trace_id(callback_context.invocation_id)
        key = (callback_context.invocation_id, callback_context.agent_name)
        self._inflight_model[key] = ModelCall(tid, callback_context.agent_name, llm_request.model or "", time.time())
        return None

    async def after_model_callback(self, *, callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
        key = (callback_context.invocation_id, callback_context.agent_name)
        call = self._inflight_model.get(key)
        if call is None:
            return None
        now = time.time()
        if llm_response.partial:
            # Streaming: the first partial chunk marks time-to-first-token.
            if call.ttft_ms is None:
                call.ttft_ms = (now - call.started_at) * 1000
            return None
        del self._inflight_model[key]
        call.latency_ms = (now - call.started_at) * 1000
        # The model that actually answered (it differs from the request after a fallback).
        call.model = llm_response.model_version or call.model
        u = llm_response.usage_metadata
        if u is not None:
            call.input_tokens = (u.prompt_token_count or 0) + (u.tool_use_prompt_token_count or 0)
            call.output_tokens = u.candidates_token_count or 0
            call.thoughts_tokens = u.thoughts_token_count or 0
            call.cached_tokens = u.cached_content_token_count or 0
        generated = call.output_tokens + call.thoughts_tokens
        # Throughput over generation time; exclude TTFT when streaming so queueing/prefill
        # doesn't deflate the decode rate.
        gen_s = (call.latency_ms - (call.ttft_ms or 0)) / 1000
        call.tokens_per_sec = round(generated / gen_s, 1) if gen_s > 0 else 0.0
        call.cost_usd = cost_usd(call.model, call.input_tokens, generated, call.cached_tokens)
        call.finish_reason = llm_response.finish_reason.name if llm_response.finish_reason else ""

        span = trace.get_current_span()
        span.set_attribute("plantops.llm.tokens_per_sec", call.tokens_per_sec)
        span.set_attribute("plantops.llm.cost_usd", call.cost_usd)
        if call.ttft_ms is not None:
            span.set_attribute("plantops.llm.ttft_ms", call.ttft_ms)

        req = self._requests.get(call.trace_id)
        if req:
            req.model_calls.append(call)
        return None

    # Tool calls --------------------------------------------------------------------------
    async def before_tool_callback(self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext) -> Optional[dict]:
        key = (tool_context.invocation_id, tool.name, tool_context.function_call_id or "")
        self._inflight_tool[key] = ToolCall(self._trace_id(tool_context.invocation_id), tool_context.agent_name, tool.name, time.time())
        return None

    async def after_tool_callback(self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, result: dict) -> Optional[dict]:
        key = (tool_context.invocation_id, tool.name, tool_context.function_call_id or "")
        call = self._inflight_tool.pop(key, None)
        if call is None:
            return None
        call.latency_ms = (time.time() - call.started_at) * 1000
        call.error = int(isinstance(result, dict) and "error" in result)
        call.result_chars = len(json.dumps(result, default=str)) if result is not None else 0
        req = self._requests.get(call.trace_id)
        if req:
            req.tool_calls.append(call)
        return None

    async def on_tool_error_callback(self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext, error: Exception) -> Optional[dict]:
        await self.after_tool_callback(tool=tool, tool_args=tool_args, tool_context=tool_context, result={"error": str(error)})
        return None


def aggregate(store: TelemetryStore, limit: int = 500) -> dict[str, Any]:
    """Fleet-level view across recent requests: latency percentiles, cost, throughput by model."""
    reqs = store.query("SELECT * FROM requests ORDER BY started_at DESC LIMIT ?", (limit,))
    if not reqs:
        return {"requests": 0}
    lat = sorted(r["latency_ms"] for r in reqs)

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1)

    by_model = store.query(
        """SELECT model, COUNT(*) calls, AVG(tokens_per_sec) avg_tokens_per_sec, AVG(latency_ms) avg_latency_ms,
                  AVG(ttft_ms) avg_ttft_ms, SUM(input_tokens) input_tokens, SUM(output_tokens + thoughts_tokens) output_tokens,
                  SUM(cost_usd) cost_usd
           FROM model_calls WHERE trace_id IN (SELECT trace_id FROM requests ORDER BY started_at DESC LIMIT ?)
           GROUP BY model""",
        (limit,),
    )
    by_tool = store.query(
        """SELECT tool, COUNT(*) calls, AVG(latency_ms) avg_latency_ms, SUM(error) errors, AVG(result_chars) avg_result_chars
           FROM tool_calls WHERE trace_id IN (SELECT trace_id FROM requests ORDER BY started_at DESC LIMIT ?)
           GROUP BY tool ORDER BY calls DESC""",
        (limit,),
    )
    n = len(reqs)
    return {
        "requests": n,
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "max": round(lat[-1], 1)},
        "avg_cost_usd_per_request": round(sum(r["cost_usd"] for r in reqs) / n, 6),
        "total_cost_usd": round(sum(r["cost_usd"] for r in reqs), 6),
        "avg_tokens_per_request": round(sum(r["input_tokens"] + r["output_tokens"] + r["thoughts_tokens"] for r in reqs) / n),
        "avg_llm_calls_per_request": round(sum(r["llm_calls"] for r in reqs) / n, 2),
        "avg_reflections": round(sum(r["reflections"] for r in reqs) / n, 2),
        "tokens_saved_by_trimming": sum(r["tokens_saved_by_trimming"] for r in reqs),
        "by_model": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()} for m in by_model],
        "by_tool": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items()} for t in by_tool],
    }
