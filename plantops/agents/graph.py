"""The PlantOps agent topology, built on Google ADK.

    plantops (SequentialAgent)
    ├── coordinator (LlmAgent, strong model)          hierarchical delegation
    │     ├── AgentTool(data_analyst)                 ReAct over guarded SQL + schema RAG
    │     └── AgentTool(knowledge_researcher)         ReAct over hybrid document search
    │     -> state["draft_answer"]
    └── reflection_loop (LoopAgent, max N passes)     self-reflection
          ├── critic   (verifies citations with tools; calls exit_loop when satisfied)
          │     -> state["critique"]
          └── refiner  (rewrites the draft from the critique)
                -> state["draft_answer"]

Specialists are wrapped as AgentTools rather than transfer targets so the coordinator keeps
control and does the synthesis; each AgentTool call also runs in a fresh child session, so
specialists only see the request they were given, not the whole conversation.
"""

from __future__ import annotations

import logging
import re
import time
from typing import AsyncGenerator

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import AgentTool, exit_loop
from google.genai import errors, types
from opentelemetry import trace

from plantops.agents import prompts
from plantops.config import Settings
from plantops.tools import functions as fn

ModelLike = str | BaseLlm
log = logging.getLogger(__name__)
_RETRYABLE = (429, 500, 503)
# 404 means the model is retired or not offered to this key (listed models can still 404), so
# it is skipped for the life of the process rather than failing the request.
_GONE = 404
# Circuit breaker shared by all agents in the process: a model that just exhausted its retries
# is skipped for a while instead of costing every following call the same retry delay.
COOLDOWN_S = 180.0
_cooling_until: dict[str, float] = {}


_RETRY_IN = re.compile(r"retry in ((?:[\d.]+[hms])+)")


def _retry_after(e: errors.APIError) -> float:
    """Seconds the server asked us to wait (RetryInfo, or "Please retry in 6h10m39s"), else 0.

    Per-minute limits come back with a short delay; an exhausted daily quota with hours.
    """
    for d in ((e.details or {}).get("error") or {}).get("details") or []:
        if str(d.get("@type", "")).endswith("RetryInfo") and str(d.get("retryDelay", "")).endswith("s"):
            try:
                return float(d["retryDelay"][:-1])
            except ValueError:
                pass
    m = _RETRY_IN.search(e.message or "")
    if not m:
        return 0.0
    units = {"h": 3600, "m": 60, "s": 1}
    return sum(float(n) * units[u] for n, u in re.findall(r"([\d.]+)([hms])", m.group(1)))


class FallbackGemini(Gemini):
    """Gemini that moves to the next model when one is overloaded or out of quota.

    Each model first gets its own HTTP retries; only when those are exhausted (and before any
    response has streamed back) does the request move down the chain, and the failed model is
    put on cooldown so later calls skip it. Switches are recorded as span events.
    """

    fallback_models: list[str] = []

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        primary = llm_request.model or self.model
        chain = [primary, *(m for m in self.fallback_models if m != primary)]
        now = time.monotonic()
        # Skip cooling models, but always keep the last one so there is something to try.
        chain = [m for m in chain[:-1] if _cooling_until.get(m, 0) <= now] + chain[-1:]
        if chain[0] != primary:
            trace.get_current_span().add_event("model.skipped_cooldown", {"model": primary, "using": chain[0]})
        for i, model in enumerate(chain):
            llm_request.model = model
            yielded = False
            try:
                async for resp in super().generate_content_async(llm_request, stream):
                    yielded = True
                    yield resp
                return
            except errors.APIError as e:
                if yielded or e.code not in (*_RETRYABLE, _GONE) or i == len(chain) - 1:
                    raise
                _cooling_until[model] = float("inf") if e.code == _GONE else time.monotonic() + max(COOLDOWN_S, _retry_after(e))
                log.warning("%s returned %s; falling back to %s", model, e.code, chain[i + 1])
                trace.get_current_span().add_event("model.fallback", {"from": model, "to": chain[i + 1], "status": e.code})


def gemini(model: str, fallbacks: list[str] | None = None) -> Gemini:
    # Free-tier quotas are low (requests/minute) and popular models return 503 under load, so
    # back off briefly, then fall back to the next model rather than failing the request.
    return FallbackGemini(
        model=model,
        fallback_models=fallbacks or [],
        retry_options=types.HttpRetryOptions(attempts=3, initial_delay=2.0, max_delay=10.0, exp_base=2.0, http_status_codes=list(_RETRYABLE)),
    )


def _tools(names: list, transport: str) -> list:
    if transport != "mcp":
        return list(names)
    from plantops.mcp_client import plantops_toolset

    return [plantops_toolset([f.__name__ for f in names])]


def build_agent(settings: Settings, worker: ModelLike | None = None, strong: ModelLike | None = None) -> SequentialAgent:
    worker = worker or gemini(settings.worker_model, settings.fallback_models)
    strong = strong or gemini(settings.strong_model, settings.fallback_models)
    precise = types.GenerateContentConfig(temperature=0.1)
    transport = settings.tool_transport

    data_analyst = LlmAgent(
        name="data_analyst",
        description="Answers quantitative questions with read-only SQL over the plant warehouse (downtime, tickets, inventory, purchase orders, suppliers, quality).",
        model=worker,
        instruction=prompts.DATA_ANALYST,
        tools=_tools(fn.DATA_TOOLS, transport),
        generate_content_config=precise,
    )
    knowledge_researcher = LlmAgent(
        name="knowledge_researcher",
        description="Finds procedures, specs, error-code meanings, policies, thresholds, root causes and past repair notes in the document corpus, with citations.",
        model=worker,
        instruction=prompts.KNOWLEDGE_RESEARCHER,
        tools=_tools(fn.DOC_TOOLS, transport),
        generate_content_config=precise,
    )
    coordinator = LlmAgent(
        name="coordinator",
        model=strong,
        instruction=prompts.COORDINATOR,
        tools=[AgentTool(data_analyst), AgentTool(knowledge_researcher)],
        output_key="draft_answer",
        generate_content_config=precise,
    )
    # Critic and refiner get the question and draft through state templating and skip the raw
    # conversation history (include_contents="none"), which keeps their prompts small.
    critic = LlmAgent(
        name="critic",
        model=strong,
        instruction=prompts.CRITIC,
        tools=[*_tools(fn.VERIFY_TOOLS, transport), exit_loop],
        include_contents="none",
        output_key="critique",
        generate_content_config=types.GenerateContentConfig(temperature=0.0),
    )
    refiner = LlmAgent(
        name="refiner",
        model=strong,
        instruction=prompts.REFINER,
        tools=_tools([fn.search_documents, fn.fetch_chunk, fn.run_sql], transport),
        include_contents="none",
        output_key="draft_answer",
        generate_content_config=precise,
    )
    reflection = LoopAgent(name="reflection_loop", sub_agents=[critic, refiner], max_iterations=settings.max_reflections)
    return SequentialAgent(name="plantops", sub_agents=[coordinator, reflection])
