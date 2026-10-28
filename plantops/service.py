"""Wires agents, persistent sessions, plugins and tracing into one callable service."""

from __future__ import annotations

import time
from typing import Any

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, DatabaseSessionService
from google.genai import types
from opentelemetry import trace

from plantops.agents.context_budget import ContextBudgetPlugin
from plantops.agents.graph import ModelLike, build_agent, gemini
from plantops.config import Settings
from plantops.telemetry.metrics import MetricsPlugin
from plantops.telemetry.store import TelemetryStore
from plantops.telemetry.tracing import current_trace_id, flush, setup_tracing

tracer = trace.get_tracer("plantops.service")
APP_NAME = "plantops"


class PlantOpsService:
    def __init__(
        self,
        settings: Settings,
        worker: ModelLike | None = None,
        strong: ModelLike | None = None,
        session_service: BaseSessionService | None = None,
        synchronous_tracing: bool = False,
    ):
        self.settings = settings
        self.telemetry = TelemetryStore(settings.telemetry_path)
        setup_tracing(self.telemetry, synchronous=synchronous_tracing)

        self.metrics = MetricsPlugin(self.telemetry)
        self.budget = ContextBudgetPlugin(on_saved=self.metrics.record_trimming)
        summarizer_llm = worker if worker is not None and not isinstance(worker, str) else gemini(settings.worker_model)
        app = App(
            name=APP_NAME,
            root_agent=build_agent(settings, worker, strong),
            plugins=[self.budget, self.metrics],
            # Long sessions: every 4 invocations, summarize older events into one compacted
            # event (keeping 1 invocation of overlap) instead of replaying the full history.
            events_compaction_config=EventsCompactionConfig(
                compaction_interval=4, overlap_size=1, summarizer=LlmEventSummarizer(llm=summarizer_llm)
            ),
        )
        self.sessions = session_service or DatabaseSessionService(db_url=settings.sessions_url)
        self.runner = Runner(app=app, session_service=self.sessions)

    async def _ensure_session(self, user_id: str, session_id: str | None) -> str:
        if session_id:
            s = await self.sessions.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
            if s is not None:
                return s.id
        s = await self.sessions.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        return s.id

    async def ask(self, question: str, user_id: str = "demo", session_id: str | None = None) -> dict[str, Any]:
        session_id = await self._ensure_session(user_id, session_id)
        steps: list[dict[str, Any]] = []
        last_text = ""
        t0 = time.perf_counter()
        with tracer.start_as_current_span("plantops.request") as span:
            span.set_attribute("plantops.question", question[:500])
            span.set_attribute("session.id", session_id)
            trace_id = current_trace_id()
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=question)]),
                # Per-turn values the critic/refiner read through instruction templating.
                state_delta={"question": question, "draft_answer": "", "critique": ""},
            ):
                for call in event.get_function_calls():
                    steps.append({"agent": event.author, "type": "tool_call", "name": call.name, "args": dict(call.args or {})})
                if event.content and event.content.parts:
                    text = "".join(p.text or "" for p in event.content.parts if not getattr(p, "thought", False))
                    if text.strip() and not event.partial:
                        last_text = text
                        steps.append({"agent": event.author, "type": "message", "text": text[:2000]})
        session = await self.sessions.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        state = session.state if session else {}
        answer = state.get("draft_answer") or last_text
        critique = str(state.get("critique") or "")
        flush()
        metrics = self.metrics.completed.pop(trace_id, None) or {"latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        # exit_loop skips the critic's closing summary, so approval is read from the tool call.
        approved = any(s["type"] == "tool_call" and s["name"] == "exit_loop" for s in steps)
        return {
            "answer": answer,
            "approved_by_critic": approved,
            "critique": critique,
            "session_id": session_id,
            "trace_id": trace_id,
            "metrics": metrics,
            "steps": steps,
        }
