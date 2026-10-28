"""Context-window budgeting: shrink stale tool output before it is resent to the model.

In a ReAct loop every tool observation is replayed on every later model call, so a
50-row SQL result fetched on turn 1 is paid for again on each later turn. This plugin keeps the
most recent turns verbatim and replaces older function responses with a short stub (the
call itself stays, so the model still knows what it did and can re-run it). It also caps any
single oversized observation. Function call/response pairing is preserved, which Gemini
requires.

Works alongside ADK's EventsCompactionConfig, which summarizes whole old invocations in
long multi-turn sessions; this plugin works inside a single invocation.
"""

from __future__ import annotations

import json
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types
from opentelemetry import trace

CHARS_PER_TOKEN = 4  # rough heuristic for English/JSON; good enough for budgeting


def _response_chars(part: types.Part) -> int:
    return len(json.dumps(part.function_response.response, default=str)) if part.function_response else 0


class ContextBudgetPlugin(BasePlugin):
    def __init__(self, keep_recent_tool_results: int = 4, max_tool_result_chars: int = 6000, on_saved=None, name: str = "context_budget"):
        super().__init__(name=name)
        self.keep_recent = keep_recent_tool_results
        self.max_chars = max_tool_result_chars
        self.on_saved = on_saved

    def trim(self, contents: list[types.Content]) -> tuple[list[types.Content], int]:
        """Return new contents (inputs are not mutated) and the number of chars removed."""
        positions = [(ci, pi) for ci, c in enumerate(contents) for pi, p in enumerate(c.parts or []) if p.function_response]
        recent = set(positions[-self.keep_recent :]) if self.keep_recent else set()
        saved = 0
        out = []
        for ci, c in enumerate(contents):
            new_parts, changed = [], False
            for pi, p in enumerate(c.parts or []):
                n = _response_chars(p)
                if not p.function_response or n == 0:
                    new_parts.append(p)
                    continue
                if (ci, pi) not in recent and n > 400:
                    stub = {"elided": f"older result of {p.function_response.name} removed to save context ({n} chars); call the tool again if needed"}
                elif n > self.max_chars:
                    raw = json.dumps(p.function_response.response, default=str)
                    stub = {"truncated": raw[: self.max_chars], "note": f"result truncated from {n} chars"}
                else:
                    new_parts.append(p)
                    continue
                fr = p.function_response.model_copy(update={"response": stub})
                new_parts.append(p.model_copy(update={"function_response": fr}))
                saved += n - len(json.dumps(stub))
                changed = True
            out.append(c.model_copy(update={"parts": new_parts}) if changed else c)
        return out, max(saved, 0)

    async def before_model_callback(self, *, callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        llm_request.contents, saved = self.trim(llm_request.contents or [])
        if saved:
            tokens = saved // CHARS_PER_TOKEN
            trace.get_current_span().set_attribute("plantops.context.tokens_saved", tokens)
            if self.on_saved:
                self.on_saved(tokens)
        return None
