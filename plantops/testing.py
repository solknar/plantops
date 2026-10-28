"""A scripted stand-in for Gemini so the full agent graph can run offline in tests.

Each agent is recognized by its system instruction; every model call for that agent pops
the next scripted step. Steps are either a text reply or a function call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

AGENT_MARKERS = {
    "coordinator": "You are the PlantOps coordinator",
    "data_analyst": "You are a data analyst",
    "knowledge_researcher": "You are a technical researcher",
    "critic": "You are a strict reviewer",
    "refiner": "You revise draft answers",
}


@dataclass
class Call:
    name: str
    args: dict = field(default_factory=dict)


class ScriptedLlm(BaseLlm):
    model: str = "gemini-3.5-flash-lite"
    script: dict[str, list] = {}
    seen: dict[str, list[LlmRequest]] = {}

    def __init__(self, script: dict[str, list], model: str = "gemini-3.5-flash-lite"):
        super().__init__(model=model, script={k: list(v) for k, v in script.items()}, seen={})

    @staticmethod
    def _agent(req: LlmRequest) -> str:
        si = req.config.system_instruction if req.config else None
        text = si if isinstance(si, str) else "".join(p.text or "" for p in (si.parts if si else []))
        for agent, marker in AGENT_MARKERS.items():
            if marker in text:
                return agent
        return "summarizer"

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        agent = self._agent(llm_request)
        self.seen.setdefault(agent, []).append(llm_request)
        steps = self.script.get(agent) or ["(no more scripted steps)"]
        step = steps.pop(0) if self.script.get(agent) else steps[0]
        if isinstance(step, Call):
            part = types.Part(function_call=types.FunctionCall(name=step.name, args=step.args))
        else:
            part = types.Part(text=step)
        prompt_chars = sum(len(str(c.model_dump(exclude_none=True))) for c in llm_request.contents or [])
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=max(1, prompt_chars // 4), candidates_token_count=40, total_token_count=max(1, prompt_chars // 4) + 40
            ),
            finish_reason=types.FinishReason.STOP,
        )
