from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import errors, types

from plantops.agents import graph


def _fake_backend(monkeypatch, down: set[str], gone: frozenset[str] = frozenset()):
    calls: list[str] = []

    async def fake(self, llm_request, stream=False):
        calls.append(llm_request.model)
        if llm_request.model in gone:
            raise errors.ClientError(404, {"error": {"code": 404, "message": "no longer available", "status": "NOT_FOUND"}})
        if llm_request.model in down:
            raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="ok")]), model_version=llm_request.model)

    monkeypatch.setattr(Gemini, "generate_content_async", fake)
    monkeypatch.setattr(graph, "_cooling_until", {})
    return calls


async def _ask(llm) -> list[LlmResponse]:
    return [r async for r in llm.generate_content_async(LlmRequest(model="primary"))]


async def test_falls_back_and_skips_cooling_model(monkeypatch):
    calls = _fake_backend(monkeypatch, down={"primary"})
    llm = graph.gemini("primary", ["backup", "last"])

    first = await _ask(llm)
    assert first[0].model_version == "backup"
    second = await _ask(llm)
    assert second[0].model_version == "backup"
    # The second request goes straight to the backup while the primary cools down.
    assert calls == ["primary", "backup", "backup"]


async def test_raises_when_every_model_fails(monkeypatch):
    _fake_backend(monkeypatch, down={"primary", "backup"})
    llm = graph.gemini("primary", ["backup"])
    try:
        await _ask(llm)
    except errors.ServerError as e:
        assert e.code == 503
    else:
        raise AssertionError("expected ServerError")


async def test_skips_retired_model(monkeypatch):
    calls = _fake_backend(monkeypatch, down={"primary"}, gone=frozenset({"retired"}))
    llm = graph.gemini("primary", ["retired", "last"])
    assert (await _ask(llm))[0].model_version == "last"
    assert calls == ["primary", "retired", "last"]


def test_retry_after_parses_server_hints():
    daily = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded. Please retry in 6h10m39.5s."}})
    assert graph._retry_after(daily) == 6 * 3600 + 10 * 60 + 39.5
    info = errors.ClientError(429, {"error": {"code": 429, "message": "slow down", "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "22s"}]}})
    assert graph._retry_after(info) == 22.0
    assert graph._retry_after(errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}})) == 0.0
