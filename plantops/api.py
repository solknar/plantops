"""HTTP API and dashboard.

    POST /api/chat            ask a question (optionally continuing a session)
    GET  /api/metrics         aggregate latency / cost / tokens-per-sec across recent requests
    GET  /api/requests        recent requests with per-request metrics
    GET  /api/traces/{id}     span tree for one request (agents, LLM calls, tools, retrieval, SQL)
    GET  /                    dashboard
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from plantops.config import get_settings
from plantops.telemetry.metrics import aggregate

_service = None


def service():
    global _service
    if _service is None:
        from plantops.service import PlantOpsService

        _service = PlantOpsService(get_settings())
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    service()
    yield


app = FastAPI(title="PlantOps", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: str | None = None
    user_id: str = "web"


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    return await service().ask(req.question, user_id=req.user_id, session_id=req.session_id)


@app.get("/api/metrics")
def metrics() -> dict:
    return aggregate(service().telemetry)


@app.get("/api/requests")
def requests(limit: int = 25) -> list[dict]:
    rows = service().telemetry.query("SELECT * FROM requests ORDER BY started_at DESC LIMIT ?", (min(limit, 200),))
    for r in rows:
        r["per_agent"] = json.loads(r["per_agent"] or "{}")
    return rows


@app.get("/api/traces/{trace_id}")
def get_trace(trace_id: str) -> list[dict]:
    tree = service().telemetry.trace_tree(trace_id)
    if not tree:
        raise HTTPException(404, "trace not found (spans are flushed ~0.5 s after a request)")
    return tree


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "web" / "index.html").read_text()
