"""Runtime settings, resolved from the environment (and .env) once per process."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Paid-tier list prices in USD per 1M tokens, from ai.google.dev/gemini-api/docs/pricing
# (retrieved 2026-10-27). The free tier bills $0; we still report list-price cost so the
# numbers mean something when the customer moves to a paid tier.
PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "cached": 0.075},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75, "cached": 0.075},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached": 0.05},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "cached": 0.03},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50, "cached": 0.025},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached": 0.125},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached": 0.03},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "cached": 0.01},
    "gemini-embedding-2": {"input": 0.20, "output": 0.0, "cached": 0.0},
}


@dataclass(frozen=True)
class Settings:
    worker_model: str
    strong_model: str
    embed_model: str
    embedder: str
    tool_transport: str
    max_reflections: int
    var_dir: Path
    fallback_models: list[str] = field(default_factory=list)
    max_llm_calls: int = 40
    docs_dir: Path = field(default=ROOT / "data" / "docs")
    structured_dir: Path = field(default=ROOT / "data" / "structured")
    embed_dim: int = 768

    @property
    def warehouse_path(self) -> Path:
        return self.var_dir / "warehouse.db"

    @property
    def vector_dir(self) -> Path:
        return self.var_dir / "lancedb"

    @property
    def manifest_path(self) -> Path:
        return self.var_dir / "ingest_manifest.json"

    @property
    def sessions_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.var_dir / 'sessions.db'}"

    @property
    def telemetry_path(self) -> Path:
        return self.var_dir / "telemetry.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    var_dir = Path(os.getenv("PLANTOPS_VAR_DIR", "var"))
    if not var_dir.is_absolute():
        var_dir = ROOT / var_dir
    var_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        worker_model=os.getenv("PLANTOPS_WORKER_MODEL", "gemini-3.5-flash-lite"),
        strong_model=os.getenv("PLANTOPS_STRONG_MODEL", "gemini-3.8-flash"),
        embed_model=os.getenv("PLANTOPS_EMBED_MODEL", "gemini-embedding-2"),
        embedder=os.getenv("PLANTOPS_EMBEDDER", "gemini"),
        tool_transport=os.getenv("PLANTOPS_TOOL_TRANSPORT", "inprocess"),
        max_reflections=int(os.getenv("PLANTOPS_MAX_REFLECTIONS", "2")),
        max_llm_calls=int(os.getenv("PLANTOPS_MAX_LLM_CALLS", "40")),
        var_dir=var_dir,
        fallback_models=[m.strip() for m in os.getenv("PLANTOPS_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3-flash-preview,gemini-3.5-flash-lite").split(",") if m.strip()],
    )
