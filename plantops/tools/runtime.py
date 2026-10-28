"""Process-wide handles to the warehouse and vector store, created on first use."""

from __future__ import annotations

from functools import lru_cache

from plantops.config import get_settings
from plantops.ingest.embeddings import make_embedder
from plantops.retrieval.store import VectorStore
from plantops.tools.sql import Warehouse


@lru_cache(maxsize=1)
def warehouse() -> Warehouse:
    return Warehouse(get_settings().warehouse_path)


@lru_cache(maxsize=1)
def store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.vector_dir, make_embedder(s))


def reset() -> None:
    """Drop cached handles (tests, or after a settings change)."""
    warehouse.cache_clear()
    store.cache_clear()
