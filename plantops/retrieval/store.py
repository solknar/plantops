"""LanceDB-backed vector store with hybrid (vector + BM25) retrieval fused by RRF."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import lancedb
import pyarrow as pa
from opentelemetry import trace

from plantops.ingest.chunking import Chunk
from plantops.ingest.embeddings import Embedder

tracer = trace.get_tracer("plantops.retrieval")

TABLE = "chunks"
FILTER_FIELDS = ("doc_type", "equipment_model", "plant_id", "machine_id", "error_code", "table")
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9 _.\-]{1,64}$")


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("title", pa.string()),
            pa.field("section", pa.string()),
            pa.field("text", pa.string()),
            # title + section path + text: headings carry a lot of the keyword signal for BM25
            pa.field("search_text", pa.string()),
            pa.field("content_hash", pa.string()),
            *[pa.field(f, pa.string()) for f in FILTER_FIELDS],
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


@dataclass
class Hit:
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    doc_type: str
    score: float
    matched_by: list[str]
    metadata: dict[str, str]

    def as_tool_result(self, max_chars: int = 700) -> dict:
        text = self.text if len(self.text) <= max_chars else self.text[:max_chars] + " …"
        return {
            "chunk_id": self.id,
            "title": self.title,
            "section": self.section,
            "doc_type": self.doc_type,
            "text": text,
            "score": round(self.score, 4),
            **{k: v for k, v in self.metadata.items() if v},
        }


def build_where(filters: dict[str, str] | None) -> str | None:
    """Translate user/agent filters into a LanceDB SQL predicate.

    Only whitelisted fields and conservative value characters are accepted, so a model
    hallucinating a filter can't inject arbitrary SQL into the store.
    """
    if not filters:
        return None
    clauses = []
    for k, v in filters.items():
        if v in (None, ""):
            continue
        if k not in FILTER_FIELDS:
            raise ValueError(f"unsupported filter field {k!r}; allowed: {FILTER_FIELDS}")
        if not _SAFE_VALUE.match(str(v)):
            raise ValueError(f"unsupported characters in filter value for {k!r}")
        clauses.append(f"`{k}` = '{v}'")
    return " AND ".join(clauses) or None


def _fts_query(q: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9]+", q))


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder):
        self.path = path
        self.embedder = embedder
        self.db = lancedb.connect(str(path))

    def _table(self):
        return self.db.open_table(TABLE)

    def exists(self) -> bool:
        return TABLE in self.db.list_tables().tables

    def reset(self) -> None:
        if self.exists():
            self.db.drop_table(TABLE)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        rows = []
        for c, v in zip(chunks, vectors, strict=True):
            md = c.metadata
            rows.append({
                "id": c.chunk_id, "doc_id": c.doc_id, "source": c.source, "title": c.title,
                "section": c.section, "text": c.text, "search_text": c.embedding_text(),
                "content_hash": c.content_hash,
                "doc_type": c.doc_type,
                **{f: md.get(f, "") for f in FILTER_FIELDS if f != "doc_type"},
                "vector": v,
            })
        if not self.exists():
            self.db.create_table(TABLE, schema=_schema(self.embedder.dim))
        if rows:
            tbl = self._table()
            tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)

    def delete_ids(self, ids: list[str]) -> None:
        if not ids or not self.exists():
            return
        tbl = self._table()
        for i in range(0, len(ids), 200):
            batch = ", ".join("'" + x.replace("'", "''") + "'" for x in ids[i : i + 200])
            tbl.delete(f"id IN ({batch})")

    def build_indexes(self) -> None:
        tbl = self._table()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # create_fts_index is deprecated but still the clearest API
            tbl.create_fts_index("search_text", replace=True)
        # ANN index only pays off at scale; below ~10k rows a flat scan is exact and fast.
        if tbl.count_rows() >= 10_000:
            tbl.create_index(metric="cosine", index_type="IVF_HNSW_SQ", replace=True)

    def count(self) -> int:
        return self._table().count_rows() if self.exists() else 0

    def get(self, chunk_id: str) -> Hit | None:
        if not self.exists() or not _SAFE_VALUE.match(chunk_id.replace("#", "-")):
            return None
        rows = self._table().search().where(f"id = '{chunk_id}'").limit(1).to_list()
        return self._to_hit(rows[0], 1.0, ["lookup"]) if rows else None

    def _to_hit(self, r: dict, score: float, matched_by: list[str]) -> Hit:
        return Hit(
            id=r["id"], doc_id=r["doc_id"], title=r["title"], section=r["section"], text=r["text"],
            doc_type=r["doc_type"], score=score, matched_by=matched_by,
            metadata={f: r.get(f, "") for f in FILTER_FIELDS if f != "doc_type"},
        )

    def search(self, query: str, k: int = 5, filters: dict[str, str] | None = None,
               candidates: int = 20, rrf_k: int = 60) -> list[Hit]:
        """Hybrid search: dense + BM25 candidates, merged with Reciprocal Rank Fusion."""
        if not self.exists():
            return []
        where = build_where(filters)
        tbl = self._table()
        with tracer.start_as_current_span("retrieval.hybrid_search") as span:
            span.set_attribute("retrieval.query", query[:200])
            span.set_attribute("retrieval.k", k)
            span.set_attribute("retrieval.filters", str(filters or {}))

            with tracer.start_as_current_span("retrieval.dense"):
                q = tbl.search(self.embedder.embed_query(query)).metric("cosine")
                if where:
                    q = q.where(where, prefilter=True)
                dense = q.limit(candidates).to_list()

            sparse: list[dict] = []
            fq = _fts_query(query)
            if fq:
                with tracer.start_as_current_span("retrieval.bm25"):
                    q = tbl.search(fq, query_type="fts")
                    if where:
                        q = q.where(where, prefilter=True)
                    sparse = q.limit(candidates).to_list()

            fused: dict[str, dict] = {}
            for name, results in (("dense", dense), ("bm25", sparse)):
                for rank, r in enumerate(results):
                    e = fused.setdefault(r["id"], {"row": r, "score": 0.0, "by": []})
                    e["score"] += 1.0 / (rrf_k + rank + 1)
                    e["by"].append(name)
            ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:k]
            hits = [self._to_hit(e["row"], e["score"], e["by"]) for e in ranked]
            span.set_attribute("retrieval.dense_candidates", len(dense))
            span.set_attribute("retrieval.bm25_candidates", len(sparse))
            span.set_attribute("retrieval.returned_ids", [h.id for h in hits])
            return hits
