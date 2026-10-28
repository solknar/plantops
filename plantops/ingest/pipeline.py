"""End-to-end ingestion: structured + unstructured sources into the warehouse and vector store.

Incremental by default: a manifest maps chunk_id -> content_hash, so re-running only
embeds chunks that are new or changed and deletes chunks whose source disappeared. A change
of embedding model, dimension or index version forces a full rebuild, since vectors from different models
aren't comparable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from opentelemetry import trace

from plantops.config import Settings
from plantops.ingest.chunking import Chunk, load_documents
from plantops.ingest.embeddings import Embedder
from plantops.ingest.structured import load_warehouse, schema_cards, ticket_note_chunks
from plantops.retrieval.store import VectorStore

tracer = trace.get_tracer("plantops.ingest")

# Bump when the vector table schema or chunking changes, to force a rebuild.
INDEX_VERSION = 2


@dataclass
class IngestStats:
    warehouse_rows: dict[str, int]
    chunks_total: int
    chunks_embedded: int
    chunks_deleted: int
    chunks_unchanged: int
    full_rebuild: bool
    seconds: float
    by_doc_type: dict[str, int]


def collect_chunks(settings: Settings) -> list[Chunk]:
    chunks = load_documents(settings.docs_dir)
    chunks += schema_cards(settings.warehouse_path)
    chunks += ticket_note_chunks(settings.warehouse_path)
    # Dedupe on chunk_id: identical text in the same doc collapses to one row.
    return list({c.chunk_id: c for c in chunks}.values())


def run_ingest(settings: Settings, embedder: Embedder, full: bool = False) -> IngestStats:
    t0 = time.perf_counter()
    with tracer.start_as_current_span("ingest.run") as span:
        with tracer.start_as_current_span("ingest.structured"):
            rows = load_warehouse(settings.structured_dir, settings.warehouse_path)

        with tracer.start_as_current_span("ingest.chunk"):
            chunks = collect_chunks(settings)

        manifest = {}
        if settings.manifest_path.exists():
            manifest = json.loads(settings.manifest_path.read_text())
        store = VectorStore(settings.vector_dir, embedder)
        index_key = f"{embedder.name}|v{INDEX_VERSION}"
        if full or manifest.get("index") != index_key or not store.exists():
            full = True
            store.reset()
            manifest = {"index": index_key, "chunks": {}}

        known: dict[str, str] = manifest["chunks"]
        current = {c.chunk_id: c for c in chunks}
        to_embed = [c for cid, c in current.items() if known.get(cid) != c.content_hash]
        to_delete = [cid for cid in known if cid not in current]

        with tracer.start_as_current_span("ingest.embed") as es:
            es.set_attribute("ingest.chunks_to_embed", len(to_embed))
            vectors = embedder.embed_documents([c.embedding_text() for c in to_embed]) if to_embed else []
            if len(vectors) != len(to_embed):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(to_embed)} chunks")

        with tracer.start_as_current_span("ingest.index"):
            store.delete_ids(to_delete)
            store.upsert(to_embed, vectors)
            if to_embed or to_delete:
                store.build_indexes()

        manifest["chunks"] = {cid: c.content_hash for cid, c in current.items()}
        settings.manifest_path.write_text(json.dumps(manifest, indent=1))

        by_type: dict[str, int] = {}
        for c in chunks:
            by_type[c.doc_type] = by_type.get(c.doc_type, 0) + 1
        stats = IngestStats(
            warehouse_rows=rows,
            chunks_total=len(chunks),
            chunks_embedded=len(to_embed),
            chunks_deleted=len(to_delete),
            chunks_unchanged=len(chunks) - len(to_embed),
            full_rebuild=full,
            seconds=round(time.perf_counter() - t0, 2),
            by_doc_type=by_type,
        )
        for k, v in asdict(stats).items():
            if isinstance(v, (int, float, bool)):
                span.set_attribute(f"ingest.{k}", v)
        return stats
