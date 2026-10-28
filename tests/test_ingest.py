import shutil
from dataclasses import replace

from plantops.ingest.chunking import chunk_markdown, parse_front_matter
from plantops.ingest.embeddings import HashEmbedder
from plantops.ingest.pipeline import run_ingest

DOC = """---
doc_type: sop
title: Test SOP
equipment_model: HX-500
---

# Test SOP

Intro paragraph.

## Steps

| Step | Action |
|------|--------|
| 1 | Prepare |
| 2 | Verify |

## Notes

Some notes here.
"""


def test_front_matter_and_heading_paths():
    meta, body = parse_front_matter(DOC)
    assert meta["doc_type"] == "sop" and body.startswith("# Test SOP")
    chunks = chunk_markdown("test_sop", "test_sop.md", DOC)
    assert [c.section for c in chunks] == ["Test SOP", "Test SOP > Steps", "Test SOP > Notes"]
    assert chunks[0].doc_type == "sop" and chunks[0].metadata == {"equipment_model": "HX-500"}
    # The table stays in one chunk with its header row.
    assert "| Step | Action |" in chunks[1].text and "| 2 | Verify |" in chunks[1].text


def test_chunk_ids_are_stable_and_content_sensitive():
    a = chunk_markdown("d", "d.md", DOC)
    b = chunk_markdown("d", "d.md", DOC)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    c = chunk_markdown("d", "d.md", DOC.replace("Some notes", "Other notes"))
    assert a[2].chunk_id != c[2].chunk_id and a[0].chunk_id == c[0].chunk_id


def test_oversized_sections_are_split_with_overlap():
    body = "# Big\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
    chunks = chunk_markdown("big", "big.md", body, max_chars=800, overlap_chars=100)
    assert len(chunks) > 5
    assert all(len(c.text) <= 1700 for c in chunks)


def test_hash_embedder_is_deterministic_and_normalized():
    e = HashEmbedder(64)
    v1, v2 = e.embed_query("E-42 seal leak"), e.embed_query("E-42 seal leak")
    assert v1 == v2 and abs(sum(x * x for x in v1) - 1) < 1e-9


def test_incremental_ingest(settings, tmp_path):
    docs = tmp_path / "docs"
    shutil.copytree(settings.docs_dir, docs)
    s = replace(settings, docs_dir=docs, var_dir=tmp_path / "var")
    s.var_dir.mkdir()
    emb = HashEmbedder(settings.embed_dim)

    first = run_ingest(s, emb)
    assert first.full_rebuild and first.chunks_embedded == first.chunks_total
    assert first.by_doc_type["schema"] == 11 and first.by_doc_type["maintenance_ticket"] > 100

    again = run_ingest(s, emb)
    assert not again.full_rebuild and again.chunks_embedded == 0 and again.chunks_deleted == 0

    p = docs / "spare_parts_policy.md"
    p.write_text(p.read_text().replace("Up to $5,000", "Up to $6,000"))
    (docs / "cv200_conveyor_manual.md").unlink()
    changed = run_ingest(s, emb)
    assert changed.chunks_embedded == 1
    assert changed.chunks_deleted >= 2
