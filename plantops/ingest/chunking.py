"""Structure-aware chunking for Markdown/text/PDF documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    doc_id: str
    source: str
    title: str
    section: str
    text: str
    doc_type: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        payload = "\x1f".join([self.title, self.section, self.text, repr(sorted(self.metadata.items()))])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def chunk_id(self) -> str:
        # Stable across runs as long as the doc and section text don't change.
        return f"{self.doc_id}#{hashlib.sha1((self.section + self.text).encode()).hexdigest()[:10]}"

    def embedding_text(self) -> str:
        # Prefixing title and section path lets short chunks (e.g. one table row) still
        # carry the context of where they came from.
        return f"{self.title}\n{self.section}\n\n{self.text}"


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---\n"):
        return {}, raw
    end = raw.find("\n---", 4)
    if end == -1:
        return {}, raw
    meta = {}
    for line in raw[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[end + 4 :].lstrip("\n")


def _blocks(body: str) -> list[str]:
    """Split into paragraphs, keeping Markdown tables and numbered lists as single blocks."""
    blocks, cur = [], []
    for line in body.splitlines():
        if not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _pack(blocks: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    chunks, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) + 2 > max_chars:
            chunks.append(cur)
            tail = cur[-overlap_chars:] if overlap_chars else ""
            # Start the overlap at a word boundary.
            cur = (tail[tail.find(" ") + 1 :] + "\n\n" if " " in tail else "") + b
        else:
            cur = f"{cur}\n\n{b}" if cur else b
    if cur:
        chunks.append(cur)
    # A single oversized block (huge table) is split on lines as a last resort.
    out = []
    for c in chunks:
        while len(c) > max_chars * 2:
            cut = c.rfind("\n", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(c[:cut])
            c = c[cut:].lstrip("\n")
        out.append(c)
    return out


def chunk_markdown(doc_id: str, source: str, raw: str, max_chars: int = 1200, overlap_chars: int = 150) -> list[Chunk]:
    meta, body = parse_front_matter(raw)
    title = meta.pop("title", doc_id)
    doc_type = meta.pop("doc_type", "document")

    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    lines: list[str] = []
    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if lines:
                sections.append((" > ".join(path), lines))
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            lines = []
        else:
            lines.append(line)
    if lines:
        sections.append((" > ".join(path), lines))

    chunks = []
    for section, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if not text:
            continue
        for piece in _pack(_blocks(text), max_chars, overlap_chars):
            chunks.append(Chunk(doc_id, source, title, section, piece, doc_type, dict(meta)))
    return chunks


def chunk_pdf(doc_id: str, path: Path, max_chars: int = 1200, overlap_chars: int = 150) -> list[Chunk]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else None) or doc_id
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for piece in _pack(_blocks(text), max_chars, overlap_chars):
            chunks.append(Chunk(doc_id, path.name, title, f"page {i}", piece, "pdf"))
    return chunks


def load_documents(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        doc_id = path.stem
        if path.suffix in (".md", ".txt"):
            chunks.extend(chunk_markdown(doc_id, path.name, path.read_text()))
        elif path.suffix == ".pdf":
            chunks.extend(chunk_pdf(doc_id, path))
    return chunks
