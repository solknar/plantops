"""Embedding backends: Gemini (real) and a deterministic hashing embedder (offline/tests)."""

from __future__ import annotations

import hashlib
import math
import re
import time
from functools import lru_cache
from typing import Protocol

from opentelemetry import trace

from plantops.config import Settings

tracer = trace.get_tracer("plantops.embeddings")

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashEmbedder:
    """Signed feature hashing over unigrams + bigrams.

    Not semantic, but deterministic, free and good at the exact-token matching that
    matters in this domain (error codes like E-42, part numbers like P-1001). Lets the
    whole pipeline and test suite run without network access.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _embed(self, text: str) -> list[float]:
        toks = _TOKEN_RE.findall(text.lower())
        feats = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _normalize(v)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class GeminiEmbedder:
    """Gemini embeddings with batching and backoff for free-tier rate limits."""

    def __init__(self, model: str, dim: int = 768, batch_size: int = 50, max_retries: int = 6):
        from google import genai

        self._client = genai.Client()
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.name = f"{model}-{dim}"

    def _call(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import errors, types

        cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                with tracer.start_as_current_span("embed.batch") as span:
                    span.set_attribute("embed.model", self.model)
                    span.set_attribute("embed.batch_size", len(texts))
                    span.set_attribute("embed.chars", sum(len(t) for t in texts))
                    # One Content per text: multimodal models (gemini-embedding-2) treat a list of
                    # plain strings as parts of a single input and return one embedding.
                    contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
                    resp = self._client.models.embed_content(model=self.model, contents=contents, config=cfg)
                if len(resp.embeddings) != len(texts):
                    raise RuntimeError(f"{self.model} returned {len(resp.embeddings)} embeddings for {len(texts)} texts")
                # Truncated outputs (dim < native) are not unit length; renormalize for cosine.
                return [_normalize(list(e.values)) for e in resp.embeddings]
            except errors.APIError as e:
                if getattr(e, "code", None) not in (429, 500, 503) or attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._call(texts[i : i + self.batch_size], "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return list(self._cached_query(text))

    @lru_cache(maxsize=512)
    def _cached_query(self, text: str) -> tuple[float, ...]:
        return tuple(self._call([text], "RETRIEVAL_QUERY")[0])


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hash":
        return HashEmbedder(settings.embed_dim)
    return GeminiEmbedder(settings.embed_model, settings.embed_dim)
