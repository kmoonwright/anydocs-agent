"""Embedding providers.

The model name and dimension go into the snapshot manifest. Opening a snapshot embedded with a
different model is a loud error rather than a silently mixed vector space — see store.py.

The API-backed provider is LangChain's, so batching and retries are the library's problem rather
than ours; what stays here is the manifest contract (model, dim) and the EMBEDDING span.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

from anydocs import telemetry
from anydocs.config import get_settings


class EmbeddingProvider(Protocol):
    model: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class ApiEmbeddings:
    """`langchain_openai.OpenAIEmbeddings`, carrying the manifest fields alongside it."""

    def __init__(self, model: str, dim: int, *, api_key: str | None = None) -> None:
        from langchain_openai import OpenAIEmbeddings

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set, but config/models.yaml maps embeddings to openai."
            )
        self.model = model
        self.dim = dim
        # The key is passed explicitly rather than left to the library: a `None` here is read as
        # an async credential callable and disables the sync client entirely.
        self._inner = OpenAIEmbeddings(model=model, dimensions=dim, api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # An empty string is not embeddable; a chunk that cleaned down to nothing becomes a
        # space so the row still gets a vector and the node ids stay aligned.
        if not texts:
            return []
        return self._inner.embed_documents([t if t.strip() else " " for t in texts])


class FakeEmbeddings:
    """Deterministic hash vectors, so the test suite needs no API key.

    Not semantically meaningful — tests that assert on ranking use BM25 or exact text, never
    these vectors' similarity.
    """

    def __init__(self, model: str = "fake-embeddings", dim: int = 1536) -> None:
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode()).digest()
        raw = [((seed[i % len(seed)] * (i + 7)) % 251) / 251.0 - 0.5 for i in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]


class TracedEmbeddings:
    """Wraps any provider so the EMBEDDING span is emitted regardless of which one is in use.

    Without this the span would appear only for the API-backed provider, and a span that comes
    and goes with configuration is worse than no span at all.
    """

    def __init__(self, inner: EmbeddingProvider) -> None:
        self._inner = inner
        self.model = inner.model
        self.dim = inner.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with telemetry.embed_span(model=self.model, count=len(texts)):
            return self._inner.embed(texts)


_provider: EmbeddingProvider | None = None


def get_embeddings() -> EmbeddingProvider:
    """The configured provider. `ANYDOCS_EMBEDDINGS=fake` forces the test provider."""
    global _provider
    if _provider is not None:
        return _provider
    settings = get_settings()
    override = os.getenv("ANYDOCS_EMBEDDINGS", "").lower()
    provider_name = override or settings.embeddings.provider
    if provider_name == "fake":
        inner: EmbeddingProvider = FakeEmbeddings(dim=settings.embeddings.dim)
    elif provider_name == "openai":
        inner = ApiEmbeddings(settings.embeddings.model, settings.embeddings.dim)
    else:
        raise ValueError(
            f"Unknown embeddings provider {provider_name!r}; "
            f"wire it in anydocs/knowledge/embeddings.py."
        )
    _provider = TracedEmbeddings(inner)
    return _provider


def reset_embeddings() -> None:
    global _provider
    _provider = None

