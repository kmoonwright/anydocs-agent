"""Reranking the fused candidate set.

The design doc's default was a local cross-encoder. That means torch, which contradicts the
API-embeddings choice, so the shipped default is an LLM pass on the cheap `router` role. The
interface keeps a local implementation one file away.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from anydocs import telemetry

RERANK_WINDOW = 30


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ...


class _Score(BaseModel):
    index: int = Field(description="0-based index of the candidate being scored")
    relevance: float = Field(description="0.0 (irrelevant) to 1.0 (directly answers the query)")


class _Scores(BaseModel):
    scores: list[_Score]


class LLMReranker:
    """A single cheap-model pass scoring every candidate in one call.

    One call rather than N: N calls would dominate hot-path latency, and the model scores more
    consistently when it sees the candidates side by side.
    """

    name = "llm-router"

    def __init__(self, role: str = "router") -> None:
        self.role = role

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from anydocs.models.roles import call_role_json, resolve_role

        if not candidates:
            return []
        window = candidates[:RERANK_WINDOW]
        listing = "\n\n".join(
            f"[{i}] {c.get('heading') or c.get('title') or c['node_id']}\n"
            f"{(c.get('text') or '')[:600]}"
            for i, c in enumerate(window)
        )
        model = resolve_role(self.role).model
        with telemetry.rerank_span(model=model, candidates=len(window)) as span:
            result = call_role_json(
                telemetry.RERANK,
                self.role,
                [
                    SystemMessage(
                        "You rank documentation passages. Score only on whether the passage "
                        "answers the question, not on how well written it is. Return one score "
                        "per passage."
                    ),
                    HumanMessage(
                        f"Question: {query}\n\n"
                        f"Score each passage for how directly it answers the question.\n\n"
                        f"{listing}"
                    ),
                ],
                _Scores,
                span=span,
            )
        by_index = {s.index: s.relevance for s in result.scores}
        for i, candidate in enumerate(window):
            candidate["rerank_score"] = float(by_index.get(i, 0.0))
        rest = candidates[RERANK_WINDOW:]
        for candidate in rest:
            candidate["rerank_score"] = 0.0
        window.sort(key=lambda c: c["rerank_score"], reverse=True)
        return window + rest


class NoopReranker:
    """Keeps fusion order. Used in tests and when the hot path must stay LLM-free."""

    name = "noop"

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, candidate in enumerate(candidates):
            # Preserve fusion order as a descending score so downstream spread maths still works.
            candidate["rerank_score"] = 1.0 / (1 + i)
        return candidates


def get_reranker() -> Reranker:
    """`ANYDOCS_RERANKER=noop` keeps the hot path free of an LLM call."""
    if os.getenv("ANYDOCS_RERANKER", "llm").lower() == "noop":
        return NoopReranker()
    return LLMReranker()
