"""Hybrid retrieval: dense and BM25 in parallel, fused with RRF, filtered, then reranked.

Every stage's candidates and scores are recorded so a retrieval eval can see *where* a good
chunk was lost — dropped by the dense leg, outranked at fusion, or demoted by the reranker.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from openinference.instrumentation import get_retriever_attributes
from opentelemetry.trace import Span

from anydocs import telemetry
from anydocs.knowledge.embeddings import get_embeddings
from anydocs.knowledge.rerank import get_reranker
from anydocs.knowledge.store import Snapshot, open_active

RRF_K = 60
DENSE_K = 40
SPARSE_K = 40
RERANK_INPUT = 30
DIGEST_WEIGHT = 0.85  # digests summarize; a matching raw chunk is the better citation
PIN_BOOST = 1.35  # session-cited docs: enough to win a close call, not enough to force one
EXPANSION_FROM_TOP = 5  # how many ranked docs to follow links from
EXPANSION_DISCOUNT = 0.5  # linked-in nodes enter below everything that ranked on its own


@dataclass
class StageLog:
    """One retrieval stage's output, as the span records it."""

    stage: str
    candidates: list[tuple[str, float]] = field(default_factory=list)

    def as_attribute(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "count": len(self.candidates),
            "top": [
                {"node_id": nid, "score": round(score, 4)} for nid, score in self.candidates[:10]
            ],
        }


@dataclass
class RetrievalResult:
    hits: list[dict[str, Any]]
    stages: list[StageLog]
    queries: list[str]
    score_spread: float
    gold_rank: int = -1

    def stage_attribute(self) -> list[dict[str, Any]]:
        return [s.as_attribute() for s in self.stages]


# Docs already cited this session. Carried on a contextvar rather than as a tool argument: the
# model should not have to remember to pass its own conversation history into every search, and
# a pin it forgot would silently change which page a follow-up lands on.
_session_pins: ContextVar[tuple[str, ...]] = ContextVar("anydocs_session_pins", default=())


def set_session_pins(doc_ids: list[str] | None) -> None:
    _session_pins.set(tuple(doc_ids or ()))


def current_session_pins() -> list[str]:
    return list(_session_pins.get())


_WORD = re.compile(r"[A-Za-z0-9_.]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def _dense(snapshot: Snapshot, query_vectors: list[list[float]], k: int) -> list[tuple[str, float]]:
    """Nearest neighbours across every rewritten query, best distance per node.

    Distance, not similarity: smaller is better, and RRF only cares about the ordering.
    """
    scored: dict[str, float] = {}
    for vector in query_vectors:
        for node_id, distance in snapshot.search_vectors(vector, k):
            if node_id not in scored or distance < scored[node_id]:
                scored[node_id] = distance
    return sorted(scored.items(), key=lambda kv: kv[1])[:k]


def _sparse(nodes: list[dict[str, Any]], queries: list[str], k: int) -> list[tuple[str, float]]:
    """BM25 in-process over the snapshot's indexed rows.

    Held in memory, which is fine at snapshot sizes and keeps a snapshot a single directory with
    no second index server to run.
    """
    if not nodes:
        return []

    corpus = [_tokenize(f"{n.get('heading', '')} {n.get('title', '')} {n['text']}") for n in nodes]
    score_query = _bm25(corpus)
    scored: dict[str, float] = {}
    for query in queries:
        for node, score in zip(nodes, score_query(_tokenize(query)), strict=True):
            node_id = node["node_id"]
            scored[node_id] = max(scored.get(node_id, 0.0), float(score))
    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    return [(nid, score) for nid, score in ranked if score > 0][:k]


def _bm25(corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
    """`rank_bm25.BM25Okapi`, term for term, returning its `get_scores`.

    A copy rather than the package because the package's only dependency is numpy — 58 MB of a
    serverless function, for ten lines of arithmetic. The scores are the ones logged on the
    `sparse_bm25` and `rrf_fusion` stages, so `tests/unit/test_bm25_parity.py` pins them to the
    original rather than trusting the transcription.
    """
    doc_freqs = [Counter(doc) for doc in corpus]
    doc_len = [len(doc) for doc in corpus]
    avgdl = sum(doc_len) / len(corpus)
    containing = Counter(word for freqs in doc_freqs for word in freqs)

    # Okapi idf, floored at a fraction of the average for words in over half the documents,
    # whose raw idf is negative.
    idf = {
        word: math.log(len(corpus) - n + 0.5) - math.log(n + 0.5)
        for word, n in containing.items()
    }
    floor = epsilon * (sum(idf.values()) / len(idf)) if idf else 0.0
    idf = {word: value if value >= 0 else floor for word, value in idf.items()}

    def get_scores(query: list[str]) -> list[float]:
        scores = [0.0] * len(corpus)
        if not avgdl:
            return scores
        for term in query:
            weight = idf.get(term) or 0
            for i, freqs in enumerate(doc_freqs):
                tf = freqs.get(term) or 0
                scores[i] += weight * (
                    tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len[i] / avgdl))
                )
        return scores

    return get_scores


def _rrf(
    ranked_lists: list[list[tuple[str, float]]], weights: dict[str, float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Rank-based, so the dense leg's distances and BM25's unbounded
    scores combine without either needing normalization."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (node_id, _score) in enumerate(ranked):
            fused[node_id] = fused.get(node_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    if weights:
        for node_id in fused:
            fused[node_id] *= weights.get(node_id, 1.0)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def _matches_filters(
    node: dict[str, Any], *, version: str = "", section: str = "", kinds: tuple[str, ...] = ()
) -> bool:
    if kinds and node.get("kind") not in kinds:
        return False
    if version and version not in (node.get("versions") or ""):
        return False
    if section and section.lower() not in (node.get("heading") or "").lower():
        return False
    return True


def search(
    queries: list[str],
    *,
    corpora: list[str] | None = None,
    k: int = 8,
    version: str = "",
    section: str = "",
    snapshots: list[Snapshot] | None = None,
    gold_chunk_id: str = "",
    rerank: bool = True,
    expand_links: bool = True,
    pinned_doc_ids: list[str] | None = None,
) -> RetrievalResult:
    """Hybrid search across the active snapshots.

    `gold_chunk_id`, when a dataset example provides one, makes recall@k measurable directly
    instead of inferred from the answer.

    `pinned_doc_ids` are docs already cited this session. They are boosted, not forced: a
    follow-up like "and how do I configure that?" should land on the same page, but a genuinely
    new question must still be able to leave it.
    """
    # The span starts here and ends when this block exits, including on a raise, so its
    # duration is the search and not the tool wrapper around it.
    queries = [q for q in queries if q.strip()] or [""]
    with telemetry.retrieval_span() as span:
        # `input.value` is a string. A string array is a legal OTel attribute and
        # Arize drops it on ingest, so the query has to be JSON text.
        telemetry.set_attributes(span, {telemetry.Attr.INPUT_VALUE: json.dumps(queries)})
        result = _search(
            queries,
            corpora=corpora,
            k=k,
            version=version,
            section=section,
            snapshots=snapshots,
            gold_chunk_id=gold_chunk_id,
            rerank=rerank,
            expand_links=expand_links,
            pinned_doc_ids=pinned_doc_ids,
        )
        _record_retriever_output(span, result)
        return result


def _search(
    queries: list[str],
    *,
    corpora: list[str] | None = None,
    k: int = 8,
    version: str = "",
    section: str = "",
    snapshots: list[Snapshot] | None = None,
    gold_chunk_id: str = "",
    rerank: bool = True,
    expand_links: bool = True,
    pinned_doc_ids: list[str] | None = None,
) -> RetrievalResult:
    queries = [q for q in queries if q.strip()] or [""]
    if pinned_doc_ids is None:
        pinned_doc_ids = current_session_pins()
    snaps = snapshots if snapshots is not None else open_active(corpora)
    stages: list[StageLog] = []

    nodes: list[dict[str, Any]] = []
    for snap in snaps:
        for node in snap.all_nodes():
            node["snapshot_id"] = snap.snapshot_id
            nodes.append(node)
    nodes = [n for n in nodes if _matches_filters(n, version=version, section=section)]
    stages.append(StageLog("filtered_pool", [(n["node_id"], 0.0) for n in nodes]))
    if not nodes:
        return RetrievalResult(hits=[], stages=stages, queries=queries, score_spread=0.0)

    by_id = {n["node_id"]: n for n in nodes}
    query_vectors = get_embeddings().embed(queries)

    # Dense and sparse run in parallel: both are independent reads over the same snapshot.
    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_future = pool.submit(
            lambda: [hit for snap in snaps for hit in _dense(snap, query_vectors, DENSE_K)]
        )
        sparse_future = pool.submit(_sparse, nodes, queries, SPARSE_K)
        dense_hits = [h for h in dense_future.result() if h[0] in by_id]
        sparse_hits = sparse_future.result()

    dense_hits.sort(key=lambda kv: kv[1])
    stages.append(StageLog("dense", dense_hits))
    stages.append(StageLog("sparse_bm25", sparse_hits))

    weights = {nid: (DIGEST_WEIGHT if by_id[nid].get("kind") == "digest" else 1.0) for nid in by_id}
    fused = _rrf([dense_hits, sparse_hits], weights)
    stages.append(StageLog("rrf_fusion", fused))

    if pinned_doc_ids:
        fused = _boost_pinned(fused, by_id, pinned_doc_ids)
        stages.append(StageLog("session_pins", fused))

    if expand_links:
        fused, added = _expand_by_links(fused, by_id, snaps)
        if added:
            # Recorded as its own stage so the log shows what expansion contributed, rather
            # than a chunk appearing in the results with no explanation.
            stages.append(StageLog("link_expansion", fused))

    candidates = []
    for node_id, score in fused[:RERANK_INPUT]:
        node = dict(by_id[node_id])
        node["fusion_score"] = score
        candidates.append(node)

    if rerank and candidates:
        candidates = get_reranker().rerank(queries[0], candidates)
        stages.append(
            StageLog("rerank", [(c["node_id"], c.get("rerank_score", 0.0)) for c in candidates])
        )
        score_key = "rerank_score"
    else:
        score_key = "fusion_score"

    hits = candidates[:k]
    for hit in hits:
        hit["score"] = float(hit.get(score_key, 0.0))

    gold_rank = -1
    if gold_chunk_id:
        ordered = [c["node_id"] for c in candidates]
        gold_rank = ordered.index(gold_chunk_id) + 1 if gold_chunk_id in ordered else -1

    return RetrievalResult(
        hits=hits,
        stages=stages,
        queries=queries,
        score_spread=_spread([h["score"] for h in hits]),
        gold_rank=gold_rank,
    )


def _record_retriever_output(span: Span, result: RetrievalResult) -> None:
    """Input, output and per-document metadata on the retriever span.

    The `search_docs` tool span keeps its own copy of the stages and the tool result. This
    records the same facts on `retrieval.run`, in the flattened document shape a retriever
    eval reads, so the new span does not replace that instrumentation.
    """
    documents = [
        {
            "id": hit.get("node_id", ""),
            "content": hit.get("text", ""),
            "score": float(hit.get("score", 0.0)),
            "metadata": {
                "doc_id": hit.get("doc_id", ""),
                "url": hit.get("url", ""),
                "title": hit.get("title", ""),
                "heading": hit.get("heading", ""),
                "corpus": hit.get("corpus", ""),
                "kind": hit.get("kind", ""),
                "snapshot_id": hit.get("snapshot_id", ""),
            },
        }
        for hit in result.hits
    ]
    telemetry.set_attributes(
        span,
        {
            telemetry.Attr.OUTPUT_VALUE: {
                "queries": result.queries,
                "score_spread": result.score_spread,
                "gold_rank": result.gold_rank,
                "hits": [
                    {
                        "node_id": hit.get("node_id", ""),
                        "doc_id": hit.get("doc_id", ""),
                        "score": hit.get("score", 0.0),
                        "url": hit.get("url", ""),
                        "heading": hit.get("heading", ""),
                    }
                    for hit in result.hits
                ],
            },
            telemetry.Attr.RETRIEVAL_QUERIES: result.queries,
            telemetry.Attr.RETRIEVAL_STAGES: result.stage_attribute(),
            telemetry.Attr.RETRIEVAL_SCORE_SPREAD: result.score_spread,
            telemetry.Attr.RETRIEVAL_GOLD_RANK: result.gold_rank,
            **get_retriever_attributes(documents=documents),
        },
    )


def _boost_pinned(
    fused: list[tuple[str, float]],
    by_id: dict[str, dict[str, Any]],
    pinned_doc_ids: list[str],
) -> list[tuple[str, float]]:
    """Nudge nodes from already-cited docs up the list.

    A multiplier rather than a pin: the boost has to be big enough to win a close call and small
    enough to lose to a clearly better match for a new question.
    """
    pinned = set(pinned_doc_ids)
    boosted = [
        (node_id, score * (PIN_BOOST if by_id[node_id].get("doc_id") in pinned else 1.0))
        for node_id, score in fused
        if node_id in by_id
    ]
    return sorted(boosted, key=lambda kv: kv[1], reverse=True)


def _expand_by_links(
    fused: list[tuple[str, float]],
    by_id: dict[str, dict[str, Any]],
    snapshots: list[Snapshot],
) -> tuple[list[tuple[str, float]], int]:
    """Pull in nodes one `see-also` or `supersedes` hop from what already ranked.

    Expansion enters below everything that ranked on its own merits: a linked page is a lead,
    not evidence, and it should only be read when the direct hits run out.
    """
    from anydocs.knowledge.links import expand

    if not fused:
        return fused, 0
    top_docs = list(
        dict.fromkeys(by_id[node_id]["doc_id"] for node_id, _ in fused[:EXPANSION_FROM_TOP])
    )
    reachable: set[str] = set()
    for snapshot in snapshots:
        reachable.update(expand(snapshot, top_docs))
    if not reachable:
        return fused, 0

    present = {node_id for node_id, _ in fused}
    floor = fused[-1][1]
    additions = [
        (node_id, floor * EXPANSION_DISCOUNT)
        for node_id, node in by_id.items()
        if node.get("doc_id") in reachable and node_id not in present
    ]
    return fused + additions, len(additions)


def _spread(scores: list[float]) -> float:
    """Gap between the best hit and the median of the rest.

    A wide spread means the reranker found something clearly better than the pack; a flat one
    means everything looked equally mediocre, which is a retrieval-confidence signal.
    """
    if len(scores) < 2:
        return 0.0
    best = scores[0]
    rest = sorted(scores[1:])
    median = rest[len(rest) // 2]
    return round(max(0.0, best - median), 4)
