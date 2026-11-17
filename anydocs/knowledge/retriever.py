"""Hybrid retrieval as a LangChain retriever.

`retrieval.search` is the implementation; this is its public interface. Two shapes, because two
callers want different things from the same search:

- `search()` returns the `RetrievalResult`, with every stage's candidates and scores. That is
  what `search_docs` publishes on the RETRIEVER span, and it is the whole reason the stages are
  computed — a retrieval eval has to be able to see *where* a good chunk was lost.
- `invoke()` returns `Document`s, so AnyDocs retrieval drops into any LangChain chain or agent
  built on `BaseRetriever` without knowing anything about snapshots.

LangChain's `EnsembleRetriever` over `BM25Retriever` would be the stock way to spell the hybrid
leg, and it was considered. It fuses internally and returns documents with no per-stage scores,
which would cost the `dense`/`sparse_bm25`/`rrf_fusion` stage logs, the digest down-weight, the
session-pin boost and the link-expansion stage. Those are the eval hooks, so the stock component
is the one wrapped here rather than the one used.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from anydocs.knowledge import retrieval
from anydocs.knowledge.retrieval import RetrievalResult


class AnyDocsRetriever(BaseRetriever):
    """Dense + BM25 over the active snapshots, fused with RRF, filtered, then reranked."""

    corpora: list[str] = []
    k: int = 8
    version: str = ""
    section: str = ""
    rerank: bool = True
    expand_links: bool = True
    gold_chunk_id: str = ""

    def search(self, queries: list[str]) -> RetrievalResult:
        """The staged search. Session pins are picked up from the turn, not passed in."""
        return retrieval.search(
            queries,
            corpora=self.corpora or None,
            k=self.k,
            version=self.version,
            section=self.section,
            rerank=self.rerank,
            expand_links=self.expand_links,
            gold_chunk_id=self.gold_chunk_id,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [as_document(hit) for hit in self.search([query]).hits]


def as_document(hit: dict[str, Any]) -> Document:
    """One hit as a LangChain `Document`.

    Everything a citation needs stays in `metadata`: a passage that cannot be attributed back to
    a doc id, a URL and a heading is not usable as evidence.
    """
    return Document(
        id=hit["node_id"],
        page_content=hit["text"],
        metadata={key: value for key, value in hit.items() if key != "text"},
    )
