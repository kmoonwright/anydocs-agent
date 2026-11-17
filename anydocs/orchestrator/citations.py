"""Resolving citations against the store.

A citation comes back from a model, which means its URL, title and heading are *claims* about
the store rather than facts from it. The doc id is the only part worth trusting, because the
model read it off a search hit — so everything else is looked up, and a doc id that does not
exist is reported rather than rendered as a source.

This is also the hallucinated-citation check: a model that invents a plausible doc id gets
caught here instead of in front of a user.
"""

from __future__ import annotations

from anydocs.knowledge.store import open_active
from anydocs.specialists.results import Citation


def resolve(
    citations: list[Citation], *, corpora: list[str] | None = None
) -> tuple[list[Citation], list[Citation]]:
    """Return (resolved, unresolvable).

    A resolved citation's corpus, URL, title, heading and snapshot id come from the store. The
    model's `quote` is kept as-is, since that is its claim about what the source says and the
    Auditor checks it.
    """
    if not citations:
        return [], []

    index = {}
    for snapshot in open_active(corpora):
        for doc in snapshot.live_docs():
            index[doc.doc_id] = (doc, snapshot.snapshot_id)

    resolved: list[Citation] = []
    unresolvable: list[Citation] = []
    for citation in citations:
        entry = index.get(citation.doc_id)
        if entry is None:
            unresolvable.append(citation)
            continue
        doc, snapshot_id = entry
        heading = _heading_for(doc, citation)
        resolved.append(
            citation.model_copy(
                update={
                    "corpus": doc.corpus,
                    "url": doc.canonical_url or doc.url,
                    "title": doc.title or doc.doc_id,
                    "heading": heading,
                    "snapshot_id": snapshot_id,
                }
            )
        )
    return resolved, unresolvable


def _heading_for(doc, citation: Citation) -> str:
    """The heading from the store, preferring the chunk the citation names.

    Falls back to matching the model's quote against the page's chunks, then to the model's own
    heading — which at least came from a search hit — then to the page title.
    """
    if citation.node_id:
        for chunk in doc.chunks:
            if chunk.chunk_id == citation.node_id:
                return chunk.heading
    if citation.quote:
        needle = citation.quote.strip()[:80]
        for chunk in doc.chunks:
            if needle and needle in chunk.text:
                return chunk.heading
    return citation.heading or doc.heading
