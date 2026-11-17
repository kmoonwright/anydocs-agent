"""Everything one doc resolves to.

A citation points at a doc id; this is what that id expands into — the digest, the raw chunks,
the provenance chain, the related-doc links and the changelog. Assembled here because all three
doors return it: `GET /docs/{id}`, the `anydocs://docs/{id}` MCP resource, and `anydocs docs`.
Three copies of this payload would drift, and a citation that resolves differently depending on
which door you came through is worse than one that does not resolve at all.
"""

from __future__ import annotations

from typing import Any

from anydocs.knowledge.freshness import changelog_for
from anydocs.knowledge.links import links_for
from anydocs.knowledge.provenance import history
from anydocs.knowledge.store import open_active


def doc_detail(doc_id: str, *, include_chunks: bool = True) -> dict[str, Any] | None:
    """The full view of one doc, or None when no active snapshot holds it."""
    for snapshot in open_active():
        doc = snapshot.get_doc(doc_id)
        if doc is None:
            continue
        payload = doc.model_dump(mode="json")
        if not include_chunks:
            payload.pop("chunks", None)
        return {
            "doc": payload,
            "provenance": [record.model_dump() for record in history(snapshot, doc_id)],
            "links": [link.model_dump() for link in links_for(snapshot, [doc_id])],
            "changelog": changelog_for(snapshot, doc_id),
            "snapshot": snapshot.stats(),
        }
    return None
