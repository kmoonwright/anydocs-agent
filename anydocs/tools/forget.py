"""Tombstone a digest: remove it from retrieval, keep the record that it existed.

The content goes and the provenance stays. "We fetched this page and discarded it" remains
answerable; what the page said does not.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.digest import Digest, utcnow
from anydocs.knowledge.provenance import ProvenanceRecord, record
from anydocs.knowledge.store import StoreError, find_doc
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register


class Input(BaseModel):
    doc_id: str
    reason: str = Field(description="Why this is being forgotten. Recorded in provenance.")
    corpus: str = ""
    snapshot_id: str = ""


class Output(BaseModel):
    doc_id: str = ""
    corpus: str = ""
    snapshot_id: str = ""
    tombstoned: bool = False
    provenance_id: str = ""
    rows_removed: int = 0
    already_tombstoned: bool = False


def run(inp: Input, ctx: ToolContext) -> Output:
    try:
        doc, snapshot = find_doc(inp.corpus, inp.snapshot_id, doc_id=inp.doc_id)
    except StoreError as exc:
        raise ToolError(str(exc)) from None

    if doc.tombstoned:
        return Output(
            doc_id=doc.doc_id,
            corpus=doc.corpus,
            snapshot_id=snapshot.snapshot_id,
            tombstoned=True,
            already_tombstoned=True,
        )

    # The digest, the chunks and the page text go. The keys and the hash stay, so provenance
    # still identifies what was removed without holding what it said.
    snapshot.write_doc(
        doc.model_copy(
            update={
                "tombstoned": True,
                "tombstone_reason": inp.reason,
                "digest": Digest(purpose=""),
                "chunks": [],
                "digest_quality": 0.0,
            }
        )
    )
    rows_removed = snapshot.unindex_doc(doc.doc_id)

    entry = record(
        snapshot,
        ProvenanceRecord.new(
            doc.doc_id,
            doc.canonical_url or doc.url,
            fetched_at=doc.fetched_at or utcnow(),
            content_hash=doc.content_hash,
            requested_by=ctx.requested_by,
            event="tombstoned",
            reason=inp.reason,
        ),
    )
    snapshot.write_manifest()

    telemetry.set_attributes(
        ctx.span,
        {
            telemetry.Attr.DOC_ID: doc.doc_id,
            telemetry.Attr.PROVENANCE_ID: entry.provenance_id,
            telemetry.Attr.SNAPSHOT_ID: snapshot.snapshot_id,
            "forget.reason": inp.reason,
            "forget.rows_removed": rows_removed,
        },
    )
    return Output(
        doc_id=doc.doc_id,
        corpus=doc.corpus,
        snapshot_id=snapshot.snapshot_id,
        tombstoned=True,
        provenance_id=entry.provenance_id,
        rows_removed=rows_removed,
    )


SPEC = register(
    ToolSpec(
        name="forget",
        description=(
            "Remove a doc from retrieval and record why. The page's content and digest are "
            "dropped; the provenance record is kept, so the fact that it was once fetched "
            "remains on file. Requires a reason, and is always logged."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="delete",
        span_kind="TOOL",
        cost_hint="low",
        policy_key="delete",
    )
)
