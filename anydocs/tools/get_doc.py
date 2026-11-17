"""The full digested page plus its raw chunks — what the Auditor verifies against."""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.digest import Chunk, Digest
from anydocs.knowledge.provenance import history
from anydocs.knowledge.store import StoreError, find_doc
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register


class Input(BaseModel):
    doc_id: str = Field(default="", description="Doc id from a search hit or citation.")
    url: str = Field(default="", description="Canonical URL, if the doc id is unknown.")
    corpus: str = Field(default="", description="Narrows the lookup when a doc id is ambiguous.")
    snapshot_id: str = Field(default="", description="Read a specific snapshot, not the default.")
    include_chunks: bool = Field(default=True)


class Output(BaseModel):
    doc_id: str
    corpus: str
    url: str
    title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    version_markers: list[str] = Field(default_factory=list)
    content_hash: str = ""
    fetched_at: str = ""
    digest: Digest
    chunks: list[Chunk] = Field(default_factory=list)
    digest_quality: float = 0.0
    tombstoned: bool = False
    provenance: list[dict] = Field(default_factory=list)
    api_surface: list[str] = Field(default_factory=list)


def run(inp: Input, ctx: ToolContext) -> Output:
    try:
        doc, snapshot = find_doc(
            inp.corpus, inp.snapshot_id, doc_id=inp.doc_id, url=inp.url
        )
    except StoreError as exc:
        raise ToolError(str(exc)) from None

    telemetry.set_attributes(
        ctx.span,
        {telemetry.Attr.DOC_ID: doc.doc_id, telemetry.Attr.SNAPSHOT_ID: snapshot.snapshot_id},
    )
    return Output(
        doc_id=doc.doc_id,
        corpus=doc.corpus,
        url=doc.canonical_url or doc.url,
        title=doc.title,
        heading_path=doc.heading_path,
        version_markers=doc.version_markers,
        content_hash=doc.content_hash,
        fetched_at=doc.fetched_at,
        digest=doc.digest,
        chunks=doc.chunks if inp.include_chunks else [],
        digest_quality=doc.digest_quality,
        tombstoned=doc.tombstoned,
        provenance=[p.model_dump() for p in history(snapshot, doc.doc_id)],
        api_surface=sorted(doc.digest.api_surface()),
    )


SPEC = register(
    ToolSpec(
        name="get_doc",
        description=(
            "Fetch one digested page in full: its structured digest, every raw chunk, the "
            "versions it applies to and its provenance. Use this to verify a claim against the "
            "page it was cited from, rather than trusting a search snippet."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="read",
        span_kind="TOOL",
        cost_hint="low",
    )
)
