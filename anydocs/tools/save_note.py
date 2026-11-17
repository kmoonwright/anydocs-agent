"""Write a digest into a corpus snapshot, with a provenance record.

This is where knowledge becomes durable. Everything needed to reproduce the answer later goes in
with it: where the page came from, when, under what robots decision, and which model and prompt
version produced the digest.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.guardrails import scan_secrets
from anydocs.knowledge.digest import Digest, Doc, chunk_markdown, content_hash, doc_id_for, utcnow
from anydocs.knowledge.links import detect_contradictions
from anydocs.knowledge.provenance import ProvenanceRecord, record
from anydocs.knowledge.store import create_snapshot, open_snapshot, publish_snapshot
from anydocs.tools._raw_pages import crawled_page
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register


class Input(BaseModel):
    corpus: str = Field(description="Corpus to write into; created if it does not exist.")
    url: str = Field(default="", description="Canonical URL of the page.")
    markdown: str = Field(
        default="",
        description="Cleaned page content, chunked for citation fidelity. Omit when passing a "
        "page_path.",
    )
    page_path: str = Field(
        default="",
        description="A `path` from a crawl_site result. Fills the page content, URL, title, "
        "version markers and HTTP metadata from what the crawl recorded.",
    )
    digest: Digest = Field(description="The structured digest from digest_page.")
    title: str = ""
    snapshot_id: str = Field(default="", description="Write into this snapshot; new if empty.")
    version_markers: list[str] = Field(default_factory=list)
    fetched_at: str = ""
    http_status: int = 0
    content_type: str = ""
    last_modified: str = ""
    robots_decision: str = ""
    digest_model: str = Field(
        default="", description="The `model` from the digest_page result that produced the digest."
    )
    digest_prompt_version: str = Field(
        default="", description="The `prompt_version` from that same digest_page result."
    )
    digest_quality: float = 0.0
    publish: bool = Field(
        default=False, description="Make this snapshot the corpus default after writing."
    )
    check_contradictions: bool = Field(
        default=False,
        description="Compare this page against overlapping ones and record what disagrees. Off "
        "by default: the ingest pipeline runs that check as a bounded pass over everything it "
        "just wrote, so doing it here as well would duplicate the work.",
    )


class Output(BaseModel):
    doc_id: str
    snapshot_id: str
    corpus: str
    provenance_id: str
    chunks: int
    indexed_nodes: int = 0
    contradictions: list[str] = Field(
        default_factory=list, description="Doc ids this page was found to disagree with."
    )
    duplicate: bool = False
    redactions: list[str] = Field(default_factory=list)
    provenance_incomplete: list[str] = Field(
        default_factory=list,
        description="Provenance fields the caller left blank. Pass them from the digest_page "
        "result so the record is complete.",
    )


def run(inp: Input, ctx: ToolContext) -> Output:
    fields = _resolve_page(inp)

    # Secrets scan before anything durable is written: a page carrying an API key is digested
    # with those spans redacted, and the provenance record says so.
    cleaned, redactions = scan_secrets(fields["markdown"])

    snapshot = (
        open_snapshot(inp.corpus, inp.snapshot_id)
        if inp.snapshot_id
        else create_snapshot(inp.corpus, source=inp.url)
    )

    url = fields["url"]
    doc_id = doc_id_for(inp.corpus, url)
    new_hash = content_hash(cleaned)
    existing = snapshot.get_doc(doc_id)
    duplicate = bool(existing and existing.content_hash == new_hash and not existing.tombstoned)

    if duplicate:
        telemetry.set_attributes(
            ctx.span,
            {telemetry.Attr.DOC_ID: doc_id, telemetry.Attr.SNAPSHOT_ID: snapshot.snapshot_id},
        )
        return Output(
            doc_id=doc_id,
            snapshot_id=snapshot.snapshot_id,
            corpus=inp.corpus,
            provenance_id="",
            chunks=len(existing.chunks) if existing else 0,
            duplicate=True,
            redactions=redactions,
        )

    doc, indexed, entry = _write_doc(
        snapshot,
        inp,
        ctx,
        doc_id=doc_id,
        url=url,
        cleaned=cleaned,
        new_hash=new_hash,
        fields=fields,
        redactions=redactions,
        replacing=existing is not None,
    )

    # Conflicts are found here, once, rather than re-judged on every answer that cites the page.
    contradictions: list[str] = []
    if inp.check_contradictions:
        try:
            contradictions = [
                link.to_doc_id for link in detect_contradictions(snapshot, doc_id)
            ]
        except Exception as exc:
            # A failed comparison must not lose the page that was already written.
            telemetry.set_attributes(
                ctx.span, {"contradiction_check.failed": f"{type(exc).__name__}: {exc}"}
            )

    if inp.publish:
        publish_snapshot(inp.corpus, snapshot.snapshot_id)

    # Every answer must be traceable to a page fetched at a time by a named model under a named
    # prompt. A caller that omits those leaves a hole in that chain, so the hole is on the span
    # rather than only in the record.
    missing = [
        name
        for name, value in (
            ("digest_model", inp.digest_model),
            ("digest_prompt_version", inp.digest_prompt_version),
        )
        if not value
    ]
    telemetry.set_attributes(
        ctx.span,
        {
            telemetry.Attr.DOC_ID: doc_id,
            telemetry.Attr.PROVENANCE_ID: entry.provenance_id,
            telemetry.Attr.SNAPSHOT_ID: snapshot.snapshot_id,
            "provenance.incomplete": missing,
            "contradictions.found": contradictions,
        },
    )
    return Output(
        doc_id=doc_id,
        snapshot_id=snapshot.snapshot_id,
        corpus=inp.corpus,
        provenance_id=entry.provenance_id,
        chunks=len(doc.chunks),
        indexed_nodes=indexed,
        contradictions=contradictions,
        duplicate=False,
        redactions=redactions,
        provenance_incomplete=missing,
    )


def _write_doc(
    snapshot,
    inp: Input,
    ctx: ToolContext,
    *,
    doc_id: str,
    url: str,
    cleaned: str,
    new_hash: str,
    fields: dict,
    redactions: list[str],
    replacing: bool,
):
    """Write the doc, index it, and append its provenance record.

    Returns (doc, indexed_node_count, provenance_record). Indexing happens here rather than
    being left to a later `reindex`, so a snapshot the agent just wrote is queryable on the next
    search instead of looking like a retrieval bug.
    """
    doc = Doc(
        doc_id=doc_id,
        corpus=inp.corpus,
        url=url,
        canonical_url=url,
        title=fields["title"],
        heading_path=[fields["title"]] if fields["title"] else [],
        version_markers=fields["version_markers"] or inp.digest.versions.applies_to,
        content_hash=new_hash,
        fetched_at=fields["fetched_at"],
        digest=inp.digest,
        chunks=chunk_markdown(cleaned, doc_id),
        digest_quality=inp.digest_quality or inp.digest.quality_score(),
    )
    snapshot.write_doc(doc)
    indexed = snapshot.index_doc(doc)

    entry = record(
        snapshot,
        ProvenanceRecord.new(
            doc_id,
            url,
            fetched_at=fields["fetched_at"],
            http_status=fields["http_status"],
            content_type=fields["content_type"],
            last_modified=fields["last_modified"],
            content_hash=new_hash,
            robots_decision=fields["robots_decision"],
            digest_model=inp.digest_model,
            digest_prompt_version=inp.digest_prompt_version,
            requested_by=ctx.requested_by,
            redactions=redactions,
            event="redigested" if replacing else "saved",
        ),
    )
    snapshot.write_manifest()
    return doc, indexed, entry


def _resolve_page(inp: Input) -> dict:
    """Page content and metadata, from inline arguments or from what the crawl recorded."""
    record = crawled_page(inp.page_path) if inp.page_path else {}

    markdown = inp.markdown or record.get("markdown", "")
    url = inp.url or record.get("url", "")
    if not markdown.strip():
        raise ToolError("save_note needs either `markdown` or a `page_path` from a crawl")
    if not url:
        raise ToolError("save_note needs a `url`, either passed in or from the crawled page")

    return {
        "markdown": markdown,
        "url": url,
        "title": inp.title or record.get("title", ""),
        "version_markers": inp.version_markers or record.get("version_markers", []),
        "fetched_at": inp.fetched_at or record.get("fetched_at") or utcnow(),
        "http_status": inp.http_status or int(record.get("status", 0) or 0),
        "content_type": inp.content_type or record.get("content_type", ""),
        "last_modified": inp.last_modified or record.get("last_modified", ""),
        "robots_decision": inp.robots_decision or record.get("robots_decision", ""),
    }


SPEC = register(
    ToolSpec(
        name="save_note",
        description=(
            "Write a digest into a corpus snapshot with a provenance record, chunking the page "
            "for citation. Skips pages whose content has not changed. Redacts credentials found "
            "in the page before saving and notes the redaction in provenance."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="write",
        span_kind="TOOL",
        cost_hint="medium",
        policy_key="write",
    )
)
