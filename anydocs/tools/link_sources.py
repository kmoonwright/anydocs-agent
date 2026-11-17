"""Record how two docs relate: supersedes, see-also, contradicts, example-of.

Links do two jobs. `see-also` and `supersedes` expand retrieval by one hop, so a question that
lands on the wrong page of a doc set can still reach the right one. `contradicts` is how an
answer comes to say "sources disagree" rather than silently picking a side.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.links import LINK_KINDS, Link, record
from anydocs.knowledge.store import resolve_snapshots
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register


class Input(BaseModel):
    from_doc_id: str
    to_doc_id: str
    kind: str = Field(description=f"One of: {', '.join(LINK_KINDS)}")
    evidence: str = Field(default="", description="Why this edge exists, quoted if possible.")
    corpus: str = ""
    snapshot_id: str = ""


class Output(BaseModel):
    from_doc_id: str = ""
    to_doc_id: str = ""
    kind: str = ""
    recorded: bool = False
    already_present: bool = False
    evidence: str = ""


def run(inp: Input, ctx: ToolContext) -> Output:
    if inp.kind not in LINK_KINDS:
        raise ToolError(f"kind must be one of {', '.join(LINK_KINDS)}; got {inp.kind!r}")
    if inp.from_doc_id == inp.to_doc_id:
        raise ToolError("a doc cannot link to itself")

    snapshots = resolve_snapshots(inp.corpus, inp.snapshot_id)
    # Both endpoints must exist. An edge to a doc id a model invented would expand retrieval
    # toward nothing and show a reader a conflict with a page that does not exist.
    target = None
    for snapshot in snapshots:
        if snapshot.get_doc(inp.from_doc_id) and snapshot.get_doc(inp.to_doc_id):
            target = snapshot
            break
    if target is None:
        raise ToolError(
            f"both {inp.from_doc_id!r} and {inp.to_doc_id!r} must be in one snapshot; "
            f"a link cannot cross corpora or name a doc that is not in the store"
        )

    before = len(target.read_jsonl(target.links_path))
    link = record(
        target,
        Link(
            from_doc_id=inp.from_doc_id,
            to_doc_id=inp.to_doc_id,
            kind=inp.kind,
            evidence=inp.evidence,
            recorded_by=ctx.requested_by,
        ),
    )
    after = len(target.read_jsonl(target.links_path))

    telemetry.set_attributes(
        ctx.span,
        {
            "link.kind": inp.kind,
            "link.from": inp.from_doc_id,
            "link.to": inp.to_doc_id,
            telemetry.Attr.SNAPSHOT_ID: target.snapshot_id,
        },
    )
    return Output(
        from_doc_id=link.from_doc_id,
        to_doc_id=link.to_doc_id,
        kind=link.kind,
        recorded=True,
        already_present=after == before,
        evidence=link.evidence,
    )


SPEC = register(
    ToolSpec(
        name="link_sources",
        description=(
            "Record a relationship between two docs — supersedes, see-also, contradicts or "
            "example-of — with the evidence for it. See-also and supersedes expand retrieval by "
            "one hop; contradictions surface in answers as 'sources disagree'. Both docs must "
            "already be in the store."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="write",
        span_kind="TOOL",
        cost_hint="low",
        policy_key="write",
    )
)
