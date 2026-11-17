"""Add or update an entry in the freshness schedule.

The cadence policy itself lives in `anydocs/knowledge/freshness.py`: double on unchanged, reset
on changed, capped. This tool only writes the entry the Archivist's scheduler works through.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.freshness import (
    DEFAULT_CADENCE_DAYS,
    MAX_CADENCE_DAYS,
    ScheduleEntry,
    cadence_for,
    load_schedule,
    next_check_at,
    upsert,
)
from anydocs.knowledge.store import resolve_snapshots
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register


class Input(BaseModel):
    doc_id: str = Field(default="", description="Doc to schedule. Empty means the whole corpus.")
    corpus: str = ""
    snapshot_id: str = ""
    cadence_days: int = Field(
        default=0,
        ge=0,
        le=MAX_CADENCE_DAYS,
        description=f"Days between checks. 0 picks a default from the page's character "
        f"(release notes and changelogs 2, API reference {DEFAULT_CADENCE_DAYS}, guides 30).",
    )


class Output(BaseModel):
    corpus: str = ""
    snapshot_id: str = ""
    scheduled: list[str] = Field(default_factory=list, description="Doc ids now scheduled.")
    cadence_days: int = 0
    next_check: str = ""


def run(inp: Input, ctx: ToolContext) -> Output:
    snapshots = resolve_snapshots(inp.corpus, inp.snapshot_id)
    if not snapshots:
        raise ToolError("no active snapshot to schedule against")

    for snapshot in snapshots:
        targets = (
            [snapshot.get_doc(inp.doc_id)] if inp.doc_id else list(snapshot.live_docs())
        )
        targets = [doc for doc in targets if doc is not None]
        if not targets:
            continue

        existing = {entry.doc_id: entry for entry in load_schedule(snapshot)}
        scheduled: list[str] = []
        cadence = inp.cadence_days
        next_check = ""
        for doc in targets:
            days = cadence or cadence_for(doc)
            entry = ScheduleEntry(
                doc_id=doc.doc_id,
                corpus=doc.corpus,
                cadence_days=days,
                next_check=next_check_at(days),
                last_result=existing[doc.doc_id].last_result if doc.doc_id in existing else "",
                last_checked=existing[doc.doc_id].last_checked if doc.doc_id in existing else "",
            )
            upsert(snapshot, entry)
            scheduled.append(doc.doc_id)
            cadence, next_check = days, entry.next_check

        telemetry.set_attributes(
            ctx.span,
            {
                telemetry.Attr.SNAPSHOT_ID: snapshot.snapshot_id,
                "freshness.scheduled": len(scheduled),
                "freshness.cadence_days": cadence,
            },
        )
        return Output(
            corpus=snapshot.corpus,
            snapshot_id=snapshot.snapshot_id,
            scheduled=scheduled,
            cadence_days=cadence,
            next_check=next_check,
        )

    raise ToolError(f"nothing to schedule for {inp.doc_id or inp.corpus or '(any corpus)'!r}")


SPEC = register(
    ToolSpec(
        name="schedule_refresh",
        description=(
            "Set how often a doc, or every doc in a corpus, is re-checked for changes. Omit "
            "`cadence_days` to pick a default from the page's character: release notes and "
            "changelogs are checked every 2 days, API reference every 14, guides every 30."
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
