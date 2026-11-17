"""The freshness workflow: workflow 3 from the design doc.

Runs headless on a timer and produces traces with no user at all — which is the point. Each
digest carries a cadence; when it comes due the page is re-fetched, compared, and re-digested if
it moved. The diff is kept as a changelog entry on the doc, so "what changed" is answerable
later without re-fetching anything.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from anydocs import telemetry
from anydocs.budget import DailySpend, current_ledger, set_ledger
from anydocs.knowledge.digest import utcnow
from anydocs.knowledge.freshness import (
    MAX_CADENCE_DAYS,
    ScheduleEntry,
    adjust_cadence,
    cadence_for,
    due_entries,
    load_schedule,
    next_check_at,
    record_change,
    upsert,
)
from anydocs.knowledge.links import detect_contradictions
from anydocs.knowledge.store import CorpusRegistry, open_snapshot
from anydocs.policy import Policy, PolicyDenied, default_policy
from anydocs.tools import registry
from anydocs.tools.registry import ToolError


@dataclass
class DocResult:
    doc_id: str
    corpus: str
    result: str  # unchanged | changed | failed
    cadence_days: int = 0
    next_check: str = ""
    sections_changed: int = 0
    detail: str = ""


@dataclass
class RefreshReport:
    checked: int = 0
    unchanged: int = 0
    changed: int = 0
    failed: int = 0
    contradictions_found: int = 0
    cost_usd: float = 0.0
    docs: list[DocResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "unchanged": self.unchanged,
            "changed": self.changed,
            "failed": self.failed,
            "contradictions_found": self.contradictions_found,
            "cost_usd": round(self.cost_usd, 4),
            "docs": [
                {
                    "doc_id": d.doc_id,
                    "corpus": d.corpus,
                    "result": d.result,
                    "cadence_days": d.cadence_days,
                    "next_check": d.next_check,
                    "sections_changed": d.sections_changed,
                    "detail": d.detail,
                }
                for d in self.docs
            ],
        }


def run_refresh(
    *,
    corpora: list[str] | None = None,
    limit: int = 25,
    policy: Policy | None = None,
    digest: bool = True,
    now: str = "",
) -> RefreshReport:
    """Work the freshness schedule once.

    Every doc gets its own ledger, as in batch ingest: the per-turn caps bound one interactive
    question, and a refresh pass over 25 pages is not one.
    """
    pol = policy or default_policy()
    registry_ = CorpusRegistry()
    names = corpora or registry_.active_corpora()
    report = RefreshReport()

    with telemetry.turn_context(
        turn_id=uuid.uuid4().hex[:12],
        user_id="system",
        corpus_active=tuple(names),
        policy_name=pol.name,
    ):
        with telemetry.turn_span(
            message=f"refresh {', '.join(names) or '(no corpora)'}", maintenance=True
        ) as span:
            for name in names:
                try:
                    snapshot = open_snapshot(name)
                except Exception as exc:
                    report.failed += 1
                    report.docs.append(
                        DocResult(doc_id="", corpus=name, result="failed", detail=str(exc))
                    )
                    continue

                # A corpus with no schedule yet is scheduled now rather than skipped forever.
                if not load_schedule(snapshot):
                    _seed_schedule(snapshot)

                for entry in due_entries(snapshot, now=now)[:limit]:
                    set_ledger(pol.new_ledger())
                    result = _refresh_one(snapshot, entry, pol, digest=digest, report=report)
                    report.cost_usd += current_ledger().cost_usd
                    report.docs.append(result)
                    report.checked += 1
                    if result.result == "changed":
                        report.changed += 1
                    elif result.result == "unchanged":
                        report.unchanged += 1
                    else:
                        report.failed += 1

            telemetry.set_attributes(
                span,
                {
                    "refresh.checked": report.checked,
                    "refresh.changed": report.changed,
                    "refresh.unchanged": report.unchanged,
                    "refresh.failed": report.failed,
                    "refresh.contradictions": report.contradictions_found,
                    telemetry.Attr.TURN_COST_USD: round(report.cost_usd, 6),
                },
            )

    DailySpend().add(report.cost_usd)
    return report


def _seed_schedule(snapshot) -> None:
    for doc in snapshot.live_docs():
        days = cadence_for(doc)
        upsert(
            snapshot,
            ScheduleEntry(
                doc_id=doc.doc_id,
                corpus=doc.corpus,
                cadence_days=days,
                # Due immediately: a corpus that has never been checked has nothing to wait for.
                next_check="",
            ),
        )


def _refresh_one(
    snapshot, entry: ScheduleEntry, policy: Policy, *, digest: bool, report: RefreshReport
) -> DocResult:
    doc = snapshot.get_doc(entry.doc_id)
    if doc is None or doc.tombstoned:
        # A forgotten doc drops off the schedule instead of being re-fetched forever.
        return DocResult(
            doc_id=entry.doc_id,
            corpus=entry.corpus,
            result="unchanged",
            detail="tombstoned or missing; removed from the schedule",
            cadence_days=entry.cadence_days,
            next_check=_record(snapshot, entry, "unchanged", cadence=MAX_CADENCE_DAYS),
        )

    try:
        diff = registry.invoke(
            "diff_page",
            {
                "doc_id": doc.doc_id,
                "corpus": snapshot.corpus,
                "snapshot_id": snapshot.snapshot_id,
            },
            policy=policy,
            requested_by="system",
        )
    except (ToolError, PolicyDenied) as exc:
        # Fail visibly, and keep going. A page whose host has disappeared, or whose domain the
        # policy now refuses, is a result the schedule records — not a reason to abandon the
        # rest of the pass. The longer cadence stops a dead link being retried every cycle.
        return DocResult(
            doc_id=doc.doc_id,
            corpus=doc.corpus,
            result="failed",
            detail=str(exc),
            cadence_days=entry.cadence_days,
            next_check=_record(snapshot, entry, "failed", cadence=entry.cadence_days * 2),
        )

    if not diff.changed:
        cadence = adjust_cadence(entry.cadence_days, "unchanged")
        return DocResult(
            doc_id=doc.doc_id,
            corpus=doc.corpus,
            result="unchanged",
            cadence_days=cadence,
            next_check=_record(snapshot, entry, "unchanged", cadence=cadence),
        )

    detail = _reingest(snapshot, doc, diff, policy, digest=digest, report=report)
    cadence = adjust_cadence(entry.cadence_days, "changed")
    return DocResult(
        doc_id=doc.doc_id,
        corpus=doc.corpus,
        result="changed",
        cadence_days=cadence,
        next_check=_record(snapshot, entry, "changed", cadence=cadence),
        sections_changed=diff.sections_changed + diff.sections_added + diff.sections_removed,
        detail=detail,
    )


def _record(snapshot, entry: ScheduleEntry, result: str, *, cadence: int) -> str:
    capped = max(1, min(MAX_CADENCE_DAYS, cadence))
    next_check = next_check_at(capped)
    upsert(
        snapshot,
        entry.model_copy(
            update={
                "cadence_days": capped,
                "next_check": next_check,
                "last_result": result,
                "last_checked": utcnow(),
            }
        ),
    )
    return next_check


def _reingest(snapshot, doc, diff, policy: Policy, *, digest: bool, report: RefreshReport) -> str:
    """Re-digest a changed page, store the diff as a changelog entry, and re-check conflicts."""
    from anydocs.knowledge.digest import heuristic_digest

    if digest:
        digest_result = registry.invoke(
            "digest_page",
            {"markdown": diff.markdown, "url": diff.url, "title": doc.title},
            policy=policy,
            requested_by="system",
        )
        new_digest, quality = digest_result.digest, digest_result.quality
        model, prompt_version = digest_result.model, digest_result.prompt_version
    else:
        new_digest = heuristic_digest(diff.markdown, doc.title)
        quality = new_digest.quality_score()
        model, prompt_version = "heuristic", "heuristic@1"

    registry.invoke(
        "save_note",
        {
            "corpus": snapshot.corpus,
            "url": diff.url,
            "markdown": diff.markdown,
            "digest": new_digest.model_dump(mode="json"),
            "title": doc.title,
            "snapshot_id": snapshot.snapshot_id,
            "fetched_at": diff.checked_at,
            "http_status": diff.http_status,
            "robots_decision": diff.robots_decision,
            "digest_model": model,
            "digest_prompt_version": prompt_version,
            "digest_quality": quality,
        },
        policy=policy,
        requested_by="system",
    )

    # The diff itself is the changelog entry: what changed, when, section by section.
    record_change(snapshot, doc.doc_id, diff)

    # A page that just changed may now disagree with a page it used to match.
    try:
        found = detect_contradictions(snapshot, doc.doc_id)
    except Exception as exc:
        return f"re-digested; contradiction check failed: {type(exc).__name__}: {exc}"
    report.contradictions_found += len(found)
    if found:
        return f"re-digested; {len(found)} contradiction(s) recorded"
    return "re-digested"

