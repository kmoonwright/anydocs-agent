"""The freshness schedule: doc id, next check, cadence, last result.

The shape is fixed here so `schedule_refresh` and the maintenance route have something to
agree on, and so a snapshot written today carries the sidecar a later version will read.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from anydocs.knowledge.store import Snapshot

# Cadence by page character, from the design doc.
DEFAULT_CADENCE_DAYS = 14
CADENCE_BY_TAG = {"release-notes": 2, "changelog": 2, "api-reference": 14, "guide": 30}
MAX_CADENCE_DAYS = 90


class ScheduleEntry(BaseModel):
    doc_id: str
    corpus: str = ""
    cadence_days: int = DEFAULT_CADENCE_DAYS
    next_check: str = ""
    last_result: str = Field(default="", description="unchanged | changed | failed")
    last_checked: str = ""


def next_check_at(cadence_days: int) -> str:
    return (
        datetime.now(UTC) + timedelta(days=cadence_days)
    ).isoformat(timespec="seconds")


def adjust_cadence(current: int, result: str) -> int:
    """Double on unchanged, reset on changed, capped by corpus policy.

    A page that has not moved in months should not be re-fetched weekly, and one that just
    changed probably will again.
    """
    if result == "changed":
        return DEFAULT_CADENCE_DAYS
    if result == "unchanged":
        return min(MAX_CADENCE_DAYS, max(1, current) * 2)
    return current


def cadence_for(doc) -> int:
    """A default cadence from the page's character.

    Release notes and changelogs move constantly and matter when they do; conceptual guides
    barely move at all. Inferred from the title, URL and headings rather than asked of a model,
    because it is a scheduling default, not a judgement.
    """
    haystack = " ".join(
        [doc.title or "", doc.canonical_url or doc.url or ""]
        + [h for chunk in doc.chunks for h in chunk.heading_path]
    ).lower()
    for marker, days in (
        ("changelog", CADENCE_BY_TAG["changelog"]),
        ("release-note", CADENCE_BY_TAG["release-notes"]),
        ("release note", CADENCE_BY_TAG["release-notes"]),
        ("whats-new", CADENCE_BY_TAG["release-notes"]),
        ("what's new", CADENCE_BY_TAG["release-notes"]),
        ("migration", CADENCE_BY_TAG["release-notes"]),
    ):
        if marker in haystack:
            return days
    # A page documenting parameters is reference material; one that does not is a guide.
    if doc.digest.parameters or any(
        marker in haystack for marker in ("reference", "/api/", "api reference")
    ):
        return CADENCE_BY_TAG["api-reference"]
    return CADENCE_BY_TAG["guide"]


def load_schedule(snapshot: Snapshot) -> list[ScheduleEntry]:
    """The current schedule, newest entry per doc winning.

    The sidecar is append-only like the others, so the last line for a doc id is its state and
    the earlier ones are its history.
    """
    latest: dict[str, ScheduleEntry] = {}
    for row in snapshot.read_jsonl(snapshot.freshness_path):
        entry = ScheduleEntry.model_validate(row)
        latest[entry.doc_id] = entry
    return list(latest.values())


def due_entries(snapshot: Snapshot, *, now: str = "") -> list[ScheduleEntry]:
    stamp = now or datetime.now(UTC).isoformat(timespec="seconds")
    return [e for e in load_schedule(snapshot) if not e.next_check or e.next_check <= stamp]


def upsert(snapshot: Snapshot, entry: ScheduleEntry) -> ScheduleEntry:
    snapshot.append_jsonl(snapshot.freshness_path, entry.model_dump())
    return entry


def record_change(snapshot: Snapshot, doc_id: str, diff) -> dict[str, Any]:
    """Store what a refresh found, so "what changed" is answerable without re-fetching."""
    entry = {
        "doc_id": doc_id,
        "checked_at": diff.checked_at,
        "old_hash": diff.old_hash,
        "new_hash": diff.new_hash,
        "sections": [section.model_dump() for section in diff.sections],
    }
    snapshot.append_jsonl(snapshot.changelog_path, entry)
    return entry


def changelog_for(snapshot: Snapshot, doc_id: str) -> list[dict[str, Any]]:
    """Every recorded change to one doc, oldest first."""
    return [
        row for row in snapshot.read_jsonl(snapshot.changelog_path) if row.get("doc_id") == doc_id
    ]
