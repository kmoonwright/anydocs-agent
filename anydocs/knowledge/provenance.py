"""Provenance records: how a digest came to exist, and what happened to it.

Append-only. A `forget` tombstones the digest but keeps its provenance, so "we fetched this and
discarded it" stays answerable while the content itself does not.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from anydocs.knowledge.store import Snapshot


class ProvenanceRecord(BaseModel):
    provenance_id: str
    doc_id: str
    url: str
    fetched_at: str
    http_status: int = 0
    content_type: str = ""
    last_modified: str = ""
    content_hash: str = ""
    robots_decision: str = ""
    digest_model: str = ""
    digest_prompt_version: str = ""
    requested_by: str = "agent"
    redactions: list[str] = Field(
        default_factory=list, description="What the secrets scanner removed before saving"
    )
    event: str = "saved"  # saved | redigested | tombstoned
    reason: str = ""

    @classmethod
    def new(cls, doc_id: str, url: str, **fields: Any) -> ProvenanceRecord:
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        seed = f"{doc_id}:{url}:{stamp}:{fields.get('event', 'saved')}"
        return cls(
            provenance_id=hashlib.sha256(seed.encode()).hexdigest()[:12],
            doc_id=doc_id,
            url=url,
            fetched_at=fields.pop("fetched_at", stamp),
            **fields,
        )


def record(snapshot: Snapshot, entry: ProvenanceRecord) -> ProvenanceRecord:
    snapshot.append_jsonl(snapshot.provenance_path, entry.model_dump())
    return entry


def history(snapshot: Snapshot, doc_id: str) -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord.model_validate(r)
        for r in snapshot.read_jsonl(snapshot.provenance_path)
        if r.get("doc_id") == doc_id
    ]


def all_records(snapshot: Snapshot) -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord.model_validate(r) for r in snapshot.read_jsonl(snapshot.provenance_path)
    ]
