"""Turns paused for a human decision.

A turn that hits `ask` mode is suspended, not restarted: the state it had reached — the plan, the
Librarian's draft, the Scout's candidates, the budget already spent — is kept against the resume
token so `/invoke/approve` continues from the Archivist rather than re-searching and re-judging.

Persisted under the data directory rather than held in memory, because `anydocs approve` runs in
a different process from the `anydocs ask` that paused, and an in-memory store would make that
command impossible to use.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anydocs.budget import Ledger
from anydocs.config import get_settings
from anydocs.orchestrator.planner import Plan
from anydocs.policy import RESUME_TOKEN_TTL_S
from anydocs.specialists.results import RESULT_SCHEMAS
from anydocs.specialists.runner import SpecialistResult

if TYPE_CHECKING:  # avoids a cycle: turn imports this module, not the other way round
    from anydocs.orchestrator.graph import TurnRequest


@dataclass
class SuspendedTurn:
    """A turn waiting on a human decision."""

    request: TurnRequest
    plan: Plan
    librarian: SpecialistResult | None
    scout: SpecialistResult | None
    notes: list[str]
    action: str
    payload: dict[str, Any]
    ledger: Ledger
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return {
            # `on_event` is a live callback; a resumed turn gets a new one or none.
            "request": {k: v for k, v in asdict(self.request).items() if k != "on_event"},
            "plan": self.plan.model_dump(mode="json"),
            "librarian": _result_to_json(self.librarian),
            "scout": _result_to_json(self.scout),
            "notes": self.notes,
            "action": self.action,
            "payload": self.payload,
            "ledger": {
                "max_tool_calls": self.ledger.max_tool_calls,
                "max_tokens": self.ledger.max_tokens,
                "max_wall_ms": self.ledger.max_wall_ms,
                "daily_usd_cap": self.ledger.daily_usd_cap,
                "tool_calls": self.ledger.tool_calls,
                "tokens_in": self.ledger.tokens_in,
                "tokens_out": self.ledger.tokens_out,
                "cost_usd": self.ledger.cost_usd,
                "search_calls": self.ledger.search_calls,
                "per_tool": self.ledger.per_tool,
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SuspendedTurn:
        # Imported here rather than at module scope: `turn` imports this module, so the other
        # direction has to be deferred to the one place that actually needs the constructor.
        from anydocs.orchestrator.graph import TurnRequest

        ledger_fields = dict(data["ledger"])
        ledger = Ledger(
            max_tool_calls=ledger_fields["max_tool_calls"],
            max_tokens=ledger_fields["max_tokens"],
            max_wall_ms=ledger_fields["max_wall_ms"],
            daily_usd_cap=ledger_fields["daily_usd_cap"],
        )
        # The wall clock restarts on resume: the elapsed time was a human reading an approval
        # prompt, which is not the agent's budget to spend.
        ledger.tool_calls = ledger_fields["tool_calls"]
        ledger.tokens_in = ledger_fields["tokens_in"]
        ledger.tokens_out = ledger_fields["tokens_out"]
        ledger.cost_usd = ledger_fields["cost_usd"]
        ledger.search_calls = ledger_fields["search_calls"]
        ledger.per_tool = dict(ledger_fields["per_tool"])
        return cls(
            request=TurnRequest(**data["request"]),
            plan=Plan.model_validate(data["plan"]),
            librarian=_result_from_json(data["librarian"]),
            scout=_result_from_json(data["scout"]),
            notes=list(data["notes"]),
            action=data["action"],
            payload=dict(data["payload"]),
            ledger=ledger,
            created_at=float(data["created_at"]),
        )


def _result_to_json(result: SpecialistResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = result.as_dict()
    payload["result_schema"] = type(result.result).__name__ if result.result else ""
    return payload


def _result_from_json(data: dict[str, Any] | None) -> SpecialistResult | None:
    if data is None:
        return None
    schema = RESULT_SCHEMAS.get(data.get("result_schema", ""))
    return SpecialistResult(
        specialist=data["specialist"],
        result=schema.model_validate(data["result"]) if schema and data.get("result") else None,
        confidence=data.get("confidence", 0.0),
        cost_usd=data.get("cost_usd", 0.0),
        tool_calls=data.get("tool_calls", 0),
        truncated=data.get("truncated", False),
        notes=list(data.get("notes", [])),
    )


class _PendingStore:
    """Suspended turns, keyed by resume token.

    Persisted under the data directory, not held in memory: `anydocs approve` runs in a
    different process from the `anydocs ask` that paused, and an in-memory store would make that
    command impossible to use. Entries expire with the resume token's own TTL.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def _dir(self) -> Path:
        path = get_settings().data_dir / "pending"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, token: str) -> Path:
        # The token is long and contains base64 padding characters; its hash is the filename.
        return self._dir / f"{hashlib.sha256(token.encode()).hexdigest()[:32]}.json"

    def put(self, token: str, state: SuspendedTurn) -> None:
        with self._lock:
            self._path(token).write_text(json.dumps(state.to_json(), default=str))
            self._sweep()

    def pop(self, token: str) -> SuspendedTurn | None:
        with self._lock:
            path = self._path(token)
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                path.unlink(missing_ok=True)
                return None
            path.unlink(missing_ok=True)
            return SuspendedTurn.from_json(data)

    def _sweep(self) -> None:
        """Drop entries older than the resume token's TTL; they can no longer be redeemed."""
        cutoff = time.time() - RESUME_TOKEN_TTL_S
        for path in self._dir.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue

    def clear(self) -> None:
        with self._lock:
            for path in self._dir.glob("*.json"):
                path.unlink(missing_ok=True)


_pending = _PendingStore()


def pending_store() -> _PendingStore:
    return _pending


def reset_pending_store() -> None:
    """Drop every in-memory handle on suspended turns.

    What a restart does. Tests use it to prove a paused turn survives the process that paused
    it, since the state itself lives on disk.
    """
    global _pending
    _pending = _PendingStore()


def estimate_crawl_usd(pages: int) -> float:
    """Crawl cost shown in the approval payload: pages x average digest tokens x price."""
    if pages <= 0:
        return 0.0
    settings = get_settings()
    model = settings.role("summarizer").model
    price = settings.price_per_mtok(model)
    avg_in, avg_out = 6000, 900
    return round(
        pages * (avg_in / 1e6 * price.get("input", 0.0) + avg_out / 1e6 * price.get("output", 0.0)),
        4,
    )
