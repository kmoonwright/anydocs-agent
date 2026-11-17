"""Per-turn budget ledger.

The runner enforces budgets; the prompt is never trusted to. Everything that costs something —
a tool call, an LLM round trip — reports it here, and the ledger answers two questions: has a
limit been crossed, and how much has this turn spent so far.

Lives in its own module so `policy.py` and `models/roles.py` can both use it without importing
each other.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """A hard limit was crossed. Callers convert this into a `truncated` partial result rather
    than letting it escape as a 500."""

    def __init__(self, limit: str, detail: str) -> None:
        super().__init__(f"budget exceeded: {limit} ({detail})")
        self.limit = limit
        self.detail = detail


@dataclass
class Ledger:
    max_tool_calls: int = 24
    max_tokens: int = 60_000
    max_wall_ms: int = 45_000
    daily_usd_cap: float = 25.0

    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    search_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)
    per_tool: dict[str, int] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)

    @property
    def fraction_spent(self) -> float:
        """How far through the turn's budget we are, on whichever axis is furthest along.

        The orchestrator reads this to downgrade the Auditor to a cheaper model past 80%.
        """
        fractions = [
            self.tool_calls / self.max_tool_calls if self.max_tool_calls else 0.0,
            self.tokens / self.max_tokens if self.max_tokens else 0.0,
            self.elapsed_ms / self.max_wall_ms if self.max_wall_ms else 0.0,
        ]
        return max(fractions)

    def check_deadline(self) -> None:
        if self.max_wall_ms and self.elapsed_ms > self.max_wall_ms:
            raise BudgetExceeded("max_wall_ms", f"{self.elapsed_ms}ms > {self.max_wall_ms}ms")

    def record_tool_call(self, name: str) -> None:
        self.check_deadline()
        if self.max_tool_calls and self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded(
                "max_tool_calls_per_turn", f"{self.tool_calls} calls already made"
            )
        self.tool_calls += 1
        self.per_tool[name] = self.per_tool.get(name, 0) + 1

    def record_search_call(self, max_calls: int) -> None:
        if max_calls and self.search_calls >= max_calls:
            raise BudgetExceeded("web_search.max_calls_per_turn", f"{self.search_calls} already")
        self.search_calls += 1

    def record_usage(self, *, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost_usd
        if self.max_tokens and self.tokens > self.max_tokens:
            raise BudgetExceeded("max_tokens_per_turn", f"{self.tokens} > {self.max_tokens}")


_ledger: ContextVar[Ledger | None] = ContextVar("anydocs_ledger", default=None)


def current_ledger() -> Ledger:
    """The active turn's ledger. Outside a turn, returns a throwaway with generous limits so
    CLI one-offs and tests do not need to set one up."""
    ledger = _ledger.get()
    if ledger is None:
        ledger = Ledger()
        _ledger.set(ledger)
    return ledger


def set_ledger(ledger: Ledger) -> None:
    _ledger.set(ledger)


def reset_ledger() -> None:
    _ledger.set(None)


# ----------------------------------------------------------------------------------------
# Daily spend, across turns
# ----------------------------------------------------------------------------------------


class DailySpend:
    """A per-day USD total, kept in a small JSON file beside the knowledge store.

    Crude on purpose: the cap exists to stop a runaway loop costing real money overnight, not to
    be an accounting system. When the cap is hit, write tools stop first, then web search, then
    the agent answers from the store only — with a visible notice in the answer.
    """

    def __init__(self) -> None:
        from anydocs.config import get_settings

        self.path = get_settings().data_dir / "spend.json"

    def _today(self) -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _read(self) -> dict:
        import json

        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text() or "{}")
        except (OSError, ValueError):
            return {}

    def total(self) -> float:
        return float(self._read().get(self._today(), 0.0))

    def add(self, usd: float) -> float:
        import json

        if usd <= 0:
            return self.total()
        data = self._read()
        day = self._today()
        data[day] = round(float(data.get(day, 0.0)) + usd, 6)
        # Keep the last 30 days; this file is a guardrail, not a ledger.
        for key in sorted(data)[:-30]:
            data.pop(key, None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, sort_keys=True))
        return data[day]

    def degradation(self, cap: float) -> str:
        """What the cap currently forbids: '', 'write', 'search' or 'all'.

        Staged rather than a hard stop, so the agent degrades to read-only before going silent.
        """
        if cap <= 0:
            return ""
        spent = self.total()
        if spent >= cap:
            return "all"
        if spent >= cap * 0.95:
            return "search"
        if spent >= cap * 0.85:
            return "write"
        return ""
