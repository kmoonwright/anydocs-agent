"""Budgets are enforced by the runner, not asked for in a prompt."""

from __future__ import annotations

import pytest

from anydocs.budget import BudgetExceeded, DailySpend, Ledger


def test_tool_call_cap():
    ledger = Ledger(max_tool_calls=2)
    ledger.record_tool_call("search_docs")
    ledger.record_tool_call("get_doc")
    with pytest.raises(BudgetExceeded, match="max_tool_calls_per_turn"):
        ledger.record_tool_call("get_doc")
    assert ledger.per_tool == {"search_docs": 1, "get_doc": 1}


def test_token_cap():
    ledger = Ledger(max_tokens=100)
    ledger.record_usage(tokens_in=60, tokens_out=20, cost_usd=0.01)
    with pytest.raises(BudgetExceeded, match="max_tokens_per_turn"):
        ledger.record_usage(tokens_in=40, tokens_out=0, cost_usd=0.01)


def test_wall_clock_cap(monkeypatch):
    ledger = Ledger(max_wall_ms=10)
    monkeypatch.setattr(ledger, "started_at", ledger.started_at - 5.0)
    with pytest.raises(BudgetExceeded, match="max_wall_ms"):
        ledger.check_deadline()


def test_search_calls_have_their_own_cap():
    ledger = Ledger()
    ledger.record_search_call(2)
    ledger.record_search_call(2)
    with pytest.raises(BudgetExceeded, match="web_search"):
        ledger.record_search_call(2)


def test_fraction_spent_tracks_the_axis_furthest_along():
    """This is what decides whether the Auditor gets downgraded to a cheaper model."""
    ledger = Ledger(max_tool_calls=10, max_tokens=1000, max_wall_ms=10_000_000)
    ledger.tool_calls = 9
    ledger.record_usage(tokens_in=10, tokens_out=10, cost_usd=0.0)
    assert ledger.fraction_spent == pytest.approx(0.9)


def test_daily_cap_degrades_in_stages_rather_than_stopping_dead(data_dir):
    spend = DailySpend()
    assert spend.degradation(10.0) == ""
    spend.add(8.6)
    assert spend.degradation(10.0) == "write"
    spend.add(1.0)
    assert spend.degradation(10.0) == "search"
    spend.add(1.0)
    assert spend.degradation(10.0) == "all"


def test_daily_spend_survives_a_corrupt_file(data_dir):
    spend = DailySpend()
    spend.path.parent.mkdir(parents=True, exist_ok=True)
    spend.path.write_text("{not json")
    assert spend.total() == 0.0
    assert spend.add(1.0) == 1.0
