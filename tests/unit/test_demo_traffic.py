"""The synthetic traffic schedule: how often sessions start, and where a fault lands."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from anydocs import demo_traffic
from anydocs.config import REPO_ROOT


@pytest.fixture
def traffic():
    return demo_traffic.load()


def _expected_sessions(traffic, day: datetime) -> float:
    minutes = traffic["schedule"]["tick_minutes"]
    ticks = range(0, 24 * 60, minutes)
    return sum(demo_traffic.start_probability(traffic, day + timedelta(minutes=m)) for m in ticks)


def test_a_weekday_expects_the_configured_sessions_and_a_weekend_fewer(traffic):
    monday, saturday = datetime(2026, 11, 16, tzinfo=UTC), datetime(2026, 11, 21, tzinfo=UTC)
    per_day = traffic["schedule"]["sessions_per_day"]
    assert _expected_sessions(traffic, monday) == pytest.approx(per_day)
    assert _expected_sessions(traffic, saturday) == pytest.approx(
        per_day * traffic["schedule"]["weekend_factor"]
    )


def test_traffic_follows_the_working_day(traffic):
    night = demo_traffic.start_probability(traffic, datetime(2026, 11, 16, 6, tzinfo=UTC))
    afternoon = demo_traffic.start_probability(traffic, datetime(2026, 11, 16, 17, tzinfo=UTC))
    assert afternoon > 5 * night


def test_the_cron_schedule_matches_the_tick_the_probabilities_assume(traffic):
    crons = json.loads((REPO_ROOT / "vercel.json").read_text())["crons"]
    assert crons == [
        {"path": "/cron/turn", "schedule": f"*/{traffic['schedule']['tick_minutes']} * * * *"}
    ]


def test_every_conversation_names_only_a_bundled_corpus(traffic):
    """A conversation pinned to a corpus the deploy does not ship would fail every time."""
    bundled = set(json.loads((REPO_ROOT / "bundled" / "corpora.json").read_text()))
    assert len(traffic["conversations"]) >= 10
    for conversation in traffic["conversations"]:
        assert set(conversation.get("corpora") or []) <= bundled, conversation
        assert 1 <= len(conversation["turns"]) <= 3
        assert all(turn["message"].strip() for turn in conversation["turns"])
    assert len(traffic["schedule"]["hourly_weights_utc"]) == 24


def test_a_fault_rides_on_the_last_turn_only(traffic, monkeypatch):
    from anydocs.orchestrator import turn as turn_module

    seen = []

    def fake_run_turn(request):
        seen.append(request)
        return SimpleNamespace(trace_id=f"t{len(seen)}", answer=None, cost_usd=0.0)

    monkeypatch.setattr(turn_module, "run_turn", fake_run_turn)
    session, turns, overrides = demo_traffic.plan(
        traffic, random.Random(0), conversation=2, fault="unknown-judge-model"
    )
    assert len(turns) == 3
    demo_traffic.run_session(session, turns, [], overrides)

    assert [r.overrides for r in seen[:-1]] == [{}, {}]
    assert seen[-1].overrides == {"model_roles": {"judge": "claude-sonnet-4-9"}}
    assert {r.session_id for r in seen} == {session.session_id}
    assert {r.user_id for r in seen} == {session.user_id}


def test_a_turn_that_raises_ends_the_session_and_is_reported(traffic, monkeypatch):
    from anydocs.orchestrator import turn as turn_module

    def failing_run_turn(request):
        raise RuntimeError("model not found")

    monkeypatch.setattr(turn_module, "run_turn", failing_run_turn)
    session, turns, overrides = demo_traffic.plan(traffic, random.Random(0), conversation=2)
    demo_traffic.run_session(session, turns, [], overrides)
    assert session.turns == [{"error": "RuntimeError: model not found"}]


def test_faults_start_at_about_the_configured_rate(traffic):
    rng = random.Random(42)
    faulty = sum(bool(demo_traffic.plan(traffic, rng)[0].fault) for _ in range(2000))
    assert faulty / 2000 == pytest.approx(traffic["schedule"]["fault_rate"], abs=0.03)
