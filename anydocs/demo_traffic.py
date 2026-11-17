"""Synthetic traffic for a demo project: a few sessions a day that look like people.

Vercel Cron can only call on a fixed schedule, so it calls often and each tick decides, by a
weighted coin, whether a session starts. The weights follow a working day, so a dashboard shows
traffic rising in the morning and falling off at night rather than a flat line or a burst on the
hour. `config/demo-traffic.yaml` holds every number and every question; this module holds none.

A session is one to three turns of one conversation under one `session.id`, from one of a small
pool of `demo-user-NN` users. A few sessions carry a fault: an override that a real deploy could
plausibly ship by mistake, such as a model id that does not exist or a budget set far too low.
The fault is visible the way it would be in production — an error span or a truncated answer,
with the offending override recorded on the turn span — which is the point: it is something to
debug from the trace.

No span is opened here. Each turn is an ordinary `run_turn`, traced exactly as `/invoke` is.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml

from anydocs.config import CONFIG_DIR

log = logging.getLogger(__name__)

# A follow-up starts only while this much of the function's 300 s is left unspent. One turn has
# measured at up to 110 s, so a later start risks the host killing the session mid-turn.
_FOLLOW_UP_CUTOFF_S = 120


def load() -> dict[str, Any]:
    with (CONFIG_DIR / "demo-traffic.yaml").open() as fh:
        return yaml.safe_load(fh) or {}


def start_probability(config: dict[str, Any], now: datetime) -> float:
    """The chance a session starts on the tick at `now` (UTC).

    Spread so that a weekday expects `sessions_per_day` sessions, shaped by `hourly_weights_utc`;
    a weekend day expects `weekend_factor` of that.
    """
    schedule = config["schedule"]
    weights = schedule["hourly_weights_utc"]
    ticks_per_hour = 60 / schedule["tick_minutes"]
    share = weights[now.hour] / sum(weights)
    p = schedule["sessions_per_day"] * share / ticks_per_hour
    if now.weekday() >= 5:
        p *= schedule["weekend_factor"]
    return min(p, 1.0)


@dataclass
class Session:
    session_id: str
    user_id: str
    conversation: int
    fault: str = ""
    turns: list[dict[str, Any]] = field(default_factory=list)


def plan(
    config: dict[str, Any],
    rng: random.Random,
    *,
    conversation: int | None = None,
    fault: str | None = None,
) -> tuple[Session, list[dict[str, Any]], dict[str, Any]]:
    """Pick the conversation, the user and whether a fault rides on it. No model is called."""
    conversations = config["conversations"]
    index = conversation % len(conversations) if conversation is not None else rng.randrange(
        len(conversations)
    )
    faults = {f["name"]: f for f in config.get("faults") or []}
    if fault is None:
        fault = (
            rng.choice(sorted(faults))
            if faults and rng.random() < config["schedule"]["fault_rate"]
            else ""
        )
    elif fault and fault not in faults:
        raise ValueError(f"Unknown fault {fault!r}; known: {', '.join(sorted(faults))}")
    session = Session(
        session_id=f"demo-{uuid.uuid4().hex[:12]}",
        user_id=f"demo-user-{rng.randint(1, config['users']):02d}",
        conversation=index,
        fault=fault,
    )
    return session, conversations[index]["turns"], faults.get(fault, {}).get("overrides", {})


def run_session(
    session: Session,
    turns: list[dict[str, Any]],
    corpora: list[str],
    fault_overrides: dict[str, Any],
) -> Session:
    """Run the turns in order under one session. The fault, if any, rides on the last turn.

    The last turn so that the turns before it complete and the session reads as a conversation
    that went wrong, not one that never started. A turn that raises ends the session: its error
    is already on its span, which is where it is meant to be found.
    """
    from anydocs.orchestrator.turn import TurnRequest, run_turn

    started = time.monotonic()
    for position, turn in enumerate(turns):
        if position and time.monotonic() - started > _FOLLOW_UP_CUTOFF_S:
            session.turns.append({"skipped": "out of time for a follow-up"})
            break
        last = position == len(turns) - 1
        try:
            outcome = run_turn(
                TurnRequest(
                    message=turn["message"],
                    session_id=session.session_id,
                    user_id=session.user_id,
                    corpora=corpora,
                    overrides=fault_overrides if last else {},
                )
            )
        except Exception as exc:
            log.warning("Demo session %s turn %d failed: %s", session.session_id, position, exc)
            session.turns.append({"error": f"{type(exc).__name__}: {exc}"})
            break
        session.turns.append(
            {
                "trace_id": outcome.trace_id,
                "confidence": outcome.answer.confidence if outcome.answer else 0.0,
                "cost_usd": outcome.cost_usd,
            }
        )
    return session
