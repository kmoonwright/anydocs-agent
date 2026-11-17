"""One turn, start to finish.

What lives here is the envelope around the flow: the budget ledger, the session and profile, the
turn span, the guardrail downgrade, and the suspend-and-resume that `/invoke/approve` needs.

Three neighbours carry the rest. `graph.py` is the flow itself — workflow 1 as a LangGraph state
graph. `delegation.py` builds what each specialist is told. `suspension.py` keeps a turn that
paused for approval so it can be continued rather than started over.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from anydocs import telemetry
from anydocs.budget import BudgetExceeded, DailySpend, Ledger, current_ledger, set_ledger
from anydocs.config import get_settings
from anydocs.guardrails import check_injection
from anydocs.knowledge.store import CorpusRegistry
from anydocs.memory.profile import get_profile_store, infer_from_turn
from anydocs.memory.session import Turn, get_session_store
from anydocs.models.roles import set_role_overrides
from anydocs.orchestrator.delegation import result_of
from anydocs.orchestrator.graph import TurnRequest, TurnState, emit, turn_graph
from anydocs.orchestrator.planner import Plan, make_plan
from anydocs.orchestrator.suspension import SuspendedTurn, estimate_crawl_usd, pending_store
from anydocs.orchestrator.synthesizer import FinalAnswer
from anydocs.policy import (
    NeedsApproval,
    Policy,
    PolicyDenied,
    policy_with_overrides,
)
from anydocs.specialists.results import LibrarianResult, ScoutResult
from anydocs.specialists.runner import SpecialistResult


@dataclass
class TurnOutcome:
    """What `/invoke` returns. Exactly one of `answer` or `pending_approval` is set."""

    turn_id: str
    answer: FinalAnswer | None = None
    pending_approval: dict[str, Any] | None = None
    trace_id: str = ""
    specialists: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0


# ----------------------------------------------------------------------------------------
# The turn
# ----------------------------------------------------------------------------------------


def run_turn(request: TurnRequest) -> TurnOutcome:
    turn_id = uuid.uuid4().hex[:12]
    settings = get_settings()
    policy = policy_with_overrides(request.overrides.get("policy"))
    set_role_overrides(request.overrides.get("model_roles"))

    ledger = policy.new_ledger()
    set_ledger(ledger)

    sessions = get_session_store()
    session = sessions.get(request.session_id or turn_id, request.user_id)
    profile = get_profile_store().get(request.user_id)

    with telemetry.turn_context(
        turn_id=turn_id,
        session_id=session.session_id,
        user_id=request.user_id or ("system" if request.maintenance else ""),
        corpus_active=tuple(request.corpora),
        policy_name=policy.name,
        operator_notes_version=settings.operator_notes_version,
        extra={telemetry.Attr.USER_PROFILE_VERSION: profile.version},
    ):
        with telemetry.turn_span(
            message=request.message,
            maintenance=request.maintenance,
            overrides=request.overrides or None,
        ) as span:
            trace_id = f"{span.get_span_context().trace_id:032x}"

            notes: list[str] = []
            notice = _spend_notice(policy, notes)

            injected, reason = check_injection(request.message)
            if injected:
                # Downgrade rather than refuse: a question can mention prompt injection without
                # being one. Read-only tools only, and the span says so.
                policy = policy_with_overrides(
                    {"crawl": {"mode": "off"}, "write": {"mode": "off"}, "delete": {"mode": "off"}}
                )
                notes.append(f"guardrail: prompt-injection pattern in the user turn ({reason})")
                telemetry.set_attributes(span, {"guardrail.injection": True})

            # Planning is inside the try: the planner's own call spends budget, so a breach can
            # happen before there is a plan at all, and that still has to reach the user as an
            # answer rather than as a 500.
            plan = Plan(specialists=[], rationale="planning did not complete")
            try:
                plan = make_plan(
                    request.message,
                    corpora=request.corpora,
                    known_corpora=CorpusRegistry().active_corpora(),
                    session_summary=session.context_blurb(),
                    profile=profile.as_prompt_block(),
                )
                outcome = _execute(request, plan, policy, notes, turn_id, trace_id, notice)
            except BudgetExceeded as exc:
                # A breach is visible on the span and in the answer, not swallowed.
                notes.append(str(exc))
                telemetry.set_attributes(span, {"budget.exceeded": exc.limit})
                outcome = TurnOutcome(
                    turn_id=turn_id,
                    answer=_budget_answer(exc),
                    trace_id=trace_id,
                    plan=plan.model_dump(mode="json"),
                )

            _finish_span(span, outcome, ledger)

        if outcome.answer:
            DailySpend().add(ledger.cost_usd)
            sessions.record_turn(
                session.session_id,
                Turn(
                    message=request.message,
                    answer=outcome.answer.answer,
                    confidence=outcome.answer.confidence,
                    citations=[c.doc_id for c in outcome.answer.citations],
                ),
                corpora=[c.corpus for c in outcome.answer.citations if c.corpus],
            )
            # Noticed, never acted on: an inference becomes a suggestion with its evidence, and
            # only the user turns it into a preference.
            infer_from_turn(request.user_id, request.message)
            _roll_up_session(sessions, session.session_id)
        outcome.cost_usd = round(ledger.cost_usd, 6)
        return outcome


def _execute(
    request: TurnRequest,
    plan: Plan,
    policy: Policy,
    notes: list[str],
    turn_id: str,
    trace_id: str,
    notice: str,
) -> TurnOutcome:
    """Run the flow in `graph.py` and shape its final state into a `TurnOutcome`."""
    state = _initial_state(request, plan, policy, notes, notice)
    emit(request.on_event, "plan", plan.model_dump(mode="json"))
    try:
        final = turn_graph().invoke(state)
    except NeedsApproval as exc:
        # The graph shares `results` with the state we passed in, so the specialists that ran
        # before the pause are recoverable even though `invoke` never returned.
        return _suspend(request, plan, state["results"], notes, exc, turn_id, trace_id)
    return _outcome(final, turn_id, trace_id, plan)


def _initial_state(
    request: TurnRequest,
    plan: Plan,
    policy: Policy,
    notes: list[str],
    notice: str,
    **overrides: Any,
) -> TurnState:
    state: TurnState = {
        "request": request,
        "plan": plan,
        "policy": policy,
        "notes": notes,
        "results": [],
        "notice": notice,
        "pre_approved": False,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _outcome(final: TurnState, turn_id: str, trace_id: str, plan: Plan) -> TurnOutcome:
    return TurnOutcome(
        turn_id=turn_id,
        answer=final.get("answer"),
        trace_id=trace_id,
        specialists=[r.as_dict() for r in final["results"]],
        plan=plan.model_dump(mode="json"),
    )


# ----------------------------------------------------------------------------------------
# Approval suspend and resume
# ----------------------------------------------------------------------------------------


def _suspend(
    request: TurnRequest,
    plan: Plan,
    results: list[SpecialistResult],
    notes: list[str],
    exc: NeedsApproval,
    turn_id: str,
    trace_id: str,
) -> TurnOutcome:
    payload = dict(exc.payload)
    payload["estimated_usd"] = estimate_crawl_usd(payload.get("estimated_pages", 0))
    pending_store().put(
        exc.resume_token,
        SuspendedTurn(
            request=request,
            plan=plan,
            librarian=_ran("librarian", results),
            scout=_ran("scout", results),
            notes=notes,
            action=exc.action,
            payload=payload,
            ledger=current_ledger(),
        ),
    )
    return TurnOutcome(
        turn_id=turn_id,
        pending_approval={
            "action": exc.action,
            "resume_token": exc.resume_token,
            "reason": f"policy requires approval for {exc.action}",
            **payload,
        },
        trace_id=trace_id,
        plan=plan.model_dump(mode="json"),
    )


def resume_turn(resume_token: str, *, approve: bool, note: str = "") -> TurnOutcome:
    """Continue a turn paused on a write action. The decision is its own span.

    The turn re-enters the same graph rather than restating its tail: approving starts at the
    Archivist with the approval already granted, and declining skips straight to the audit, so
    the answer is produced by exactly the nodes that would have produced it had nobody been asked.
    """
    from anydocs.policy import verify_resume_token

    # Signature and TTL first: a forged token must not even find the pending turn.
    verify_resume_token(resume_token)
    state = pending_store().pop(resume_token)
    if state is None:
        raise PolicyDenied("approve", "no suspended turn for that resume token")

    set_ledger(state.ledger)
    settings = get_settings()
    policy = policy_with_overrides(state.request.overrides.get("policy"))
    turn_id = uuid.uuid4().hex[:12]

    with telemetry.turn_context(
        turn_id=turn_id,
        session_id=state.request.session_id,
        user_id=state.request.user_id,
        policy_name=policy.name,
        operator_notes_version=settings.operator_notes_version,
    ):
        with telemetry.turn_span(message=state.request.message) as span:
            trace_id = f"{span.get_span_context().trace_id:032x}"
            with telemetry.approval_span(
                action=state.action,
                mode="ask",
                pending=False,
                decision="approved" if approve else "declined",
                **{"policy.note": note},
            ):
                pass

            notes = list(state.notes)
            if not approve:
                # A declined crawl is remembered as a decision, never as content: nothing the
                # Scout fetched is ingested.
                declined = f"a human declined the {state.action} action"
                notes.append(f"{declined}: {note}" if note else declined)

            scout_result = result_of(state.scout, ScoutResult)
            can_ingest = approve and scout_result and scout_result.recommended_root
            resume_at = "archivist" if can_ingest else "verify"
            initial = _initial_state(
                state.request,
                state.plan,
                policy,
                notes,
                "",
                pre_approved=True,
                librarian=state.librarian,
                lib_result=result_of(state.librarian, LibrarianResult),
                scout_result=scout_result,
            )
            # The specialists that ran before the pause are part of this turn's report.
            initial["results"].extend(r for r in (state.librarian, state.scout) if r)

            final = turn_graph(resume_at).invoke(initial)
            outcome = _outcome(final, turn_id, trace_id, state.plan)
            outcome.cost_usd = round(state.ledger.cost_usd, 6)
            _finish_span(span, outcome, state.ledger)
            return outcome


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------


def _ran(specialist: str, results: list[SpecialistResult]) -> SpecialistResult | None:
    """The result one specialist produced, for a turn being suspended part-way through."""
    for result in results:
        if result.specialist == specialist:
            return result
    return None


def _roll_up_session(sessions, session_id: str) -> None:
    """Compact the session once the raw turns pass the token budget.

    After the answer has been returned, so a user never waits on it, and failures are swallowed:
    a summary that could not be written costs context on the next turn, which is a smaller
    problem than failing a turn that already succeeded.
    """
    try:
        summary = sessions.roll_up(session_id)
    except Exception as exc:
        summary = ""
        telemetry.annotate(**{"session.summary_failed": f"{type(exc).__name__}: {exc}"})
    if summary:
        telemetry.annotate(**{telemetry.Attr.SESSION_SUMMARY: summary})


def _spend_notice(policy: Policy, notes: list[str]) -> str:
    """The daily cap degrades the agent in stages, with the reason visible in the answer."""
    cap = float(policy.budget.get("daily_usd_cap", 0) or 0)
    stage = DailySpend().degradation(cap)
    if not stage:
        return ""
    messages = {
        "write": "Daily spend cap nearly reached: this agent will not ingest new documentation "
        "until tomorrow.",
        "search": "Daily spend cap nearly reached: web search is paused; answers come from "
        "already-ingested documentation only.",
        "all": "Daily spend cap reached: answering from already-ingested documentation only.",
    }
    notes.append(f"daily spend cap: degraded to {stage}")
    return messages[stage]


def _budget_answer(exc: BudgetExceeded) -> FinalAnswer:
    return FinalAnswer(
        answer=(
            f"I ran out of this turn's budget ({exc.limit}) before I could finish answering. "
            f"What I did get to is incomplete, so I would rather tell you that than hand you a "
            f"half-checked answer. Ask again, or narrow the question."
        ),
        confidence=0.0,
        reasons=[str(exc)],
        caveats=["This turn hit its budget limit before finishing."],
    )


def _finish_span(span, outcome: TurnOutcome, ledger: Ledger) -> None:
    attrs: dict[str, Any] = {
        telemetry.Attr.TURN_COST_USD: round(ledger.cost_usd, 6),
        telemetry.Attr.TURN_TOOL_CALLS: ledger.tool_calls,
    }
    if outcome.answer:
        attrs.update(
            {
                # `confidence` on the turn span is the routing signal for the harness's
                # low-confidence labeling queue.
                telemetry.Attr.TURN_CONFIDENCE: outcome.answer.confidence,
                telemetry.Attr.TURN_ANSWER: outcome.answer.answer,
                telemetry.Attr.OUTPUT_VALUE: outcome.answer.answer,
                telemetry.Attr.TURN_CITATIONS: [
                    c.model_dump(mode="json") for c in outcome.answer.citations
                ],
                telemetry.Attr.AUDIT_REASONS: outcome.answer.reasons,
            }
        )
        if outcome.answer.notice:
            attrs[telemetry.Attr.TURN_NOTICE] = outcome.answer.notice
    if outcome.pending_approval:
        attrs[telemetry.Attr.POLICY_ACTION] = outcome.pending_approval.get("action", "")
        attrs[telemetry.Attr.POLICY_DECISION] = "pending"
    telemetry.set_attributes(span, attrs)
