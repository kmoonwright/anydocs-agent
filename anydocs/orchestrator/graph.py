"""The turn, as a LangGraph state graph.

`turn.py` keeps the envelope — the ledger, the session, the turn span, the suspension — and this
module holds the flow inside it: workflow 1 of the design doc. Ask the Librarian; if retrieval
confidence is below threshold, ask the Scout for sources, let the policy approve or decline the
crawl, let the Archivist ingest, and ask the Librarian again against the new snapshot. **One
learning loop per turn at most** — the retry is its own node, not a cycle, so a second miss
cannot happen by construction.

Writing the flow as a graph is also what lets a resumed turn re-enter it rather than restate it:
`/invoke/approve` starts at the `archivist` node with the approval already granted, and the tail
— resolve citations, read contradictions, audit, synthesize — is the same nodes either way.

`NeedsApproval` is left to propagate out of `invoke`. It is not a failure the flow routes around;
it suspends the turn, and only the caller holding the pending store can do that.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from anydocs import telemetry
from anydocs.knowledge import links
from anydocs.knowledge.store import StoreError, open_active
from anydocs.orchestrator import citations
from anydocs.orchestrator.delegation import (
    ask_analyst,
    ask_archivist,
    ask_auditor,
    ask_librarian,
    ask_scout,
    learned_summary,
    result_of,
)
from anydocs.orchestrator.planner import Plan
from anydocs.orchestrator.synthesizer import FinalAnswer, synthesize
from anydocs.policy import Policy
from anydocs.specialists.results import (
    AnalystResult,
    AuditorResult,
    LibrarianResult,
    ScoutResult,
)
from anydocs.specialists.runner import SpecialistResult

# Below this, the orchestrator is allowed to go learn.
LEARN_THRESHOLD = 0.55


EventSink = Callable[[str, dict[str, Any]], None] | None


@dataclass
class TurnRequest:
    message: str
    session_id: str = ""
    user_id: str = ""
    corpora: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    maintenance: bool = False
    # Whether this turn may go and learn. Separate from `maintenance`: workflow 2 (`/learn`) is
    # headless *and* exists to learn, while the freshness workflow is headless and must not
    # start a fresh crawl mid-turn.
    allow_learning: bool = True
    requested_by: str = "user"
    # Called as the turn progresses, for the SSE route. A streaming turn and a non-streaming one
    # run the same code; only what happens to the events differs.
    on_event: EventSink = None


class TurnState(TypedDict, total=False):
    """What the flow carries between nodes.

    `notes` and `results` are mutated in place rather than returned as updates. The delegation
    functions already take the notes list and append to it — a specialist's "the crawl was
    blocked" has to reach the answer from several frames down — so threading them through
    channel updates would mean diffing a list the callee already wrote to.
    """

    request: TurnRequest
    plan: Plan
    policy: Policy
    notes: list[str]
    results: list[SpecialistResult]
    notice: str
    pre_approved: bool

    librarian: SpecialistResult | None
    lib_result: LibrarianResult | None
    scout_result: ScoutResult | None
    learned: dict[str, Any] | None
    conflicts: list[dict[str, str]]
    audit_result: AuditorResult | None
    answer: FinalAnswer | None


# ----------------------------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------------------------


def _analyst(state: TurnState) -> dict[str, Any]:
    """A question about the documentation itself rather than its content.

    The Analyst answers from the analytics store; there is nothing for the Librarian to retrieve
    and nothing for the Auditor to verify against a page, so a result ends the turn here.
    """
    analyst = _run(
        state, ask_analyst(state["request"], state["plan"], state["policy"])
    )
    result = result_of(analyst, AnalystResult)
    if result is None:
        state["notes"].append(
            "the analyst returned nothing; falling back to the documentation itself"
        )
        return {}
    return {"answer": _numeric_answer(result, state.get("notice", ""))}


def _librarian(state: TurnState) -> dict[str, Any]:
    librarian = _run(
        state, ask_librarian(state["request"], state["plan"], state["policy"])
    )
    return {"librarian": librarian, "lib_result": result_of(librarian, LibrarianResult)}


def _scout(state: TurnState) -> dict[str, Any]:
    scout = _run(state, ask_scout(state["request"], state["plan"], state["policy"]))
    result = result_of(scout, ScoutResult)
    if result and not result.recommended_root:
        state["notes"].append("scout found no authoritative documentation to ingest")
    return {"scout_result": result}


def _archivist(state: TurnState) -> dict[str, Any]:
    """Ingest what the Scout recommended. Raises `NeedsApproval` when the policy says to ask."""
    archivist = _run(
        state,
        ask_archivist(
            state["request"],
            state["plan"],
            state["policy"],
            state["scout_result"],
            pre_approved=state.get("pre_approved", False),
        ),
    )
    learned = learned_summary(archivist, state["notes"])
    if learned:
        emit(state["request"].on_event, "learned", learned)
    return {"learned": learned}


def _relibrarian(state: TurnState) -> dict[str, Any]:
    """The one learning loop: retry retrieval against the snapshot just written.

    Kept only if it beat the first pass. A crawl that ingested the wrong site must not replace a
    weak-but-real answer with a worse one.
    """
    learned = state["learned"] or {}
    retry = _run(
        state,
        ask_librarian(
            state["request"],
            state["plan"],
            state["policy"],
            corpora=[learned["corpus"]],
        ),
    )
    retry_result = result_of(retry, LibrarianResult)
    first = state.get("lib_result")
    if retry_result and (
        first is None or retry_result.retrieval_confidence > first.retrieval_confidence
    ):
        return {"librarian": retry, "lib_result": retry_result}
    return {}


def _verify(state: TurnState) -> dict[str, Any]:
    """Resolve the citations against the store, read any recorded conflicts, then audit."""
    notes = state["notes"]
    plan, lib_result = state["plan"], state.get("lib_result")

    if lib_result is not None:
        lib_result = _with_resolved_citations(lib_result, plan.corpora, notes)

    conflicts = _contradictions_among(lib_result, plan.corpora)
    if conflicts:
        notes.append(f"sources disagree: {len(conflicts)} recorded contradiction(s)")
        telemetry.annotate(**{"answer.contradictions": len(conflicts)})

    auditor = ask_auditor(state["request"], plan, state["policy"], lib_result, notes, conflicts)
    if auditor:
        _run(state, auditor)
    return {
        "lib_result": lib_result,
        "conflicts": conflicts,
        "audit_result": result_of(auditor, AuditorResult) if auditor else None,
    }


def _synthesize(state: TurnState) -> dict[str, Any]:
    results = state["results"]
    answer = synthesize(
        state["request"].message,
        state.get("lib_result"),
        state.get("audit_result"),
        learned=state.get("learned"),
        notes=state["notes"] + [n for r in results for n in r.notes],
        notice=state.get("notice", ""),
        contradictions=state.get("conflicts") or [],
    )
    if state.get("learned"):
        telemetry.annotate(**{telemetry.Attr.TURN_LEARNED: state["learned"]})
    return {"answer": answer}


# ----------------------------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------------------------


def _entry(state: TurnState) -> str:
    plan = state["plan"]
    return "analyst" if plan.is_numeric and "analyst" in plan.specialists else "librarian"


def _after_analyst(state: TurnState) -> str:
    return END if state.get("answer") else "librarian"


def _after_librarian(state: TurnState) -> str:
    """Whether to go and learn. A Librarian that returned nothing at all counts as a miss."""
    plan, lib_result, request = state["plan"], state.get("lib_result"), state["request"]
    may_learn = (
        lib_result is None
        or plan.needs_learning
        or lib_result.retrieval_confidence < LEARN_THRESHOLD
    )
    if may_learn and request.allow_learning and state["policy"].mode("crawl") != "off":
        return "scout"
    return "verify"


def _after_scout(state: TurnState) -> str:
    result = state.get("scout_result")
    return "archivist" if result and result.recommended_root else "verify"


def _after_archivist(state: TurnState) -> str:
    return "relibrarian" if state.get("learned") else "verify"


@lru_cache(maxsize=4)
def turn_graph(entry: str = "route"):
    """The compiled flow. `entry` is how a resumed turn re-enters it part-way through.

    Cached because compiling is pure: the graph is the same every turn, and only the state
    passed through it differs.
    """
    builder = StateGraph(TurnState)
    builder.add_node("analyst", _analyst)
    builder.add_node("librarian", _librarian)
    builder.add_node("scout", _scout)
    builder.add_node("archivist", _archivist)
    builder.add_node("relibrarian", _relibrarian)
    builder.add_node("verify", _verify)
    builder.add_node("synthesize", _synthesize)

    if entry == "route":
        builder.set_conditional_entry_point(
            _entry, {"analyst": "analyst", "librarian": "librarian"}
        )
    else:
        builder.set_entry_point(entry)

    builder.add_conditional_edges("analyst", _after_analyst, {END: END, "librarian": "librarian"})
    builder.add_conditional_edges(
        "librarian", _after_librarian, {"scout": "scout", "verify": "verify"}
    )
    builder.add_conditional_edges(
        "scout", _after_scout, {"archivist": "archivist", "verify": "verify"}
    )
    builder.add_conditional_edges(
        "archivist", _after_archivist, {"relibrarian": "relibrarian", "verify": "verify"}
    )
    builder.add_edge("relibrarian", "verify")
    builder.add_edge("verify", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------


def _run(state: TurnState, result: SpecialistResult) -> SpecialistResult:
    """Record a specialist's result and stream it. Every node routes its result through here."""
    state["results"].append(result)
    emit(state["request"].on_event, "specialist", result.as_dict())
    return result


def emit(sink: EventSink, event: str, payload: dict[str, Any]) -> None:
    """Stream one event, without letting a broken consumer fail the turn.

    A client that disconnected mid-stream must not take the answer down with it — the turn is
    still worth finishing, and its trace is still worth having.

    Takes the sink rather than the state because `turn.py` emits the plan from outside the graph:
    planning happens in the envelope, so that one event has no state to read it from.
    """
    if sink is None:
        return
    try:
        sink(event, payload)
    except Exception as exc:
        telemetry.annotate(**{"stream.emit_failed": f"{type(exc).__name__}: {exc}"})


def _numeric_answer(result: AnalystResult, notice: str) -> FinalAnswer:
    """The Analyst's answer, used as written.

    No synthesis pass: the Analyst already formatted the number in human units and stated its
    assumptions, and re-wording a number through another model is how a digit changes.
    """
    caveats = list(result.assumptions)
    if not result.rows:
        caveats.append("The analytics store returned no rows for this question.")
    return FinalAnswer(
        answer=result.answer,
        citations=[],
        confidence=0.8 if result.rows else 0.2,
        reasons=[f"Answered from the analytics store with: {result.query}"]
        if result.query
        else ["The analytics store was queried but returned no SQL."],
        caveats=caveats,
        notice=notice,
    )


def _contradictions_among(
    librarian: LibrarianResult | None, corpora: list[str]
) -> list[dict[str, str]]:
    """Conflicts the Archivist recorded between the docs this answer cites.

    Read, not judged: the comparison was made and quoted at ingest, so the answer path only has
    to decide whether to disclose it.
    """
    if librarian is None or not librarian.citations:
        return []
    doc_ids = [c.doc_id for c in librarian.citations if c.doc_id]
    if not doc_ids:
        return []
    try:
        return links.contradictions_for(open_active(corpora or None), doc_ids)
    except StoreError:
        return []


def _with_resolved_citations(
    librarian: LibrarianResult, corpora: list[str], notes: list[str]
) -> LibrarianResult:
    """Replace the model's claims about each source with what the store actually holds.

    A citation whose doc id is not in the store is dropped and reported: rendering it would put
    an invented source in front of a user, and the Auditor would have nothing to verify against.
    """
    resolved, unresolvable = citations.resolve(librarian.citations, corpora=corpora or None)
    if unresolvable:
        ids = ", ".join(c.doc_id or "(no doc id)" for c in unresolvable)
        notes.append(f"dropped {len(unresolvable)} citation(s) not found in the store: {ids}")
    return librarian.model_copy(update={"citations": resolved})
