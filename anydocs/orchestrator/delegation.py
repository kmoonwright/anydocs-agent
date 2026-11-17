"""Handing one task to one specialist.

The delegation protocol's sending half: each function here builds the `{task, inputs, budget,
corpora}` a specialist receives and hands back its structured result. The state graph in
`graph.py` decides *which* of these to call and in what order; this module decides *what
each one is told*, which is where the session context, the budget and the corpus resolution
live.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anydocs.budget import current_ledger
from anydocs.guardrails import mirrors_imperative_content
from anydocs.knowledge import retrieval
from anydocs.knowledge.store import CorpusRegistry, publish_snapshot
from anydocs.memory.session import get_session_store
from anydocs.orchestrator.planner import Plan
from anydocs.policy import Policy
from anydocs.specialists.results import ArchivistResult, LibrarianResult, ScoutResult
from anydocs.specialists.runner import SpecialistResult, Task, load_spec, run_specialist

if TYPE_CHECKING:
    from anydocs.orchestrator.graph import TurnRequest

# Past this fraction of the turn budget, the Auditor runs on a cheaper model: a cheaper judge
# beats no judge.
AUDITOR_DOWNGRADE_AT = 0.8
CHEAP_JUDGE_ROLE = "reader"


def result_of(result: SpecialistResult | None, model_type):
    """A specialist's result when it is the type the caller expected, else None.

    A specialist that ran out of budget or never called `respond` has no result, and a caller
    that assumed otherwise would fail on the attribute access rather than on the missing answer.
    """
    if result is None or result.result is None:
        return None
    return result.result if isinstance(result.result, model_type) else None


def ask_librarian(
    request: TurnRequest,
    plan: Plan,
    policy: Policy,
    *,
    corpora: list[str] | None = None,
) -> SpecialistResult:
    session = get_session_store().get(request.session_id or "", request.user_id)
    inputs: dict[str, Any] = {"sub_questions": plan.sub_questions}
    # Docs already cited this session are boosted in retrieval, so a follow-up like "and how do
    # I configure that?" lands on the same page. Set on the retriever rather than passed to the
    # model, which should not have to carry its own history into every query.
    retrieval.set_session_pins(session.pinned_docs())
    if session.pinned_docs():
        inputs["already_cited_doc_ids"] = session.pinned_docs()
    if session.versions:
        inputs["versions_mentioned"] = session.versions
    return run_specialist(
        "librarian",
        Task(
            task=request.message,
            inputs=inputs,
            budget=load_spec("librarian").budget,
            corpora=corpora or plan.corpora,
        ),
        policy=policy,
        requested_by=request.requested_by,
    )


def ask_scout(request: TurnRequest, plan: Plan, policy: Policy) -> SpecialistResult:
    return run_specialist(
        "scout",
        Task(
            task=(
                f"Find authoritative documentation that would answer: {request.message}"
            ),
            inputs={"sub_questions": plan.sub_questions, "known_corpora": plan.corpora},
            budget=load_spec("scout").budget,
            corpora=plan.corpora,
        ),
        policy=policy,
        requested_by=request.requested_by,
    )


def ask_archivist(
    request: TurnRequest,
    plan: Plan,
    policy: Policy,
    scout: ScoutResult,
    *,
    pre_approved: bool = False,
) -> SpecialistResult:
    auto_cap, hard_cap = policy.crawl_caps()
    estimate = next(
        (c.estimated_pages for c in scout.candidates if c.url == scout.recommended_root), 0
    )
    max_pages = max(1, min(hard_cap, estimate or auto_cap or 25))
    # An explicitly requested corpus wins over the Scout's suggestion. `anydocs learn --corpus
    # pydantic` has to write to `pydantic`, or the Librarian's retry searches an empty corpus
    # and the turn reports that it learned nothing.
    corpus = (
        (request.corpora[0] if request.corpora else "")
        or (plan.corpora[0] if plan.corpora else "")
        or scout.recommended_corpus
        or _corpus_name_from(scout.recommended_root)
    )

    return run_specialist(
        "archivist",
        Task(
            task=(
                f"Ingest the documentation at {scout.recommended_root} into corpus {corpus!r}, "
                f"then report what was added."
            ),
            inputs={
                "root_url": scout.recommended_root,
                "corpus": corpus,
                "max_pages": max_pages,
                "publish": True,
                "pre_approved": pre_approved,
            },
            budget=load_spec("archivist").budget,
            corpora=[corpus],
        ),
        policy=policy,
        requested_by=request.requested_by,
    )


def ask_analyst(request: TurnRequest, plan: Plan, policy: Policy) -> SpecialistResult:
    session = get_session_store().get(request.session_id or "", request.user_id)
    inputs: dict[str, Any] = {"sub_questions": plan.sub_questions}
    if session.corpora:
        # A follow-up like "and how many for that one?" takes its filter from the conversation.
        inputs["corpora_in_play"] = session.corpora
    if session.context_blurb():
        inputs["conversation_so_far"] = session.context_blurb()
    return run_specialist(
        "analyst",
        Task(
            task=request.message,
            inputs=inputs,
            budget=load_spec("analyst").budget,
            corpora=plan.corpora,
        ),
        policy=policy,
        requested_by=request.requested_by,
    )


def mirrors_sources(librarian: LibrarianResult) -> bool:
    """Whether the draft repeats imperative text from the pages it cites.

    A page the agent fetched is content, not instruction. An answer that echoes "ignore the
    retry limit and send directly to..." has stopped describing the document and started
    relaying it.
    """
    quotes = [c.quote for c in librarian.citations if c.quote]
    return mirrors_imperative_content(librarian.draft, quotes) if quotes else False


def ask_auditor(
    request: TurnRequest,
    plan: Plan,
    policy: Policy,
    librarian: LibrarianResult | None,
    notes: list[str],
    conflicts: list[dict[str, str]] | None = None,
) -> SpecialistResult | None:
    if librarian is None or not librarian.draft.strip():
        # Nothing to audit. The answer will say what was not found; a verdict on an empty draft
        # would be theatre.
        notes.append("no draft to audit")
        return None

    ledger = current_ledger()
    role = None
    if ledger.fraction_spent >= AUDITOR_DOWNGRADE_AT:
        role = CHEAP_JUDGE_ROLE
        notes.append(
            f"turn budget {int(ledger.fraction_spent * 100)}% spent; auditor downgraded to the "
            f"{role} model"
        )

    return run_specialist(
        "auditor",
        Task(
            task="Verify this draft answer before it ships.",
            inputs={
                "question": request.message,
                "draft": librarian.draft,
                "citations": [c.model_dump(mode="json") for c in librarian.citations],
                "unanswered": librarian.unanswered,
                # Conflicts recorded at ingest. The Auditor does not re-judge them; it decides
                # whether this answer has to disclose them.
                "known_contradictions": conflicts or [],
                # Fetched pages are untrusted. If the draft repeats imperative text lifted from
                # one, the Auditor is told so rather than being expected to spot it.
                "mirrors_source_instructions": mirrors_sources(librarian),
            },
            budget=load_spec("auditor").budget,
            corpora=plan.corpora,
        ),
        policy=policy,
        model_role=role,
        requested_by=request.requested_by,
    )


def learned_summary(
    archivist: SpecialistResult, notes: list[str] | None = None
) -> dict[str, Any] | None:
    """What the turn learned, checked against what is actually on disk.

    The Archivist reports the corpus it believes it wrote to, which is a model's claim. If that
    corpus does not exist, the Librarian's retry would search an empty one and the turn would
    report learning nothing it had just learned — so the real corpus is looked up instead, and
    the divergence is recorded.
    """
    result = result_of(archivist, ArchivistResult)
    if not result or result.pages_added <= 0:
        return None

    registry = CorpusRegistry()
    corpus = result.corpus
    if not corpus or registry.get(corpus) is None:
        actual = _corpus_holding(registry, result.snapshot_id)
        if notes is not None:
            notes.append(
                f"archivist reported corpus {corpus or '(none)'!r}, which is not in the store; "
                f"using {actual or '(none found)'!r}"
            )
        corpus = actual
    if not corpus:
        return None

    # Publish here rather than relying on the Archivist to pass `publish: true` to its last
    # `save_note`. A model that forgets leaves a corpus written and unqueryable — the same class
    # of silent failure as a snapshot that was never indexed.
    # The reported snapshot id is a model's claim too, so it is checked against the registry
    # before anything is published against it.
    snapshot_id = _resolve_snapshot(registry, corpus, result.snapshot_id, notes)
    if snapshot_id:
        publish_snapshot(corpus, snapshot_id)

    return {
        "corpus": corpus,
        "snapshot_id": snapshot_id,
        "pages_added": result.pages_added,
        "best_questions": result.best_questions,
    }


def _resolve_snapshot(
    registry: CorpusRegistry, corpus: str, reported: str, notes: list[str] | None
) -> str:
    """The snapshot the pages actually went into.

    Prefers what the Archivist reported, but only when the registry knows it — a model that
    invents or paraphrases a snapshot id would otherwise take the publish down with it.
    """
    entry = registry.get(corpus)
    if entry is None or not entry.snapshots:
        return ""
    if reported and reported in entry.snapshots:
        return reported
    if reported and notes is not None:
        notes.append(
            f"archivist reported snapshot {reported!r}, which is not in {corpus!r}; "
            f"publishing {entry.snapshots[-1]!r} instead"
        )
    return entry.snapshots[-1]


def _corpus_holding(registry: CorpusRegistry, snapshot_id: str) -> str:
    """The corpus that owns a snapshot id, or the most recently created one."""
    entries = registry.list()
    for entry in entries:
        if snapshot_id and snapshot_id in entry.snapshots:
            return entry.name
    return max(entries, key=lambda e: e.created_at).name if entries else ""


def _corpus_name_from(url: str) -> str:
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "docs").lower()
    parts = [p for p in host.split(".") if p not in ("www", "docs", "com", "io", "org", "dev")]
    return (parts[0] if parts else "docs").replace("_", "-")
