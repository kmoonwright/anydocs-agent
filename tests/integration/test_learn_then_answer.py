"""Step 5's acceptance test: an unknown product triggers an approved crawl and a second-pass
answer, all in one turn."""

from __future__ import annotations

import pytest

from anydocs import telemetry
from anydocs.orchestrator.turn import TurnRequest, pending_store, resume_turn, run_turn
from anydocs.telemetry import Attr

QUESTION = "In Lumen, what does bucket_by default to for a percentage rollout?"
ROOT = "https://lumen.example.dev/docs/"

PLAN = (
    "plan",
    {
        "specialists": ["librarian", "scout", "archivist", "auditor"],
        "corpora": [],
        "intent_tools": ["search_docs", "web_search", "crawl_site", "save_note", "get_doc"],
        "needs_learning": True,
        "is_numeric": False,
        "sub_questions": ["what is bucket_by", "what is its default"],
        "rationale": "No corpus covers Lumen, so the store cannot answer this yet.",
    },
)

SCOUT_SEARCH = ("web_search", {"query": "Lumen feature flags documentation", "k": 5})
SCOUT_RESPOND = (
    "respond",
    {
        "candidates": [
            {
                "url": ROOT,
                "title": "Lumen — Documentation",
                "kind": "docs",
                "reason": "The vendor's own documentation root, with a nav tree over the SDK.",
                "estimated_pages": 3,
                "authority": 0.85,
            },
            {
                "url": "https://blog.example.com/lumen",
                "title": "Lumen tutorial",
                "kind": "blog",
                "reason": "A third-party tutorial, not authoritative.",
                "estimated_pages": 1,
                "authority": 0.2,
            },
        ],
        "recommended_root": ROOT,
        "recommended_corpus": "lumen",
        "notes": "Three doc pages visible from the nav.",
    },
)

EMPTY_DRAFT = (
    "respond",
    {
        "draft": "",
        "citations": [],
        "retrieval_confidence": 0.05,
        "unanswered": [QUESTION],
        "corpora_searched": [],
    },
)

GOOD_DRAFT = (
    "respond",
    {
        "draft": "`bucket_by` defaults to `user_id` for a percentage rollout [1].",
        "citations": [
            {
                "doc_id": "lumen-doc",
                "url": "https://lumen.example.dev/docs/rollouts",
                "heading": "Rollouts",
                "corpus": "lumen",
                "quote": "bucket_by — the context attribute to hash. Defaults to user_id.",
            }
        ],
        "retrieval_confidence": 0.86,
        "unanswered": [],
        "corpora_searched": ["lumen"],
    },
)

ARCHIVIST = (
    "respond",
    {
        "corpus": "lumen",
        "snapshot_id": "set-by-the-crawl",
        "pages_added": 3,
        "pages_unchanged": 0,
        "pages_failed": 0,
        "pages_blocked": 1,
        "average_digest_quality": 0.7,
        "best_questions": [
            "What does bucket_by default to for a percentage rollout?",
            "How often does the SDK poll for ruleset changes?",
            "What is the maximum length of a flag key?",
        ],
        "notes": "robots.txt kept the crawler out of /marketing/.",
    },
)

ASK_MODE = {"policy": {"crawl": {"mode": "ask"}}}

AUDIT_PASS = (
    "respond",
    {
        "verdict": "pass",
        "claims": [
            {
                "claim": "bucket_by defaults to user_id.",
                "verdict": "supported",
                "evidence": "bucket_by — the context attribute to hash. Defaults to user_id.",
                "doc_id": "lumen-doc",
            }
        ],
        "confidence": 0.88,
        "reasons": ["The claim is quoted verbatim from the cited page."],
    },
)


def _report_what_was_saved(messages):
    """Report the corpus and snapshot the saves actually landed in."""
    import json

    corpus, snapshot_id = "lumen", ""
    for message in reversed(messages):
        if message.type == "tool":
            try:
                payload = json.loads(message.content)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and "provenance_id" in payload:
                corpus = payload.get("corpus", corpus)
                snapshot_id = payload.get("snapshot_id", "")
                break
    return [("respond", {**ARCHIVIST[1], "corpus": corpus, "snapshot_id": snapshot_id})]


def _save_every_crawled_page(messages):
    """Turn the crawl manifest into one `save_note` call per page.

    A real Archivist reads the manifest from the crawl_site result and hands each page's `path`
    to save_note; this does the same thing deterministically.
    """
    import json

    manifest = {}
    for message in reversed(messages):
        if message.type == "tool":
            try:
                payload = json.loads(message.content)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and "pages" in payload:
                manifest = payload
                break
    digest = {
        "purpose": "Lumen rollout and flag reference.",
        "facts": [
            {
                "claim": "bucket_by defaults to user_id.",
                "source_span": "bucket_by - the context attribute to hash. Defaults to user_id.",
            }
        ],
        "answers": [
            {"question": "What does bucket_by default to?", "answer": "user_id"},
            {"question": "How often does the SDK poll?", "answer": "every 30 seconds"},
            {"question": "How long can a flag key be?", "answer": "128 characters"},
        ],
    }
    return [
        (
            "save_note",
            {
                "corpus": manifest.get("corpus", "lumen"),
                "snapshot_id": manifest.get("snapshot_id", ""),
                "page_path": page["path"],
                "digest": digest,
                "publish": True,
            },
        )
        for page in manifest.get("pages", [])
    ]


@pytest.fixture
def learning_scripts(scripted, lumen_site, data_dir, monkeypatch):
    monkeypatch.setenv("ANYDOCS_RERANKER", "noop")
    from anydocs.knowledge.embeddings import reset_embeddings

    reset_embeddings()
    pending_store().clear()

    def install(crawl_attempts: int = 1):
        """`crawl_attempts` is how many times the Archivist is expected to reach for
        `crawl_site`. A turn that pauses for approval and then resumes reaches for it twice —
        once in the pass that was gated, once in the pass that was approved — because the
        Archivist is re-run from the start on resume."""
        def crawl_call(messages):
            """Read the corpus out of the task, the way a real Archivist does."""
            corpus = "lumen"
            for message in messages:
                if message.type == "human" and '"corpus"' in message.content:
                    import json as _json
                    import re as _re

                    match = _re.search(r'"corpus":\s*"([^"]+)"', message.content)
                    if match:
                        corpus = match.group(1)
                    del _json
                    break
            return [
                (
                    "crawl_site",
                    {"root_url": ROOT, "corpus": corpus, "max_pages": 3, "max_depth": 2},
                )
            ]
        archivist_script = [crawl_call for _ in range(crawl_attempts)] + [
            _save_every_crawled_page,
            _report_what_was_saved,
        ]
        return scripted(
            {
                # The planner and the Scout both run on `router`, in that order.
                "router": [[PLAN], [SCOUT_SEARCH], [SCOUT_RESPOND]],
                "reader": [[EMPTY_DRAFT], [GOOD_DRAFT]],
                "summarizer": archivist_script,
                "judge": [[AUDIT_PASS]],
                "synthesizer": [
                    "`bucket_by` defaults to `user_id` for a percentage rollout [1]. I had to "
                    "read the Lumen docs first — three pages are now ingested."
                ],
            }
        )

    yield install
    reset_embeddings()
    pending_store().clear()


def test_the_whole_loop_runs_in_one_trace_when_crawl_is_auto(spans, learning_scripts):
    learning_scripts()
    outcome = run_turn(
        TurnRequest(
            message=QUESTION,
            session_id="s1",
            user_id="u1",
            overrides={"policy": {"crawl": {"mode": "auto"}}},
        )
    )

    assert outcome.answer is not None
    assert outcome.answer.confidence == 0.88
    assert "bucket_by" in outcome.answer.answer

    # The trace reads as the doc's short story: asked, looked, found, checked, answered.
    assert spans.path("crawl_site") == [telemetry.TURN, "archivist", "crawl_site"]
    for name in ("orchestrator.plan", "librarian", "scout", "archivist", "auditor"):
        assert spans.parent_name(name) == telemetry.TURN, name
    assert spans.has(telemetry.SYNTHESIZE)
    # The Librarian ran twice: once against an empty store, once against the new snapshot.
    assert len(spans.find_all("librarian")) == 2
    assert spans.attr(telemetry.TURN, Attr.TURN_LEARNED)
    assert spans.attr(telemetry.TURN, Attr.TURN_CONFIDENCE) == 0.88


def test_ask_mode_suspends_the_turn_with_a_cost_estimate(spans, learning_scripts):
    learning_scripts()
    outcome = run_turn(
        TurnRequest(
            message=QUESTION, session_id="s1", overrides=ASK_MODE
        )
    )

    assert outcome.answer is None
    pending = outcome.pending_approval
    assert pending["action"] == "crawl_site"
    assert pending["resume_token"]
    assert pending["estimated_pages"] == 3
    assert pending["estimated_usd"] > 0

    assert spans.has(telemetry.APPROVAL_PENDING)
    assert spans.attr(telemetry.TURN, Attr.POLICY_DECISION) == "pending"
    # Nothing was written: the crawl was gated before it ran.
    assert spans.attr("crawl_site", Attr.TOOL_OUTCOME) == "needs_approval"
    assert not spans.has("save_note")


def test_approving_continues_the_turn_rather_than_restarting_it(spans, learning_scripts):
    learning_scripts(crawl_attempts=2)
    paused = run_turn(
        TurnRequest(
            message=QUESTION, session_id="s1", overrides=ASK_MODE
        )
    )
    token = paused.pending_approval["resume_token"]

    # The Scout and the first Librarian pass are not re-run, so their scripts stay consumed.
    resumed = resume_turn(token, approve=True, note="looks like the real docs")

    assert resumed.answer is not None
    assert resumed.answer.confidence == 0.88
    assert spans.attr(telemetry.APPROVAL_DECISION, Attr.POLICY_DECISION) == "approved"
    # The first crawl_site span was gated; the second actually ran.
    outcomes = [s.attributes.get(Attr.TOOL_OUTCOME) for s in spans.find_all("crawl_site")]
    assert outcomes == ["needs_approval", "ran"]
    assert spans.has("save_note")
    # The Scout ran once, in the paused pass, and was not re-run on resume.
    assert len(spans.find_all("web_search")) == 1


def test_declining_ingests_nothing_and_says_so(spans, learning_scripts):
    learning_scripts()
    paused = run_turn(
        TurnRequest(
            message=QUESTION, session_id="s1", overrides=ASK_MODE
        )
    )
    resumed = resume_turn(
        paused.pending_approval["resume_token"], approve=False, note="not now"
    )

    assert resumed.answer is not None
    # The one crawl_site span is the gated attempt, and it says so; nothing was written.
    outcomes = [s.attributes.get(Attr.TOOL_OUTCOME) for s in spans.find_all("crawl_site")]
    assert outcomes == ["needs_approval"]
    assert not spans.has("save_note")
    assert spans.attr(telemetry.APPROVAL_DECISION, Attr.POLICY_DECISION) == "declined"


def test_a_resume_token_cannot_be_used_twice(spans, learning_scripts):
    learning_scripts()
    paused = run_turn(
        TurnRequest(
            message=QUESTION, session_id="s1", overrides=ASK_MODE
        )
    )
    token = paused.pending_approval["resume_token"]
    resume_turn(token, approve=False)
    from anydocs.policy import PolicyDenied

    with pytest.raises(PolicyDenied, match="no suspended turn"):
        resume_turn(token, approve=False)


def test_crawl_off_means_the_turn_answers_from_the_store_only(
    spans, scripted, data_dir, monkeypatch
):
    monkeypatch.setenv("ANYDOCS_RERANKER", "noop")
    scripted(
        {
            "router": [[PLAN]],
            "reader": [[EMPTY_DRAFT]],
            "synthesizer": ["I do not have any documentation for Lumen ingested yet."],
        }
    )
    outcome = run_turn(
        TurnRequest(message=QUESTION, overrides={"policy": {"crawl": {"mode": "off"}}})
    )
    assert outcome.answer is not None
    assert not spans.has("scout")
    assert not spans.has("archivist")
    # No draft means no audit: a verdict on an empty draft would be theatre.
    assert not spans.has("auditor")


def test_the_learned_pages_are_really_in_the_store(spans, learning_scripts):
    """Not just that the loop ran: that the second pass had something new to search."""
    from anydocs.knowledge import retrieval
    from anydocs.knowledge.store import CorpusRegistry, open_snapshot

    learning_scripts()
    run_turn(
        TurnRequest(
            message=QUESTION, session_id="s1", overrides={"policy": {"crawl": {"mode": "auto"}}}
        )
    )

    assert CorpusRegistry().get("lumen") is not None
    snapshot = open_snapshot("lumen")
    assert len(snapshot.live_docs()) == 3
    assert len(snapshot.read_jsonl(snapshot.provenance_path)) == 3

    hits = retrieval.search(["bucket_by default"], corpora=["lumen"], k=5).hits
    assert hits and any("rollouts" in h["url"] for h in hits)


def test_save_note_refuses_a_page_path_outside_the_knowledge_store(spans, data_dir):
    """The path reaching these tools came from a model, so it is untrusted input."""
    from anydocs.tools.registry import ToolError, invoke

    with pytest.raises(ToolError, match="outside the knowledge store"):
        invoke(
            "save_note",
            {
                "corpus": "lumen",
                "page_path": "/etc/passwd",
                "digest": {"purpose": "x"},
            },
        )


def test_digest_page_refuses_a_traversal_path(spans, data_dir):
    from anydocs.tools.registry import ToolError, invoke

    with pytest.raises(ToolError, match="outside the knowledge store"):
        invoke("digest_page", {"page_path": str(data_dir / ".." / ".." / "etc" / "hosts")})


def test_save_note_without_content_or_a_path_is_a_tool_error(spans, data_dir):
    from anydocs.tools.registry import ToolError, invoke

    with pytest.raises(ToolError, match="needs either"):
        invoke(
            "save_note",
            {"corpus": "x", "url": "https://x.example/", "digest": {"purpose": "p"}},
        )


def test_the_learn_route_is_headless_but_still_learns(spans, learning_scripts):
    """`/learn` is traced as maintenance and must still crawl — it exists to. Only the freshness
    workflow is headless *and* barred from starting a crawl."""
    learning_scripts()
    outcome = run_turn(
        TurnRequest(
            message=f"Learn the documentation at {ROOT}.",
            session_id="learn-1",
            user_id="system",
            maintenance=True,
            allow_learning=True,
            overrides={"policy": {"crawl": {"mode": "auto"}}},
        )
    )
    assert outcome.answer is not None
    assert spans.has(telemetry.MAINTENANCE)
    assert spans.has("crawl_site")
    assert spans.has("save_note")


def test_a_maintenance_turn_with_learning_off_never_crawls(spans, scripted, data_dir, monkeypatch):
    """The freshness workflow re-digests what it already has; it does not go shopping."""
    monkeypatch.setenv("ANYDOCS_RERANKER", "noop")
    scripted(
        {
            "router": [[PLAN]],
            "reader": [[EMPTY_DRAFT]],
            "synthesizer": ["Nothing ingested for Lumen."],
        }
    )
    outcome = run_turn(
        TurnRequest(
            message=QUESTION,
            maintenance=True,
            allow_learning=False,
            overrides={"policy": {"crawl": {"mode": "auto"}}},
        )
    )
    assert outcome.answer is not None
    assert not spans.has("scout")
    assert not spans.has("crawl_site")


def test_a_suspended_turn_survives_the_process_that_paused_it(spans, learning_scripts, data_dir):
    """`anydocs approve` runs in a different process from the `anydocs ask` that paused. An
    in-memory pending store would make that command impossible to use."""
    from anydocs.orchestrator.suspension import reset_pending_store

    learning_scripts(crawl_attempts=2)
    paused = run_turn(
        TurnRequest(message=QUESTION, session_id="s1", overrides=ASK_MODE)
    )
    token = paused.pending_approval["resume_token"]

    # Stand in for a fresh process: drop every in-memory handle to the suspended turn.
    reset_pending_store()

    resumed = resume_turn(token, approve=True)
    assert resumed.answer is not None
    assert resumed.answer.confidence == 0.88
    assert spans.has("save_note")


def test_the_pending_entry_is_consumed_so_a_token_works_once(spans, learning_scripts, data_dir):
    from anydocs.policy import PolicyDenied

    learning_scripts(crawl_attempts=2)
    paused = run_turn(
        TurnRequest(message=QUESTION, session_id="s1", overrides=ASK_MODE)
    )
    token = paused.pending_approval["resume_token"]
    resume_turn(token, approve=True)
    with pytest.raises(PolicyDenied, match="no suspended turn"):
        resume_turn(token, approve=True)


def test_the_resumed_turn_keeps_what_the_paused_one_already_spent(
    spans, learning_scripts, data_dir
):
    """Cost and tool calls carry across the pause; the wall clock does not, because the elapsed
    time was a human reading an approval prompt."""
    learning_scripts(crawl_attempts=2)
    paused = run_turn(
        TurnRequest(message=QUESTION, session_id="s1", overrides=ASK_MODE)
    )
    resumed = resume_turn(paused.pending_approval["resume_token"], approve=True)
    assert resumed.cost_usd >= paused.cost_usd


def test_an_explicitly_requested_corpus_beats_the_scouts_suggestion(spans, learning_scripts):
    """`--corpus X` must write to X. Otherwise the Librarian's retry searches an empty corpus
    and the turn reports that it learned nothing it just learned."""
    from anydocs.knowledge.store import CorpusRegistry

    learning_scripts()
    run_turn(
        TurnRequest(
            message=QUESTION,
            session_id="s1",
            # The Scout's script recommends the corpus name "lumen"; the caller asked for this.
            corpora=["feature-flags"],
            overrides={"policy": {"crawl": {"mode": "auto"}}},
        )
    )
    names = {e.name for e in CorpusRegistry().list()}
    assert "feature-flags" in names
    assert "lumen" not in names


def test_a_misreported_corpus_falls_back_to_what_is_on_disk(
    spans, scripted, lumen_site, data_dir, monkeypatch
):
    """The Archivist's reported corpus is a model's claim. If it names one that does not exist,
    the retry must still search the corpus the pages actually landed in."""
    monkeypatch.setenv("ANYDOCS_RERANKER", "noop")
    from anydocs.knowledge.embeddings import reset_embeddings

    reset_embeddings()
    try:
        scripted(
            {
                "router": [[PLAN], [SCOUT_SEARCH], [SCOUT_RESPOND]],
                "reader": [[EMPTY_DRAFT], [GOOD_DRAFT]],
                "summarizer": [
                    [
                        (
                            "crawl_site",
                            {
                                "root_url": ROOT,
                                "corpus": "lumen",
                                "max_pages": 3,
                                "max_depth": 2,
                            },
                        )
                    ],
                    _save_every_crawled_page,
                    # Reports a corpus that was never created.
                    [("respond", {**ARCHIVIST[1], "corpus": "not-a-real-corpus"})],
                ],
                "judge": [[AUDIT_PASS]],
                "synthesizer": ["bucket_by defaults to user_id [1]."],
            }
        )
        outcome = run_turn(
            TurnRequest(message=QUESTION, overrides={"policy": {"crawl": {"mode": "auto"}}})
        )
        assert outcome.answer is not None
        # The retry still ran, against the corpus the saves actually created.
        assert len(spans.find_all("librarian")) == 2
        notes = [n for r in outcome.specialists for n in r["notes"]]
        assert any("not in the store" in n for n in notes) or outcome.answer.confidence == 0.88
    finally:
        reset_embeddings()
