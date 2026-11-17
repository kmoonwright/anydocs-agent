"""Step 8's acceptance test: a numeric follow-up lands on the Analyst with the right filters
taken from conversation context."""

from __future__ import annotations

import pytest

from anydocs import telemetry
from anydocs.memory.profile import get_profile_store, infer_from_turn, reset_profile_store
from anydocs.memory.session import SUMMARY_TOKEN_BUDGET, Turn, get_session_store
from anydocs.orchestrator.turn import TurnRequest, run_turn
from anydocs.telemetry import Attr


@pytest.fixture(autouse=True)
def _clean_memory(data_dir):
    get_session_store().clear()
    reset_profile_store()
    yield
    get_session_store().clear()
    reset_profile_store()


NUMERIC_PLAN = (
    "plan",
    {
        "specialists": ["analyst"],
        "corpora": ["relayr"],
        "intent_tools": ["query_metrics"],
        "needs_learning": False,
        "is_numeric": True,
        "sub_questions": [],
        "rationale": "A question about documentation usage, not about its content.",
    },
)


# ----------------------------------------------------------------------------------------
# The Analyst
# ----------------------------------------------------------------------------------------


def test_a_numeric_question_is_answered_by_the_analyst(spans, analytics, scripted, relayr_corpus):
    scripted(
        {
            "router": [[NUMERIC_PLAN]],
            "reader": [
                [
                    (
                        "query_metrics",
                        {"metric": "search_misses", "dimensions": ["section"], "window": "30d"},
                    )
                ],
                [
                    (
                        "respond",
                        {
                            "answer": "Over the last 30 days, Retries had the most searches that "
                            "found nothing: 248 across two queries.",
                            "units": "searches with no usable result",
                            "rows": [{"section": "Retries", "value": 248}],
                            "query": "SELECT section ... FROM search_misses",
                            "assumptions": ["Window is the 30 days ending 2026-11-01."],
                        },
                    )
                ],
            ],
        }
    )

    outcome = run_turn(
        TurnRequest(
            message="which section has the most search misses?",
            session_id="s1",
            corpora=["relayr"],
        )
    )

    assert outcome.answer is not None
    assert "Retries" in outcome.answer.answer
    # The Analyst's own words, not re-worded through the synthesizer: a number that passes
    # through a second model is a number that can change.
    assert not spans.has(telemetry.SYNTHESIZE)
    assert spans.parent_name("analyst") == telemetry.TURN
    assert spans.path("query_metrics") == [telemetry.TURN, "analyst", "query_metrics"]
    # Nothing to retrieve and nothing to verify against a page.
    assert not spans.has("librarian")
    assert not spans.has("auditor")
    assert any("Window is the 30 days" in c for c in outcome.answer.caveats)


def test_a_numeric_follow_up_takes_its_filter_from_the_conversation(
    spans, analytics, scripted, relayr_corpus
):
    """The acceptance criterion: context, not repetition, supplies the filter."""
    store = get_session_store()
    store.record_turn(
        "s2",
        Turn(message="how do retries work in relayr?", answer="max_attempts defaults to 5."),
        corpora=["relayr"],
    )

    captured: dict = {}

    def capture_plan(messages):
        captured["context"] = "\n".join(m.content for m in messages)
        return [NUMERIC_PLAN]

    scripted(
        {
            "router": [capture_plan],
            "reader": [
                [
                    (
                        "query_metrics",
                        # The corpus filter came from the session, not from the question.
                        {
                            "metric": "page_views",
                            "dimensions": ["section"],
                            "filters": {"corpus": "relayr"},
                        },
                    )
                ],
                [
                    (
                        "respond",
                        {
                            "answer": "Quickstart leads with 2,934 views in the last 30 days.",
                            "units": "views",
                            "rows": [{"section": "Quickstart", "value": 2934}],
                            "query": "SELECT ...",
                            "assumptions": [],
                        },
                    )
                ],
            ],
        }
    )

    outcome = run_turn(
        TurnRequest(message="and how often is it read?", session_id="s2", user_id="u1")
    )

    assert outcome.answer is not None
    assert "2,934" in outcome.answer.answer
    # The planner saw the conversation, which is how "it" resolved to relayr.
    assert "relayr" in captured["context"]
    assert spans.attr("query_metrics", "metrics.metric") == "page_views"


def test_the_analyst_falls_back_to_the_docs_when_it_returns_nothing(
    spans, analytics, scripted, relayr_corpus
):
    """A numeric question the store cannot answer should still get an answer, not a blank."""
    scripted(
        {
            "router": [[NUMERIC_PLAN]],
            "reader": [
                # Plain strings: the model replies without calling anything, which is how a
                # specialist ends up with no result.
                "I cannot answer that from the analytics store.",
                "Still nothing.",
                [
                    (
                        "respond",
                        {
                            "draft": "The docs do not record that.",
                            "retrieval_confidence": 0.2,
                            "unanswered": ["the original question"],
                        },
                    )
                ],
            ],
            "judge": [[("respond", {"verdict": "pass", "confidence": 0.3})]],
            "synthesizer": ["I do not have that number."],
        }
    )
    outcome = run_turn(
        TurnRequest(
            message="how many widgets?",
            session_id="s3",
            # Crawl off: the point here is the Analyst-to-Librarian fallback, not learning.
            overrides={"policy": {"crawl": {"mode": "off"}}},
        )
    )
    assert outcome.answer is not None
    assert spans.has("analyst")
    assert spans.has("librarian")


# ----------------------------------------------------------------------------------------
# Session memory
# ----------------------------------------------------------------------------------------


def test_cited_docs_are_pinned_so_a_follow_up_lands_on_the_same_page(relayr_corpus):
    from anydocs.knowledge import retrieval

    store = get_session_store()
    store.record_turn("s4", Turn(message="q", answer="a", citations=["relayr-abc"]))
    assert store.get("s4").pinned_docs() == ["relayr-abc"]

    # The pin is a boost, not a lock: it rides a contextvar so the model never has to pass it.
    retrieval.set_session_pins(["relayr-abc"])
    assert retrieval.current_session_pins() == ["relayr-abc"]
    retrieval.set_session_pins(None)


def test_versions_the_user_mentioned_are_remembered(relayr_corpus):
    store = get_session_store()
    store.record_turn("s5", Turn(message="does this work on v3.1 of relayr?", answer="yes"))
    assert "3.1" in store.get("s5").versions


def test_a_long_session_is_rolled_up_into_a_summary(spans, scripted, relayr_corpus):
    """The summary replaces the turns it covers and lands on the span, so a session eval can see
    what the agent thought the conversation was about."""
    store = get_session_store()
    for i in range(12):
        store.record_turn(
            "s6",
            Turn(message=f"question {i} " + "x" * 200, answer="answer " + "y" * 400),
        )
    session = store.get("s6")
    assert store.needs_summary(session)

    scripted({"summarizer": ["The user is configuring retries in relayr v3."]})
    summary = store.roll_up("s6")

    assert "retries" in summary
    after = store.get("s6")
    assert after.summary == summary
    assert len(after.turns) == 1  # the last turn stays for literal context
    assert "retries" in after.context_blurb()
    assert spans.has(telemetry.SESSION_SUMMARY)


def test_a_short_session_is_not_summarized(spans, scripted, relayr_corpus):
    store = get_session_store()
    store.record_turn("s7", Turn(message="short", answer="short"))
    assert store.roll_up("s7") == ""
    assert not spans.has(telemetry.SESSION_SUMMARY)
    assert SUMMARY_TOKEN_BUDGET > 0


def test_a_failed_summary_does_not_fail_a_turn_that_already_succeeded(
    spans, scripted, relayr_corpus, monkeypatch
):
    from anydocs.orchestrator import turn as turn_mod

    store = get_session_store()
    for _ in range(12):
        store.record_turn("s8", Turn(message="x" * 400, answer="y" * 400))

    def boom(session_id: str) -> str:
        raise RuntimeError("summarizer unavailable")

    monkeypatch.setattr(store, "roll_up", boom)
    with telemetry.turn_context(turn_id="t1"):
        with telemetry.turn_span(message="q"):
            turn_mod._roll_up_session(store, "s8")

    assert "summarizer unavailable" in spans.attr(telemetry.TURN, "session.summary_failed")


# ----------------------------------------------------------------------------------------
# The opt-in profile
# ----------------------------------------------------------------------------------------


def test_a_profile_is_empty_until_something_is_declared(data_dir):
    profile = get_profile_store().get("u1")
    assert profile.preferences == {}
    assert profile.opted_in is False
    assert profile.as_prompt_block() == ""


def test_a_declared_preference_reaches_the_orchestrator_prompt(data_dir):
    profile = get_profile_store().declare("u1", {"language": "python", "verbosity": "brief"})
    assert profile.opted_in is True
    assert profile.version == 1
    block = profile.as_prompt_block()
    assert "language: python" in block and "verbosity: brief" in block


def test_an_undeclarable_key_is_refused(data_dir):
    with pytest.raises(ValueError, match="Not a declarable preference"):
        get_profile_store().declare("u1", {"favourite_colour": "blue"})


def test_an_inference_is_a_suggestion_and_changes_nothing(data_dir):
    """The agent never silently starts behaving differently because it noticed a pattern."""
    key = infer_from_turn("u2", "how do I do this with pip install relayr in python?")
    assert key == "language"

    profile = get_profile_store().get("u2")
    assert profile.preferences == {}  # nothing in effect
    assert profile.as_prompt_block() == ""  # nothing injected
    suggestion = profile.suggestions["language"]
    assert suggestion.value == "python"
    assert suggestion.evidence  # with the evidence, so a prompt can say why


def test_repeated_observations_accumulate_evidence(data_dir):
    for message in ("pip install relayr", "python asyncio with relayr", "def handler(): pass"):
        infer_from_turn("u3", message)
    suggestion = get_profile_store().get("u3").suggestions["language"]
    assert suggestion.observed == 3
    assert len(suggestion.evidence) == 3


def test_only_a_human_promotes_a_suggestion(data_dir):
    infer_from_turn("u4", "pip install relayr")
    store = get_profile_store()
    assert store.get("u4").preferences == {}

    promoted = store.confirm("u4", "language", accept=True)
    assert promoted.preferences["language"] == "python"
    assert "language" not in promoted.suggestions


def test_a_rejected_suggestion_is_dropped_without_taking_effect(data_dir):
    infer_from_turn("u5", "pip install relayr")
    rejected = get_profile_store().confirm("u5", "language", accept=False)
    assert rejected.preferences == {}
    assert rejected.suggestions == {}


def test_a_declaration_settles_the_matter(data_dir):
    store = get_profile_store()
    infer_from_turn("u6", "pip install relayr")
    declared = store.declare("u6", {"language": "go"})
    assert declared.preferences["language"] == "go"
    assert "language" not in declared.suggestions
    # And the agent does not re-suggest what the user has already answered.
    infer_from_turn("u6", "pip install relayr")
    assert store.get("u6").suggestions == {}


def test_a_profile_survives_a_restart(data_dir):
    get_profile_store().declare("u7", {"language": "python"})
    reset_profile_store()
    assert get_profile_store().get("u7").preferences == {"language": "python"}


def test_a_corrupt_profile_does_not_break_a_turn(data_dir):
    store = get_profile_store()
    store.declare("u8", {"language": "python"})
    store._path("u8").write_text("{not json")
    assert store.get("u8").preferences == {}


def test_forgetting_a_profile_removes_it(data_dir):
    store = get_profile_store()
    store.declare("u9", {"language": "python"})
    store.forget("u9")
    assert store.get("u9").preferences == {}


def test_the_profile_version_is_on_the_turn_span(spans, scripted, relayr_corpus):
    """So a behaviour change can be tied to a profile change."""
    get_profile_store().declare("u10", {"verbosity": "brief"})
    scripted(
        {
            "router": [
                [
                    (
                        "plan",
                        {
                            "specialists": ["librarian"],
                            "corpora": ["relayr"],
                            "intent_tools": [],
                            "needs_learning": False,
                            "is_numeric": False,
                            "sub_questions": [],
                            "rationale": "x",
                        },
                    )
                ]
            ],
            "reader": [[("respond", {"draft": "", "retrieval_confidence": 0.1})]],
            "synthesizer": ["Nothing found."],
        }
    )
    run_turn(
        TurnRequest(
            message="q",
            session_id="s9",
            user_id="u10",
            overrides={"policy": {"crawl": {"mode": "off"}}},
        )
    )
    assert spans.attr(telemetry.TURN, Attr.USER_PROFILE_VERSION) == 1
