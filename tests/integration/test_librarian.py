"""The Librarian end to end: rewrite, retrieve, draft, cite.

The runner, registry, retrieval and spans are all real; only the model is scripted.
"""

from __future__ import annotations

from anydocs import telemetry
from anydocs.specialists.results import LibrarianResult
from anydocs.specialists.runner import Task, run_specialist
from anydocs.telemetry import Attr


def test_a_turn_produces_the_span_tree_from_the_design_doc(spans, relayr_corpus, scripted):
    scripted(
        {
            "reader": [
                [("search_docs", {"query": "max_attempts retry limit", "k": 5})],
                [
                    (
                        "respond",
                        {
                            "draft": "`max_attempts` defaults to 5, up to 10 [1].",
                            "citations": [
                                {
                                    "doc_id": "relayr-doc",
                                    "url": "https://docs.relayr.example/retries.md",
                                    "heading": "Retries > Configuring retries",
                                    "corpus": "relayr",
                                    "quote": "max_attempts defaults to 5",
                                }
                            ],
                            "retrieval_confidence": 0.85,
                            "unanswered": [],
                            "corpora_searched": ["relayr"],
                        },
                    )
                ],
            ]
        }
    )

    with telemetry.turn_context(turn_id="t1", session_id="s1", corpus_active=("relayr",)):
        with telemetry.turn_span(message="what is the maximum max_attempts?"):
            result = run_specialist(
                "librarian",
                Task(task="what is the maximum max_attempts?", corpora=["relayr"]),
            )

    assert isinstance(result.result, LibrarianResult)
    assert result.confidence == 0.85
    assert not result.truncated

    assert spans.path("search_docs") == [telemetry.TURN, "librarian", "search_docs"]
    assert spans.attr("librarian", Attr.SPAN_KIND) == "AGENT"
    assert spans.attr("librarian", Attr.TOOLS_ALLOWED) == (
        "get_doc",
        "list_corpora",
        "search_docs",
    )
    assert spans.attr("librarian", Attr.TOOLS_CALLED) == ("search_docs",)
    assert spans.attr("librarian", Attr.RESULT_CONFIDENCE) == 0.85
    assert spans.attr("librarian", Attr.PROMPT_VERSION) == "librarian@1"


def test_the_retriever_span_records_every_stage(spans, relayr_corpus, scripted):
    scripted(
        {
            "reader": [
                [("search_docs", {"query": "receipt retention", "k": 4})],
                [
                    (
                        "respond",
                        {"draft": "Kept 90 days [1].", "retrieval_confidence": 0.8},
                    )
                ],
            ]
        }
    )
    run_specialist("librarian", Task(task="how long are receipts kept?", corpora=["relayr"]))

    stages = spans.attr("search_docs", Attr.RETRIEVAL_STAGES)
    assert "filtered_pool" in stages and "rrf_fusion" in stages and "rerank" in stages
    assert spans.attr("search_docs", Attr.RETRIEVAL_QUERIES) == ("receipt retention",)
    assert isinstance(spans.attr("search_docs", Attr.RETRIEVAL_SCORE_SPREAD), float)


def test_the_librarian_cannot_crawl(spans, relayr_corpus, scripted):
    """Tool permissions are enforceable because the allowlist lives in the spec, not the prompt."""
    scripted(
        {
            "reader": [
                [("crawl_site", {"root_url": "https://docs.relayr.example/", "corpus": "relayr"})],
                [("respond", {"draft": "I cannot crawl.", "retrieval_confidence": 0.1})],
            ]
        }
    )
    result = run_specialist("librarian", Task(task="go learn the docs", corpora=["relayr"]))

    assert any("not allowed to call 'crawl_site'" in n for n in result.notes)
    assert spans.attr("librarian", Attr.TOOLS_DENIED) == ("crawl_site",)
    assert not spans.has("crawl_site")
    # The refusal did not consume a tool-call budget slot, because no tool ran.
    assert result.tool_calls == 0


def test_a_budget_breach_returns_a_truncated_result_with_a_confidence_penalty(
    spans, relayr_corpus, scripted
):
    scripted(
        {
            "reader": [
                [("search_docs", {"query": "a"})],
                [("search_docs", {"query": "b"})],
                [("search_docs", {"query": "c"})],
            ]
        }
    )
    from anydocs.specialists.runner import Budget

    result = run_specialist(
        "librarian",
        Task(task="q", corpora=["relayr"], budget=Budget(max_tool_calls=2)),
    )
    assert result.truncated
    assert result.confidence == 0.0  # no result at all, penalty applied to zero
    assert any("stopped after 2 tool calls" in n for n in result.notes)
    assert spans.attr("librarian", Attr.RESULT_TRUNCATED) is True


def test_a_tool_failure_is_handed_back_to_the_model_rather_than_crashing(
    spans, relayr_corpus, scripted
):
    scripted(
        {
            "reader": [
                [("get_doc", {"doc_id": "does-not-exist"})],
                [
                    (
                        "respond",
                        {
                            "draft": "That page is not in the store.",
                            "retrieval_confidence": 0.1,
                            "unanswered": ["the original question"],
                        },
                    )
                ],
            ]
        }
    )
    result = run_specialist("librarian", Task(task="q", corpora=["relayr"]))
    assert result.result is not None
    assert any("No doc matching" in n for n in result.notes)


def test_an_unexpected_tool_failure_does_not_kill_the_turn(
    spans, relayr_corpus, scripted, monkeypatch
):
    """The backstop. A tool failing in a way nobody anticipated — an unconfigured provider, a
    library raising something new — must reach the model as a result, not end the turn."""
    from dataclasses import replace

    from anydocs.tools import registry

    def boom(inp, ctx):
        raise RuntimeError("the index went away")

    broken = replace(registry.get_spec("search_docs"), run=boom)
    monkeypatch.setitem(registry._REGISTRY, "search_docs", broken)

    scripted(
        {
            "reader": [
                [("search_docs", {"query": "x"})],
                [
                    (
                        "respond",
                        {
                            "draft": "I could not search.",
                            "retrieval_confidence": 0.0,
                            "unanswered": ["everything"],
                        },
                    )
                ],
            ]
        }
    )
    result = run_specialist("librarian", Task(task="q", corpora=["relayr"]))

    assert result.result is not None
    assert any("the index went away" in n for n in result.notes)
    assert any("the index went away" in f for f in spans.attr("librarian", Attr.TOOLS_FAILED))


def test_a_result_the_model_could_not_finish_degrades_instead_of_killing_the_turn(
    spans, relayr_corpus, scripted, monkeypatch
):
    """A specialist whose result schema came back unparseable is a partial result.

    Seen live: the Auditor's answer was cut off at the model's output limit, which arrives as
    invalid JSON rather than as a short result, and `create_agent` raises. Before this, that
    ended the whole turn with a stack trace instead of an answer saying what was missing.
    """
    from langchain.agents.structured_output import StructuredOutputValidationError

    from anydocs.specialists import runner

    def truncated_output(*args, **kwargs):
        raise StructuredOutputValidationError(
            "LibrarianResult", ValueError("Unterminated string at line 1 column 8524"), None
        )

    agent = runner.create_agent
    monkeypatch.setattr(
        runner, "create_agent", lambda *a, **k: type("Stub", (), {"invoke": truncated_output})()
    )
    assert agent is not None  # the real factory is what we stubbed out

    scripted({"reader": [[("search_docs", {"query": "a"})]]})
    result = run_specialist("librarian", Task(task="q", corpora=["relayr"]))

    assert result.truncated
    assert result.result is None
    assert any("could not return a usable result" in n for n in result.notes)
    assert spans.attr("librarian", Attr.RESULT_TRUNCATED) is True
