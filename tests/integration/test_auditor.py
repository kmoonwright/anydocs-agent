"""Step 3's acceptance test: a deliberately wrong draft is caught, and the trace shows why."""

from __future__ import annotations

from anydocs import telemetry
from anydocs.knowledge.digest import doc_id_for
from anydocs.specialists.results import AuditorResult
from anydocs.specialists.runner import Task, run_specialist
from anydocs.telemetry import Attr

RETRIES_DOC = "https://docs.relayr.example/retries.md"


def _audit_task(draft: str, doc_id: str) -> Task:
    return Task(
        task="Verify this draft answer before it ships.",
        inputs={
            "question": "what is the maximum max_attempts?",
            "draft": draft,
            "citations": [{"doc_id": doc_id, "url": RETRIES_DOC, "heading": "Retries"}],
        },
        corpora=["relayr"],
    )


def test_a_wrong_claim_is_caught_and_the_span_says_which(spans, relayr_corpus, scripted):
    doc_id = doc_id_for("relayr", RETRIES_DOC)
    scripted(
        {
            "judge": [
                [("get_doc", {"doc_id": doc_id})],
                [
                    (
                        "respond",
                        {
                            "verdict": "revise",
                            "claims": [
                                {
                                    "claim": "max_attempts can be set as high as 50.",
                                    "verdict": "unsupported",
                                    "evidence": "",
                                    "doc_id": doc_id,
                                },
                                {
                                    "claim": "Retries happen on 5xx responses.",
                                    "verdict": "supported",
                                    "evidence": "Relayr retries a delivery when the destination "
                                    "returns a 5xx status",
                                    "doc_id": doc_id,
                                },
                            ],
                            "confidence": 0.3,
                            "reasons": [
                                "The cited page states max_attempts may be set between 1 and 10, "
                                "not 50."
                            ],
                            "revisions": ["Change 50 to 10, or cut the sentence."],
                        },
                    )
                ],
            ]
        }
    )

    result = run_specialist(
        "auditor", _audit_task("max_attempts can be set as high as 50 [1].", doc_id)
    )

    assert isinstance(result.result, AuditorResult)
    assert result.result.verdict == "revise"
    unsupported = [c for c in result.result.claims if c.verdict == "unsupported"]
    assert len(unsupported) == 1
    assert "50" in unsupported[0].claim
    assert result.result.revisions

    # The trace says why, not just that it failed.
    assert spans.attr("auditor", Attr.RESULT_CONFIDENCE) == 0.3
    assert spans.path("get_doc") == ["auditor", "get_doc"]
    assert spans.attr("get_doc", Attr.DOC_ID) == doc_id


def test_a_correct_draft_passes(spans, relayr_corpus, scripted):
    """Proof the Auditor is verifying rather than always failing."""
    doc_id = doc_id_for("relayr", RETRIES_DOC)
    scripted(
        {
            "judge": [
                [
                    (
                        "respond",
                        {
                            "verdict": "pass",
                            "claims": [
                                {
                                    "claim": "Relayr retries on 5xx responses and connection "
                                    "failures.",
                                    "verdict": "supported",
                                    "evidence": "Relayr retries a delivery when the destination "
                                    "returns a 5xx status or the connection fails.",
                                    "doc_id": doc_id,
                                }
                            ],
                            "confidence": 0.92,
                            "reasons": ["Every claim is quoted from the cited page."],
                        },
                    )
                ]
            ]
        }
    )
    result = run_specialist(
        "auditor",
        _audit_task("Relayr retries on 5xx responses and connection failures [1].", doc_id),
    )
    assert result.result.verdict == "pass"
    assert result.result.confidence == 0.92
    assert all(c.verdict == "supported" for c in result.result.claims)


def test_a_bad_code_sample_is_caught_by_lint_code(spans, relayr_corpus, scripted):
    doc_id = doc_id_for("relayr", RETRIES_DOC)
    scripted(
        {
            "judge": [
                [
                    (
                        "lint_code",
                        {
                            "code": "from relayr import Client\nclient = Client(\n",
                            "language": "python",
                            "context_doc_ids": [doc_id],
                        },
                    )
                ],
                [
                    (
                        "respond",
                        {
                            "verdict": "revise",
                            "claims": [],
                            "confidence": 0.2,
                            "reasons": ["The code sample does not parse."],
                            "code_findings": ["unterminated call to Client("],
                            "revisions": ["Close the call to Client()."],
                        },
                    )
                ],
            ]
        }
    )
    result = run_specialist("auditor", _audit_task("Here is how [1]:", doc_id))
    assert result.result.verdict == "revise"
    assert spans.attr("lint_code", "lint.status") == "fail"
    assert result.result.code_findings


def test_the_auditor_cannot_write(spans, relayr_corpus, scripted):
    doc_id = doc_id_for("relayr", RETRIES_DOC)
    scripted(
        {
            "judge": [
                [("forget", {"doc_id": doc_id, "reason": "I disagree with it"})],
                [("respond", {"verdict": "pass", "confidence": 0.5})],
            ]
        }
    )
    result = run_specialist("auditor", _audit_task("x", doc_id))
    assert any("not allowed to call 'forget'" in n for n in result.notes)
    assert not spans.has("forget")


def test_turn_confidence_is_the_routing_signal_for_the_labeling_queue(
    spans, relayr_corpus, scripted
):
    """The harness routes low-confidence answers without re-judging them, so the number has to
    be on the turn span."""
    from anydocs.orchestrator.synthesizer import synthesize
    from anydocs.specialists.results import AuditorResult as AR
    from anydocs.specialists.results import ClaimVerdict, LibrarianResult

    scripted({"synthesizer": ["max_attempts may be set between 1 and 10 [1]."]})
    with telemetry.turn_context(turn_id="t1"):
        with telemetry.turn_span(message="q") as span:
            answer = synthesize(
                "q",
                LibrarianResult(draft="d", retrieval_confidence=0.8),
                AR(
                    verdict="revise",
                    confidence=0.35,
                    claims=[ClaimVerdict(claim="c", verdict="unsupported")],
                    reasons=["one claim unsupported"],
                ),
            )
            telemetry.set_attributes(span, {Attr.TURN_CONFIDENCE: answer.confidence})

    assert answer.confidence == 0.35
    assert spans.attr(telemetry.TURN, Attr.TURN_CONFIDENCE) == 0.35
    assert "could not be verified" in " ".join(answer.caveats)
    assert "Confidence in this answer is low" in " ".join(answer.caveats)


def test_the_auditor_publishes_its_verdicts_as_span_attributes(spans, relayr_corpus, scripted):
    """The design doc's eval hook: per-claim verdicts are span attributes, so a groundedness eval
    is a cheap code check rather than a parse of `output.value`. They were missing from a live
    AX run — the data was in the output blob and nowhere an evaluator would look."""
    doc_id = doc_id_for("relayr", RETRIES_DOC)
    scripted(
        {
            "judge": [
                [
                    (
                        "respond",
                        {
                            "verdict": "revise",
                            "claims": [
                                {"claim": "max_attempts can be 50.", "verdict": "unsupported"},
                                {"claim": "Retries happen on 5xx.", "verdict": "supported"},
                                {"claim": "The default is 3.", "verdict": "contradicted"},
                            ],
                            "confidence": 0.3,
                            "reasons": ["Two claims are not in the cited pages."],
                        },
                    )
                ]
            ]
        }
    )
    run_specialist("auditor", _audit_task("max_attempts can be 50 [1].", doc_id))

    assert spans.attr("auditor", Attr.AUDIT_VERDICT) == "revise"
    assert spans.attr("auditor", Attr.AUDIT_UNSUPPORTED) == 2
    assert spans.attr("auditor", Attr.AUDIT_REASONS)
    claims = spans.attr("auditor", Attr.AUDIT_CLAIMS)
    assert "unsupported" in claims and "contradicted" in claims


def test_the_librarian_publishes_what_a_retrieval_eval_needs(spans, relayr_corpus, scripted):
    scripted(
        {
            "reader": [
                [
                    (
                        "respond",
                        {
                            "draft": "d [1].",
                            "citations": [{"doc_id": doc_id_for("relayr", RETRIES_DOC)}],
                            "retrieval_confidence": 0.77,
                            "unanswered": ["the second half of the question"],
                        },
                    )
                ]
            ]
        }
    )
    run_specialist("librarian", Task(task="q", corpora=["relayr"]))

    assert spans.attr("librarian", Attr.RESULT_CONFIDENCE) == 0.77
    assert spans.attr("librarian", "retrieval.citation_count") == 1
    assert spans.attr("librarian", "retrieval.unanswered")
