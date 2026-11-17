"""Two disagreeing sources become a "sources disagree" answer.

The comparison happens once, at ingest, and the answer path reads the edge rather than
re-judging it — so these tests split into: does the Archivist find the conflict, and does the
answer disclose it.
"""

from __future__ import annotations

from anydocs import telemetry
from anydocs.knowledge.links import Link, candidates_for, detect_contradictions, load_links, record
from anydocs.knowledge.store import open_snapshot
from anydocs.orchestrator.turn import TurnRequest, run_turn
from anydocs.specialists.results import AuditorResult, Citation, ClaimVerdict, LibrarianResult

CONFLICT_FOUND = (
    "conflict",
    {
        "shared_subject": "max_attempts maximum value",
        "quote_a": "max_attempts may be set between 1 and 10.",
        "quote_b": "max_attempts may be set between 1 and 20.",
        "same_version": True,
        "conflicts": True,
        "explanation": "The two pages state different maximum retry counts.",
    },
)
NO_CONFLICT = (
    "conflict",
    {"shared_subject": "", "conflicts": False, "same_version": False, "explanation": ""},
)
# The three false positives this detector produced on real documentation, now rejected in code.
DIFFERENT_SUBJECTS = (
    "conflict",
    {
        "shared_subject": "retention window",
        "quote_a": "Relayr stores the key for 24 hours.",
        "quote_b": "Receipt retention was extended to 90 days in v3.",
        "same_version": True,
        "conflicts": True,
        "explanation": "Different retention windows.",
    },
)
VERSION_DIFFERENCE = (
    "conflict",
    {
        "shared_subject": "idempotency_key requirement",
        "quote_a": "Every send requires an idempotency_key.",
        "quote_b": "In v3, idempotency_key became required, whereas it was optional in v2.",
        "same_version": False,
        "conflicts": True,
        "explanation": "One says required, one says optional.",
    },
)
SAME_NUMBERS = (
    "conflict",
    {
        "shared_subject": "max_delay_seconds maximum",
        "quote_a": "The max_delay_seconds parameter can be up to 3600 seconds.",
        "quote_b": "max_delay_seconds caps the wait, default 600, maximum 3600.",
        "same_version": True,
        "conflicts": True,
        "explanation": "They disagree about the cap.",
    },
)


# ----------------------------------------------------------------------------------------
# Detection, at ingest
# ----------------------------------------------------------------------------------------


def test_only_overlapping_pages_are_compared(relayr_corpus):
    """Comparing every pair is quadratic, and two pages sharing neither a heading nor an API
    symbol are unlikely to disagree."""
    from anydocs.knowledge.digest import doc_id_for

    snapshot = open_snapshot("relayr")
    retries = doc_id_for("relayr", "https://docs.relayr.example/retries.md")
    candidates = candidates_for(snapshot, retries)

    assert retries not in candidates
    assert len(candidates) <= 4
    titles = {snapshot.get_doc(c).title for c in candidates}
    # The pages that also mention `max_attempts` — which is where a disagreement about it
    # would live. Not the pages that merely share a section heading.
    assert titles == {"Versioning", "Limits", "Changelog"}
    assert "Quickstart" not in titles


def test_a_detected_conflict_becomes_a_contradicts_link(spans, relayr_corpus, scripted):
    from anydocs.knowledge.digest import doc_id_for

    snapshot = open_snapshot("relayr")
    retries = doc_id_for("relayr", "https://docs.relayr.example/retries.md")
    candidates = candidates_for(snapshot, retries)
    # One conflict, then no-conflict for the rest.
    scripted({"summarizer": [[CONFLICT_FOUND]] + [[NO_CONFLICT]] * (len(candidates) - 1)})

    found = detect_contradictions(snapshot, retries)

    assert len(found) == 1
    assert found[0].kind == "contradicts"
    assert found[0].from_doc_id == retries
    assert "max_attempts" in found[0].evidence
    # Both readings are quoted, because a reader is shown them.
    assert "1 and 10" in found[0].evidence and "1 and 20" in found[0].evidence
    assert found[0].recorded_by == "archivist"
    assert len(load_links(open_snapshot("relayr"))) == 1


def test_claimed_conflicts_the_quotes_do_not_support_are_rejected(spans, relayr_corpus, scripted):
    """Every case here is a false positive this detector actually produced on real pages.

    A false conflict makes the agent hedge a correct answer, which is worse than missing one, so
    the model's verdict is checked against its own quotes before anything is recorded.
    """
    from anydocs.knowledge.digest import doc_id_for

    snapshot = open_snapshot("relayr")
    retries = doc_id_for("relayr", "https://docs.relayr.example/retries.md")
    candidates = candidates_for(snapshot, retries)
    assert len(candidates) >= 3

    scripted(
        {"summarizer": [[DIFFERENT_SUBJECTS], [VERSION_DIFFERENCE], [SAME_NUMBERS]]
        + [[NO_CONFLICT]] * (len(candidates) - 3)}
    )
    assert detect_contradictions(snapshot, retries) == []
    assert load_links(open_snapshot("relayr")) == []


def test_pages_that_agree_record_nothing(spans, relayr_corpus, scripted):
    from anydocs.knowledge.digest import doc_id_for

    snapshot = open_snapshot("relayr")
    retries = doc_id_for("relayr", "https://docs.relayr.example/retries.md")
    scripted({"summarizer": [[NO_CONFLICT]] * len(candidates_for(snapshot, retries))})

    assert detect_contradictions(snapshot, retries) == []
    assert load_links(open_snapshot("relayr")) == []


def test_a_failed_comparison_does_not_fail_the_ingest(data_dir, monkeypatch):
    """The pages are already saved and correct by the time conflicts are checked. A missing
    edge is a gap, not a corruption, so it must not take the ingest down with it."""
    import pathlib

    from anydocs import ingest
    from anydocs.knowledge.embeddings import reset_embeddings
    from tests.conftest import FIXTURES

    reset_embeddings()

    def boom(*args, **kwargs):
        raise RuntimeError("comparison model unavailable")

    monkeypatch.setattr("anydocs.knowledge.links.detect_across", boom)
    # `digest` must be truthy for the pass to run at all, and the digests themselves come from
    # structure, so the only model call this would make is the one being made to fail.
    monkeypatch.setattr(ingest, "_digest_for", _structural_digest)
    try:
        report = ingest.ingest_folder(
            pathlib.Path(FIXTURES / "corpus"), "relayr", digest=True, publish=True
        )
    finally:
        reset_embeddings()

    assert report.pages_added == 11
    assert report.contradictions_found == 0


def _structural_digest(markdown, url, title, use_model, policy):
    from anydocs.knowledge.digest import heuristic_digest

    digest = heuristic_digest(markdown, title)
    return digest, digest.quality_score(), "structural", "structural@1"


def test_the_post_pass_caps_how_many_comparisons_it_makes(spans, relayr_corpus, scripted):
    """A large ingest checks the pages most likely to conflict rather than every pair."""
    from anydocs.knowledge.links import detect_across

    snapshot = open_snapshot("relayr")
    doc_ids = [d.doc_id for d in snapshot.live_docs()]
    scripted({"summarizer": [[NO_CONFLICT]] * 3})

    found = detect_across(snapshot, doc_ids, max_comparisons=3)
    assert found == []


def test_one_disagreement_is_counted_once():
    """Both pages of a pair get compared, and `record` hands back the existing edge for the
    second one. Counting both would report one disagreement as two."""
    from anydocs.knowledge.links import _unique

    mirrored = [
        Link(from_doc_id="a", to_doc_id="b", kind="contradicts"),
        Link(from_doc_id="b", to_doc_id="a", kind="contradicts"),
    ]
    assert len(_unique(mirrored)) == 1

    # Direction is the content of a `supersedes` edge, so both survive.
    directional = [
        Link(from_doc_id="a", to_doc_id="b", kind="supersedes"),
        Link(from_doc_id="b", to_doc_id="a", kind="supersedes"),
    ]
    assert len(_unique(directional)) == 2

    # Distinct pairs stay distinct.
    assert len(_unique([
        Link(from_doc_id="a", to_doc_id="b", kind="contradicts"),
        Link(from_doc_id="a", to_doc_id="c", kind="contradicts"),
    ])) == 2


# ----------------------------------------------------------------------------------------
# Surfacing, at answer time
# ----------------------------------------------------------------------------------------


def _conflicting_pair(corpus: str = "relayr") -> tuple[str, str]:
    from anydocs.knowledge.digest import doc_id_for

    snapshot = open_snapshot(corpus)
    left = doc_id_for(corpus, "https://docs.relayr.example/retries.md")
    right = doc_id_for(corpus, "https://docs.relayr.example/limits.md")
    record(
        snapshot,
        Link(
            from_doc_id=left,
            to_doc_id=right,
            kind="contradicts",
            evidence="A: 'max_attempts may be set between 1 and 10.' B: 'max_attempts up to 20.'",
        ),
    )
    return left, right


def test_the_synthesizer_is_told_to_show_both_readings(spans, relayr_corpus, scripted):
    from anydocs.knowledge.links import contradictions_for
    from anydocs.orchestrator.synthesizer import synthesize

    left, right = _conflicting_pair()
    conflicts = contradictions_for([open_snapshot("relayr")], [left, right])
    assert len(conflicts) == 1

    scripted({"synthesizer": ["The docs disagree: one says 10, another says 20."]})
    answer = synthesize(
        "what is the maximum max_attempts?",
        LibrarianResult(
            draft="max_attempts goes up to 10 [1].",
            citations=[Citation(doc_id=left), Citation(doc_id=right)],
            retrieval_confidence=0.8,
        ),
        AuditorResult(verdict="pass", confidence=0.7),
        contradictions=conflicts,
    )

    assert "Sources disagree on 1 point(s)" in " ".join(answer.caveats)


def test_a_turn_reads_contradictions_for_the_docs_it_cited(spans, relayr_corpus, scripted):
    """Read, not re-judged: the conflict was quoted at ingest."""
    left, right = _conflicting_pair()
    scripted(
        {
            "router": [
                [
                    (
                        "plan",
                        {
                            "specialists": ["librarian", "auditor"],
                            "corpora": ["relayr"],
                            "intent_tools": ["search_docs"],
                            "needs_learning": False,
                            "is_numeric": False,
                            "sub_questions": [],
                            "rationale": "relayr covers this",
                        },
                    )
                ]
            ],
            "reader": [
                [
                    (
                        "respond",
                        {
                            "draft": "max_attempts goes up to 10 [1].",
                            "citations": [
                                {"doc_id": left, "quote": "between 1 and 10"},
                                {"doc_id": right, "quote": "between 1 and 10"},
                            ],
                            "retrieval_confidence": 0.82,
                        },
                    )
                ]
            ],
            "judge": [
                [
                    (
                        "respond",
                        {
                            "verdict": "revise",
                            "claims": [
                                {"claim": "max_attempts goes up to 10.", "verdict": "supported"}
                            ],
                            "confidence": 0.55,
                            "reasons": ["The cited pages disagree on the maximum."],
                            "contradictions": ["Retries says 10; Limits says 20."],
                            "revisions": ["Show both values and which page states each."],
                        },
                    )
                ]
            ],
            "synthesizer": ["The docs disagree: Retries says 10, Limits says 20."],
        }
    )

    outcome = run_turn(
        TurnRequest(message="what is the maximum max_attempts?", corpora=["relayr"])
    )

    assert outcome.answer is not None
    caveats = " ".join(outcome.answer.caveats)
    assert "Sources disagree" in caveats
    assert "do not agree with each other" in caveats
    assert spans.attr(telemetry.TURN, "answer.contradictions") == 1


def test_no_recorded_conflict_means_no_disagreement_caveat(spans, relayr_corpus, scripted):
    from anydocs.orchestrator.synthesizer import synthesize

    scripted({"synthesizer": ["max_attempts goes up to 10 [1]."]})
    answer = synthesize(
        "q",
        LibrarianResult(draft="d [1].", retrieval_confidence=0.9),
        AuditorResult(
            verdict="pass",
            confidence=0.9,
            claims=[ClaimVerdict(claim="c", verdict="supported")],
        ),
        contradictions=[],
    )
    assert not any("disagree" in c for c in answer.caveats)


def test_a_symmetric_edge_is_stored_once_whichever_way_it_is_found(spans, relayr_corpus):
    """Both pages get compared against each other during an ingest. "A contradicts B" is the
    same fact as "B contradicts A", so one disagreement must not be reported as two."""
    from anydocs.knowledge.digest import doc_id_for

    snapshot = open_snapshot("relayr")
    left = doc_id_for("relayr", "https://docs.relayr.example/retries.md")
    right = doc_id_for("relayr", "https://docs.relayr.example/limits.md")

    record(snapshot, Link(from_doc_id=left, to_doc_id=right, kind="contradicts", evidence="x"))
    record(snapshot, Link(from_doc_id=right, to_doc_id=left, kind="contradicts", evidence="x"))
    assert len(load_links(open_snapshot("relayr"))) == 1

    from anydocs.knowledge.links import contradictions_for

    assert len(contradictions_for([open_snapshot("relayr")], [left, right])) == 1


def test_a_directional_edge_keeps_its_direction(spans, relayr_corpus):
    """`supersedes` is not symmetric: which page replaces which is the whole content of it."""
    from anydocs.knowledge.digest import doc_id_for

    snapshot = open_snapshot("relayr")
    left = doc_id_for("relayr", "https://docs.relayr.example/versioning.md")
    right = doc_id_for("relayr", "https://docs.relayr.example/changelog.md")

    record(snapshot, Link(from_doc_id=left, to_doc_id=right, kind="supersedes"))
    record(snapshot, Link(from_doc_id=right, to_doc_id=left, kind="supersedes"))
    assert len(load_links(open_snapshot("relayr"))) == 2
