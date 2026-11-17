"""A citation from a model is a claim about the store, not a fact from it."""

from __future__ import annotations

from anydocs.knowledge.digest import doc_id_for
from anydocs.orchestrator import citations
from anydocs.specialists.results import Citation

RETRIES = "https://docs.relayr.example/retries.md"


def test_the_store_overrides_what_the_model_said_about_a_source(relayr_corpus):
    doc_id = doc_id_for("relayr", RETRIES)
    resolved, unresolvable = citations.resolve(
        [
            Citation(
                doc_id=doc_id,
                url="https://wrong.example/invented",
                title="Something Else",
                heading="A Heading That Is Not In The Page",
                corpus="",
                quote="max_attempts defaults to 5",
            )
        ]
    )
    assert not unresolvable
    assert resolved[0].url == RETRIES
    assert resolved[0].title == "Retries"
    assert resolved[0].corpus == "relayr"
    assert resolved[0].snapshot_id == relayr_corpus.snapshot_id
    # The quote is the model's claim about the source, which the Auditor checks; it is kept.
    assert resolved[0].quote == "max_attempts defaults to 5"


def test_an_invented_doc_id_is_reported_not_rendered(relayr_corpus):
    """A plausible-looking doc id a model made up gets caught here, not in front of a user."""
    resolved, unresolvable = citations.resolve(
        [Citation(doc_id="relayr-0000deadbeef", url="https://docs.relayr.example/ghost.md")]
    )
    assert resolved == []
    assert len(unresolvable) == 1


def test_the_heading_comes_from_the_chunk_the_quote_is_in(relayr_corpus):
    doc_id = doc_id_for("relayr", RETRIES)
    resolved, _ = citations.resolve(
        [Citation(doc_id=doc_id, quote="With exponential backoff, attempt")]
    )
    assert resolved[0].heading == "Retries > Backoff schedule"


def test_a_node_id_pins_the_heading_exactly(relayr_corpus):
    from anydocs.knowledge.store import open_snapshot

    doc = open_snapshot("relayr").get_doc(doc_id_for("relayr", RETRIES))
    target = doc.chunks[-1]
    resolved, _ = citations.resolve(
        [Citation(doc_id=doc.doc_id, node_id=target.chunk_id, quote="irrelevant")]
    )
    assert resolved[0].heading == target.heading


def test_no_citations_is_not_an_error(relayr_corpus):
    assert citations.resolve([]) == ([], [])
