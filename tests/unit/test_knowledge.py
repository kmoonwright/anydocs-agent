"""Chunking, digests, snapshots and hybrid retrieval."""

from __future__ import annotations

import pytest

from anydocs.knowledge import retrieval
from anydocs.knowledge.digest import (
    Digest,
    Example,
    Fact,
    Parameter,
    QAPair,
    chunk_markdown,
    content_hash,
    doc_id_for,
    extract_version_markers,
    heuristic_digest,
)
from anydocs.knowledge.store import EmbeddingMismatch, open_snapshot

# ----------------------------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------------------------


def test_chunks_carry_the_heading_path_that_makes_a_citation_resolvable():
    chunks = chunk_markdown(
        "# Retries\n\nIntro text that is long enough to stand on its own as a chunk because it "
        "needs to clear the minimum.\n\n## Configuring retries\n\nmax_attempts defaults to 5 and "
        "may be set between 1 and 10 for any client you construct.\n",
        "doc1",
    )
    assert [c.heading for c in chunks] == ["Retries", "Retries > Configuring retries"]
    assert chunks[0].chunk_id == "doc1#000"


def test_a_chunk_never_spans_two_headings():
    chunks = chunk_markdown(
        "## A\n\n" + "alpha " * 40 + "\n\n## B\n\n" + "beta " * 40,
        "doc1",
    )
    assert len(chunks) == 2
    assert "beta" not in chunks[0].text
    assert "alpha" not in chunks[1].text


def test_code_fences_are_not_cut_mid_block():
    body = "## Example\n\n```python\n" + "x = 1\n" * 600 + "```\n"
    chunks = chunk_markdown(body, "doc1")
    joined = "\n".join(c.text for c in chunks)
    assert joined.count("```") == 2


def test_headings_inside_a_fence_do_not_start_a_section():
    chunks = chunk_markdown(
        "## Shell\n\n```bash\n# install the thing\npip install relayr\n```\n\nThat is the whole "
        "install step and there is nothing else to do here.\n",
        "doc1",
    )
    assert all(c.heading == "Shell" for c in chunks)


def test_tiny_fragments_fold_only_within_the_same_heading():
    """Folding across a heading boundary would erase the heading a citation points at."""
    same_heading = chunk_markdown("## A\n\n" + "alpha " * 400 + "\n\ntail\n", "doc1")
    assert all(c.heading == "A" for c in same_heading)
    assert "tail" in same_heading[-1].text

    across_headings = chunk_markdown("## A\n\n" + "alpha " * 40 + "\n\n## B\n\nshort\n", "doc1")
    assert [c.heading for c in across_headings] == ["A", "B"]
    assert across_headings[1].text == "short"


# ----------------------------------------------------------------------------------------
# Digests
# ----------------------------------------------------------------------------------------


def test_quality_rewards_grounding_and_completeness():
    full = Digest(
        purpose="Explains retries.",
        facts=[
            Fact(claim="retries on 5xx", source_span="retries on 5xx"),
            Fact(claim="b", source_span="b"),
        ],
        parameters=[Parameter(name="max_attempts", type="int", default="5")],
        examples=[Example(language="python", code="x=1", symbols=["RetryPolicy"])],
        answers=[QAPair(question=f"q{i}", answer=f"a{i}") for i in range(3)],
    )
    ungrounded = full.model_copy(update={"facts": [Fact(claim="retries on 5xx")]})
    assert full.quality_score() == 1.0
    assert ungrounded.quality_score() < full.quality_score()


def test_a_structure_only_digest_scores_below_a_filled_one():
    """The offline path must be visibly second-class in the store, not pass as a real digest."""
    markdown = (
        "# Retries\n\n- Retries happen on 5xx responses only.\n"
        "- max_attempts defaults to 5.\n\n## Configuring\n\n```python\nRetryPolicy()\n```\n"
    )
    heuristic = heuristic_digest(markdown, "Retries")
    assert 0.0 < heuristic.quality_score() < 1.0
    assert not heuristic.parameters  # it cannot infer types or defaults


def test_api_surface_is_what_lint_code_checks_against():
    digest = Digest(
        purpose="x",
        parameters=[Parameter(name="max_attempts")],
        examples=[Example(language="python", code="y", symbols=["RetryPolicy", "Client"])],
    )
    assert digest.api_surface() == {"max_attempts", "RetryPolicy", "Client"}


def test_doc_ids_are_deterministic_so_a_recrawl_updates_rather_than_forks():
    assert doc_id_for("relayr", "https://x/y") == doc_id_for("relayr", "https://x/y")
    assert doc_id_for("relayr", "https://x/y") != doc_id_for("lumen", "https://x/y")


def test_content_hash_ignores_surrounding_whitespace():
    assert content_hash("  body  ") == content_hash("body")


def test_version_markers_cover_semver_and_dated_apis():
    found = extract_version_markers("Applies to v3 and API version 2026-06-01, not version 2.1")
    assert "3" in found and "2026-06-01" in found and "2.1" in found


# ----------------------------------------------------------------------------------------
# Snapshots
# ----------------------------------------------------------------------------------------


def test_ingest_writes_docs_provenance_and_an_index(relayr_corpus):
    assert relayr_corpus.pages_added == 11
    assert relayr_corpus.indexed_nodes > relayr_corpus.pages_added  # chunks plus digests
    snapshot = open_snapshot("relayr")
    assert snapshot.provenance_path.exists()
    assert len(snapshot.read_jsonl(snapshot.provenance_path)) == 11
    assert snapshot.manifest()["embedding_model"] == "fake-embeddings"


def test_reingesting_unchanged_pages_skips_them(relayr_corpus, data_dir):
    from pathlib import Path

    from anydocs.ingest import ingest_folder
    from tests.conftest import FIXTURES

    again = ingest_folder(
        Path(FIXTURES / "corpus"),
        "relayr",
        base_url="https://docs.relayr.example",
        digest=False,
        snapshot_id=relayr_corpus.snapshot_id,
    )
    assert again.pages_added == 0
    assert again.pages_unchanged == 11


def test_opening_a_snapshot_embedded_with_another_model_is_a_loud_error(
    relayr_corpus, monkeypatch
):
    """Silently mixing vector spaces produces retrieval that looks fine and is wrong."""
    from anydocs.knowledge import embeddings, store

    monkeypatch.setattr(
        store, "get_embeddings", lambda: embeddings.FakeEmbeddings("other-model", 1536)
    )
    with pytest.raises(EmbeddingMismatch, match="Re-index the snapshot"):
        open_snapshot("relayr")


def test_corpus_registry_tracks_the_default_snapshot(relayr_corpus):
    from anydocs.knowledge.store import CorpusRegistry

    entry = CorpusRegistry().get("relayr")
    assert entry.default_snapshot == relayr_corpus.snapshot_id
    assert CorpusRegistry().active_corpora() == ["relayr"]


# ----------------------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------------------


def test_every_stage_is_recorded_so_a_lost_chunk_is_traceable(relayr_corpus):
    result = retrieval.search(["max_attempts maximum value"], k=5)
    stages = [s.stage for s in result.stages]
    assert stages == ["filtered_pool", "dense", "sparse_bm25", "rrf_fusion", "rerank"]
    assert all(isinstance(s.as_attribute()["count"], int) for s in result.stages)


def test_keyword_search_finds_the_page_that_states_the_limit(relayr_corpus):
    """Asserted on BM25's contribution, not on vector similarity — the test embeddings are
    deterministic hashes with no semantic meaning."""
    result = retrieval.search(["max_attempts between 1 and 10"], k=8)
    urls = {h["url"] for h in result.hits}
    assert any(u.endswith(("limits.md", "retries.md")) for u in urls)


def test_version_filter_narrows_to_pages_mentioning_it(relayr_corpus):
    unfiltered = retrieval.search(["idempotency key retention"], k=10)
    filtered = retrieval.search(["idempotency key retention"], k=10, version="2026-06-01")
    assert len(filtered.hits) <= len(unfiltered.hits)
    for hit in filtered.hits:
        assert "2026-06-01" in hit["versions"]


def test_section_filter_matches_the_heading_path(relayr_corpus):
    result = retrieval.search(["backoff"], k=10, section="Configuring retries")
    assert result.hits
    for hit in result.hits:
        assert "configuring retries" in hit["heading"].lower()


def test_gold_chunk_rank_is_recorded_for_retrieval_evals(relayr_corpus):
    first = retrieval.search(["retries"], k=10)
    gold = first.hits[0]["node_id"]
    measured = retrieval.search(["retries"], k=10, gold_chunk_id=gold)
    assert measured.gold_rank >= 1
    assert retrieval.search(["retries"], k=10, gold_chunk_id="no-such-node").gold_rank == -1


def test_an_unknown_corpus_returns_nothing_rather_than_raising(relayr_corpus):
    assert retrieval.search(["anything"], corpora=["not-a-corpus"]).hits == []


def test_rrf_fuses_two_rankings_by_rank_not_by_score():
    """Dense distances and BM25 scores are on different scales; fusion must not need either
    normalized."""
    # BM25's 99.0 must not swamp the dense leg just because the number is bigger.
    dense = [("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.4)]
    sparse = [("c", 99.0), ("b", 50.0), ("a", 1.0)]
    fused = dict(retrieval._rrf([dense, sparse]))
    # d appears in one list only, at the bottom; everything in both lists beats it.
    assert fused["d"] == min(fused.values())
    # c is last in dense and first in sparse; b is second in both. They land close together,
    # which is the point — neither list's scale decides the outcome.
    assert abs(fused["b"] - fused["c"]) < 0.002
    assert max(fused.values()) - min(fused.values()) < 0.02


def test_ingest_warms_the_corpus_up_on_its_own_best_questions(relayr_corpus):
    """A question the corpus claims to answer that returns nothing means the ingest went
    wrong — better seen in the report than in an answer."""
    assert relayr_corpus.best_questions
    assert len(relayr_corpus.warm_up) == len(relayr_corpus.best_questions)
    for line in relayr_corpus.warm_up:
        assert "FAILED" not in line
        assert not line.endswith(": 0 hits")


def test_the_retriever_is_a_langchain_retriever(relayr_corpus):
    """The hybrid search is usable by anything built on `BaseRetriever`.

    Not framework box-ticking: the same object serves the tool, which needs every stage's scores,
    and a plain LangChain chain, which only wants documents.
    """
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    from anydocs.knowledge.retriever import AnyDocsRetriever

    retriever = AnyDocsRetriever(corpora=["relayr"], k=3, rerank=False)
    assert isinstance(retriever, BaseRetriever)

    docs = retriever.invoke("max_attempts retry limit")
    assert docs and all(isinstance(d, Document) for d in docs)
    assert len(docs) <= 3
    # A passage that cannot be attributed is not evidence, so the citation fields travel with it.
    assert all(d.metadata["doc_id"] and d.metadata["url"] for d in docs)
    assert all(d.page_content for d in docs)

    # Same search, both shapes: the documents are the staged result's hits.
    staged = retriever.search(["max_attempts retry limit"])
    assert [d.id for d in docs] == [h["node_id"] for h in staged.hits]
    assert [s.stage for s in staged.stages][:1] == ["filtered_pool"]
