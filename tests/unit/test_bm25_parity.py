"""`retrieval._bm25` is a copy of `rank_bm25.BM25Okapi`; these tests keep it one.

Its scores are logged on the `sparse_bm25` and `rrf_fusion` stages, so a drift here would change
what a trace says about retrieval without any span changing shape.
"""

from __future__ import annotations

import pytest
from rank_bm25 import BM25Okapi

from anydocs.knowledge.retrieval import _bm25, _tokenize

QUERIES = [
    "how long are receipts kept",
    "retry backoff after a 429",
    "the",  # in nearly every chunk: exercises the negative-idf floor
    "webhook webhook signature",  # a repeated term counts twice, as in the original
    "zzz-not-a-word",
]


@pytest.fixture
def corpus(relayr_corpus):
    from anydocs.knowledge.store import open_snapshot

    nodes = open_snapshot("relayr").all_nodes()
    return [_tokenize(f"{n.get('heading', '')} {n.get('title', '')} {n['text']}") for n in nodes]


@pytest.mark.parametrize("query", QUERIES)
def test_scores_match_rank_bm25(corpus, query):
    expected = BM25Okapi(corpus).get_scores(_tokenize(query))
    actual = _bm25(corpus)(_tokenize(query))
    assert actual == pytest.approx(list(expected), abs=1e-9)


def test_the_floor_branch_is_exercised(corpus):
    """Guards the test itself: if no word is in over half the chunks, the floor is untested."""
    common = sum(1 for doc in corpus if "the" in doc)
    assert common > len(corpus) / 2
