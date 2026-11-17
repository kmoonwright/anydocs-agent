"""Step 4's acceptance test: learning a URL produces a queryable snapshot with provenance."""

from __future__ import annotations

import pytest

from anydocs import telemetry
from anydocs.ingest import crawl_and_ingest
from anydocs.knowledge import retrieval
from anydocs.knowledge.provenance import all_records
from anydocs.knowledge.store import open_snapshot
from anydocs.policy import policy_with_overrides
from anydocs.telemetry import Attr


@pytest.fixture
def crawled(data_dir, monkeypatch, lumen_site):
    monkeypatch.setenv("ANYDOCS_RERANKER", "noop")
    from anydocs.knowledge.embeddings import reset_embeddings

    reset_embeddings()
    report = crawl_and_ingest(
        "https://lumen.example.dev/docs/",
        "lumen",
        max_pages=10,
        max_depth=2,
        digest=False,
        policy=policy_with_overrides({"crawl": {"mode": "auto"}}),
    )
    yield report, lumen_site
    reset_embeddings()


def test_a_crawl_becomes_a_queryable_snapshot(spans, crawled):
    report, _ = crawled
    assert report.pages_added == 3
    assert report.pages_failed == 0
    assert report.indexed_nodes > 3

    hits = retrieval.search(["bucket_by rollout percentage"], corpora=["lumen"], k=5).hits
    assert hits
    assert any("rollouts" in h["url"] for h in hits)


def test_robots_keeps_the_crawler_out_of_disallowed_paths(spans, crawled):
    _, site = crawled
    assert not site.marketing_route.called


def test_every_saved_page_has_a_provenance_record(spans, crawled):
    report, _ = crawled
    snapshot = open_snapshot("lumen", report.snapshot_id)
    records = all_records(snapshot)
    assert len(records) == 3
    for record in records:
        assert record.url.startswith("https://lumen.example.dev/docs")
        assert record.fetched_at
        assert record.content_hash
        assert record.robots_decision == "allowed"
        assert record.event == "saved"
        assert record.requested_by == "operator"


def test_the_crawl_span_reports_what_happened(spans, crawled):
    assert spans.attr("crawl_site", Attr.PAGES_FETCHED) == 3
    assert spans.attr("crawl_site", Attr.TOOL_SIDE_EFFECT) == "write"
    assert spans.path("crawl_site") == [telemetry.MAINTENANCE, "crawl_site"]
    assert len(spans.find_all("save_note")) == 3
    assert spans.attr("save_note", Attr.PROVENANCE_ID)


def test_headless_ingest_is_traced_as_maintenance(spans, crawled):
    """So dashboards show it but user-facing eval scores can filter it out."""
    assert spans.has(telemetry.MAINTENANCE)
    assert not spans.has(telemetry.TURN)
    assert spans.attr(telemetry.MAINTENANCE, Attr.USER_ID) == "system"


def test_recrawling_into_the_same_snapshot_adds_nothing(spans, crawled):
    """Duplicate detection is per-snapshot: "we already have this page, unchanged"."""
    report, _ = crawled
    again = crawl_and_ingest(
        "https://lumen.example.dev/docs/",
        "lumen",
        max_pages=10,
        max_depth=2,
        digest=False,
        snapshot_id=report.snapshot_id,
        policy=policy_with_overrides({"crawl": {"mode": "auto"}}),
    )
    assert again.snapshot_id == report.snapshot_id
    assert again.pages_added == 0
    assert again.pages_unchanged == 3


def test_recrawling_into_a_new_snapshot_captures_the_pages_again(spans, crawled):
    """A snapshot is a point-in-time capture, so every page is new to a new one. Detecting that
    a page changed *between* snapshots is diff_page's job, in build step 6."""
    report, _ = crawled
    again = crawl_and_ingest(
        "https://lumen.example.dev/docs/",
        "lumen",
        max_pages=10,
        max_depth=2,
        digest=False,
        policy=policy_with_overrides({"crawl": {"mode": "auto"}}),
    )
    assert again.snapshot_id != report.snapshot_id
    assert again.pages_added == 3

    from anydocs.knowledge.store import CorpusRegistry

    entry = CorpusRegistry().get("lumen")
    assert set(entry.snapshots) >= {report.snapshot_id, again.snapshot_id}
    assert entry.default_snapshot == again.snapshot_id


def test_two_crawls_in_the_same_second_do_not_share_a_snapshot_id():
    """Without a distinguishing suffix they write into one directory, which looks exactly like
    duplicate detection working."""
    from anydocs.knowledge.store import new_snapshot_id

    assert len({new_snapshot_id() for _ in range(50)}) == 50


def test_a_crawl_over_the_policy_cap_is_refused(spans, data_dir, lumen_site):
    from anydocs.tools.registry import ToolError, invoke

    policy = policy_with_overrides({"crawl": {"mode": "auto", "max_pages_per_crawl": 2}})
    with pytest.raises(ToolError, match="exceeds policy max_pages_per_crawl"):
        invoke(
            "crawl_site",
            {"root_url": "https://lumen.example.dev/docs/", "corpus": "lumen", "max_pages": 50},
            policy=policy,
        )


def test_a_batch_ingest_is_not_bound_by_the_per_turn_wall_clock(data_dir, monkeypatch, lumen_site):
    """The 45-second turn budget exists because a user is waiting. A multi-page crawl is not a
    turn, and must not die partway through because of it."""
    monkeypatch.setenv("ANYDOCS_RERANKER", "noop")
    from anydocs.knowledge.embeddings import reset_embeddings

    reset_embeddings()
    try:
        report = crawl_and_ingest(
            "https://lumen.example.dev/docs/",
            "lumen",
            max_pages=10,
            max_depth=2,
            digest=False,
            policy=policy_with_overrides(
                # A wall clock so short that a shared ledger would fail every page after the
                # first; with a per-page ledger, every page still lands.
                {"crawl": {"mode": "auto"}, "budget": {"max_wall_ms": 1}}
            ),
        )
        assert report.pages_added == 3
        assert report.pages_failed == 0
    finally:
        reset_embeddings()
