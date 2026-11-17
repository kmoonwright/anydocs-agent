"""Step 6's acceptance test: a changed page produces a changelog entry, and two disagreeing
sources produce a "sources disagree" answer."""

from __future__ import annotations

import pytest

from anydocs import telemetry
from anydocs.ingest import crawl_and_ingest
from anydocs.knowledge.freshness import changelog_for, load_schedule
from anydocs.knowledge.links import Link, load_links, record
from anydocs.knowledge.store import open_snapshot
from anydocs.orchestrator.maintenance import run_refresh
from anydocs.policy import policy_with_overrides
from anydocs.telemetry import Attr
from anydocs.tools.registry import ToolError, invoke

AUTO = policy_with_overrides({"crawl": {"mode": "auto"}, "write": {"mode": "auto"}})
FLAGS = "https://lumen.example.dev/docs/flags"


@pytest.fixture
def lumen(data_dir, monkeypatch, lumen_site):
    monkeypatch.setenv("ANYDOCS_RERANKER", "noop")
    from anydocs.knowledge.embeddings import reset_embeddings

    reset_embeddings()
    report = crawl_and_ingest(
        "https://lumen.example.dev/docs/",
        "lumen",
        max_pages=10,
        max_depth=2,
        digest=False,
        policy=AUTO,
    )
    yield report
    reset_embeddings()


# ----------------------------------------------------------------------------------------
# diff_page
# ----------------------------------------------------------------------------------------


def test_an_unchanged_page_reports_no_diff(spans, lumen):
    result = invoke("diff_page", {"url": FLAGS, "corpus": "lumen"}, policy=AUTO)
    assert result.changed is False
    assert result.old_hash == result.new_hash
    assert result.sections == []
    assert spans.attr("diff_page", "diff.changed") is False


def test_a_changed_page_reports_which_sections_moved(spans, lumen, lumen_site):
    lumen_site.serve(FLAGS, "flags-v2.html")
    result = invoke("diff_page", {"url": FLAGS, "corpus": "lumen"}, policy=AUTO)

    assert result.changed is True
    assert result.old_hash != result.new_hash
    changes = {s.heading: s.change for s in result.sections}
    # Two documented values were edited under one heading, and a section was added.
    assert changes.get("Flags > Parameters") == "changed"
    assert changes.get("Flags > Caching") == "added"
    assert result.sections_added == 1
    assert result.sections_changed >= 1
    # The fresh content comes back, so a changed page is re-digested without fetching twice.
    assert "Rulesets are cached" in result.markdown


def test_whitespace_churn_is_not_a_change():
    """Doc sites re-render whitespace constantly; treating that as an edit would re-digest the
    whole corpus on every pass."""
    from anydocs.tools import diff_page as mod

    assert mod._materially_different("the  default   is 5", "the default is 5") is False
    assert mod._materially_different("the default is 5", "the default is 10") is True


def test_diffing_a_doc_that_is_not_in_the_store_is_a_tool_error(spans, lumen):
    with pytest.raises(ToolError, match="No doc matching"):
        invoke("diff_page", {"doc_id": "lumen-nope"}, policy=AUTO)


# ----------------------------------------------------------------------------------------
# The freshness workflow
# ----------------------------------------------------------------------------------------


def test_a_changed_page_produces_a_changelog_entry(spans, lumen, lumen_site):
    lumen_site.serve(FLAGS, "flags-v2.html")
    report = run_refresh(corpora=["lumen"], digest=False, policy=AUTO)

    assert report.checked == 3
    assert report.changed == 1
    assert report.unchanged == 2

    snapshot = open_snapshot("lumen")
    changed = next(d for d in report.docs if d.result == "changed")
    entries = changelog_for(snapshot, changed.doc_id)
    assert len(entries) == 1
    assert entries[0]["old_hash"] != entries[0]["new_hash"]
    headings = {s["heading"]: s["change"] for s in entries[0]["sections"]}
    assert "Flags > Caching" in headings

    # The re-digest actually replaced the page: the new section is now retrievable.
    from anydocs.knowledge import retrieval

    hits = retrieval.search(["rulesets cached 60 seconds"], corpora=["lumen"], k=5).hits
    assert any("Caching" in h["heading"] for h in hits)


def test_cadence_doubles_on_unchanged_and_resets_on_changed(spans, lumen, lumen_site):
    lumen_site.serve(FLAGS, "flags-v2.html")
    run_refresh(corpora=["lumen"], digest=False, policy=AUTO)
    schedule = {e.doc_id: e for e in load_schedule(open_snapshot("lumen"))}

    from anydocs.knowledge.freshness import DEFAULT_CADENCE_DAYS, cadence_for

    docs = {d.doc_id: d for d in open_snapshot("lumen").docs().values()}
    assert {e.last_result for e in schedule.values()} == {"changed", "unchanged"}
    for entry in schedule.values():
        seeded = cadence_for(docs[entry.doc_id])
        if entry.last_result == "changed":
            # Reset to the default: a page that just moved probably will again.
            assert entry.cadence_days == DEFAULT_CADENCE_DAYS
        else:
            assert entry.cadence_days == seeded * 2
        assert entry.next_check


def test_the_refresh_is_traced_as_maintenance_with_no_user(spans, lumen, lumen_site):
    """Headless traffic shows on dashboards but can be filtered out of eval scores."""
    lumen_site.serve(FLAGS, "flags-v2.html")
    run_refresh(corpora=["lumen"], digest=False, policy=AUTO)

    assert not spans.has(telemetry.TURN)
    # The ingest that set this test up is also maintenance traffic, so pick the refresh pass.
    refresh_spans = [
        s for s in spans.find_all(telemetry.MAINTENANCE) if "refresh.checked" in s.attributes
    ]
    assert len(refresh_spans) == 1
    root = refresh_spans[0]
    assert root.attributes[Attr.USER_ID] == "system"
    assert root.attributes["refresh.changed"] == 1
    assert root.attributes["refresh.checked"] == 3
    assert spans.path("diff_page") == [telemetry.MAINTENANCE, "diff_page"]


def test_a_second_pass_finds_nothing_due(spans, lumen, lumen_site):
    lumen_site.serve(FLAGS, "flags-v2.html")
    run_refresh(corpora=["lumen"], digest=False, policy=AUTO)
    again = run_refresh(corpora=["lumen"], digest=False, policy=AUTO)
    assert again.checked == 0


# ----------------------------------------------------------------------------------------
# Links
# ----------------------------------------------------------------------------------------


def test_a_link_needs_both_endpoints_in_the_store(spans, lumen):
    docs = open_snapshot("lumen").live_docs()
    with pytest.raises(ToolError, match="must be in one snapshot"):
        invoke(
            "link_sources",
            {"from_doc_id": docs[0].doc_id, "to_doc_id": "lumen-invented", "kind": "see-also"},
            policy=AUTO,
        )


def test_an_unknown_link_kind_is_rejected(spans, lumen):
    docs = open_snapshot("lumen").live_docs()
    with pytest.raises(ToolError, match="kind must be one of"):
        invoke(
            "link_sources",
            {"from_doc_id": docs[0].doc_id, "to_doc_id": docs[1].doc_id, "kind": "rhymes-with"},
            policy=AUTO,
        )


def test_recording_the_same_edge_twice_does_not_duplicate_it(spans, lumen):
    docs = open_snapshot("lumen").live_docs()
    args = {
        "from_doc_id": docs[0].doc_id,
        "to_doc_id": docs[1].doc_id,
        "kind": "see-also",
        "evidence": "both cover flag evaluation",
    }
    first = invoke("link_sources", args, policy=AUTO)
    second = invoke("link_sources", args, policy=AUTO)
    assert first.already_present is False
    assert second.already_present is True
    assert len(load_links(open_snapshot("lumen"))) == 1


def test_see_also_links_reach_a_doc_that_did_not_rank(spans, lumen):
    """A question landing on the wrong page of a doc set should still reach the right one.

    Asserted on the expansion step itself: over a corpus this small the dense leg already
    returns every node, so there is nothing left for expansion to contribute end to end.
    """
    from anydocs.knowledge import retrieval

    snapshot = open_snapshot("lumen")
    docs = {d.title: d for d in snapshot.live_docs()}
    flags = next(d for t, d in docs.items() if "Flags" in t)
    rollouts = next(d for t, d in docs.items() if "Rollouts" in t)
    record(snapshot, Link(from_doc_id=flags.doc_id, to_doc_id=rollouts.doc_id, kind="see-also"))

    # Only the flags doc ranked; the linked rollouts doc was not in the result set.
    wanted = (flags.doc_id, rollouts.doc_id)
    by_id = {n["node_id"]: n for n in snapshot.all_nodes() if n["doc_id"] in wanted}
    ranked = [(nid, 0.01) for nid, n in by_id.items() if n["doc_id"] == flags.doc_id]
    expanded, added = retrieval._expand_by_links(ranked, by_id, [snapshot])

    assert added > 0
    assert any(by_id[nid]["doc_id"] == rollouts.doc_id for nid, _ in expanded[len(ranked):])
    # Linked-in nodes enter below everything that ranked on its own merits.
    assert all(score <= ranked[-1][1] for _, score in expanded[len(ranked):])


def test_contradicts_links_do_not_expand_retrieval(spans, lumen):
    """A conflicting page is surfaced as a conflict, not promoted as if it were more evidence."""
    from anydocs.knowledge.links import expand

    snapshot = open_snapshot("lumen")
    docs = snapshot.live_docs()
    record(
        snapshot,
        Link(from_doc_id=docs[0].doc_id, to_doc_id=docs[1].doc_id, kind="contradicts"),
    )
    assert expand(snapshot, [docs[0].doc_id]) == []


# ----------------------------------------------------------------------------------------
# forget
# ----------------------------------------------------------------------------------------


def test_forget_drops_the_content_and_keeps_the_record(spans, lumen):
    from anydocs.knowledge.provenance import history

    snapshot = open_snapshot("lumen")
    doc = snapshot.live_docs()[0]
    before_nodes = len(snapshot.all_nodes())

    result = invoke(
        "forget",
        {"doc_id": doc.doc_id, "reason": "superseded by the v3 page"},
        policy=policy_with_overrides({"delete": {"mode": "auto"}}),
    )
    assert result.tombstoned is True
    assert result.rows_removed > 0

    fresh = open_snapshot("lumen")
    tombstone = fresh.get_doc(doc.doc_id)
    # The record survives; the content does not.
    assert tombstone is not None
    assert tombstone.tombstoned is True
    assert tombstone.chunks == []
    assert tombstone.digest.facts == []
    assert tombstone.content_hash == doc.content_hash

    events = [p.event for p in history(fresh, doc.doc_id)]
    assert "saved" in events and "tombstoned" in events
    assert len(fresh.all_nodes()) < before_nodes


def test_a_forgotten_doc_is_gone_from_retrieval(spans, lumen):
    from anydocs.knowledge import retrieval

    snapshot = open_snapshot("lumen")
    doc = next(d for d in snapshot.live_docs() if "Rollouts" in d.title)
    invoke(
        "forget",
        {"doc_id": doc.doc_id, "reason": "test"},
        policy=policy_with_overrides({"delete": {"mode": "auto"}}),
    )
    hits = retrieval.search(["bucket_by percentage rollout"], corpora=["lumen"], k=10).hits
    assert all(h["doc_id"] != doc.doc_id for h in hits)


def test_forget_is_gated_by_the_delete_policy(spans, lumen):
    from anydocs.policy import NeedsApproval

    doc = open_snapshot("lumen").live_docs()[0]
    with pytest.raises(NeedsApproval):
        # `delete` defaults to `ask`, so this must pause rather than proceed.
        invoke("forget", {"doc_id": doc.doc_id, "reason": "test"})


def test_forgetting_twice_is_not_an_error(spans, lumen):
    doc = open_snapshot("lumen").live_docs()[0]
    args = {"doc_id": doc.doc_id, "reason": "test"}
    policy = policy_with_overrides({"delete": {"mode": "auto"}})
    invoke("forget", args, policy=policy)
    second = invoke("forget", args, policy=policy)
    assert second.already_tombstoned is True


def test_a_doc_forgotten_after_scheduling_drops_off_the_schedule(spans, lumen, lumen_site):
    """A forgotten page must not be re-fetched forever. One forgotten before it was ever
    scheduled is simply never checked, which needs no special case."""
    invoke("schedule_refresh", {"corpus": "lumen", "cadence_days": 1}, policy=AUTO)
    doc = next(d for d in open_snapshot("lumen").live_docs() if "Flags" in d.title)
    invoke(
        "forget",
        {"doc_id": doc.doc_id, "reason": "test"},
        policy=policy_with_overrides({"delete": {"mode": "auto"}}),
    )

    # Force everything due by asking for a pass dated far in the future.
    report = run_refresh(corpora=["lumen"], digest=False, policy=AUTO, now="2099-01-01T00:00:00")
    entry = next(d for d in report.docs if d.doc_id == doc.doc_id)
    assert "tombstoned" in entry.detail
    assert entry.next_check
