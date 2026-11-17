"""The model chooses parameters; the code writes the query."""

from __future__ import annotations

import pytest

from anydocs.tools.query_metrics import METRICS
from anydocs.tools.registry import ToolError, invoke


def test_a_grouped_query_returns_exact_numbers(spans, analytics):
    """The seed is deterministic precisely so a numeric eval has one right answer."""
    result = invoke(
        "query_metrics",
        {"metric": "page_views", "dimensions": ["section"], "window": "30d", "limit": 3},
    )
    sections = [row["section"] for row in result.rows]
    assert sections == ["Quickstart", "Retries", "Changelog"]
    assert result.rows[0]["value"] == 2934
    assert result.units == "views"


def test_the_query_that_ran_comes_back_with_the_answer(spans, analytics):
    result = invoke("query_metrics", {"metric": "search_misses", "dimensions": ["section"]})
    assert "SELECT" in result.query and "FROM search_misses" in result.query
    # Values are bound, never interpolated.
    assert "?" in result.query
    assert result.parameters == ["-30 days"]
    assert spans.attr("query_metrics", "metrics.sql") == result.query


def test_filters_are_bound_parameters_not_string_fragments(spans, analytics):
    result = invoke(
        "query_metrics",
        {"metric": "page_views", "dimensions": ["section"], "filters": {"corpus": "relayr"}},
    )
    assert "corpus = ?" in result.query
    assert "relayr" in result.parameters
    assert result.rows


def test_a_filter_value_cannot_inject_sql(spans, analytics):
    """It is a bound parameter, so a value that looks like SQL is just a value that matches
    nothing."""
    result = invoke(
        "query_metrics",
        {"metric": "page_views", "filters": {"corpus": "relayr'; DROP TABLE page_views; --"}},
    )
    assert result.rows == [] or result.rows[0]["value"] is None
    # The table is still there.
    assert invoke("query_metrics", {"metric": "page_views"}).total > 0


def test_an_unknown_metric_is_refused_with_the_vocabulary(spans, analytics):
    with pytest.raises(ToolError, match="Unknown metric"):
        invoke("query_metrics", {"metric": "revenue"})


def test_a_dimension_the_metric_does_not_have_is_refused(spans, analytics):
    with pytest.raises(ToolError, match="cannot be grouped by"):
        invoke("query_metrics", {"metric": "citation_count", "dimensions": ["query"]})


def test_a_filter_on_an_unknown_dimension_is_refused(spans, analytics):
    with pytest.raises(ToolError, match="cannot be filtered on"):
        invoke("query_metrics", {"metric": "page_views", "filters": {"sneaky": "x"}})


def test_a_malformed_window_is_refused(spans, analytics):
    with pytest.raises(ToolError, match="window must look like"):
        invoke("query_metrics", {"metric": "page_views", "window": "last month"})


def test_the_window_is_anchored_to_the_data_not_the_wall_clock(spans, analytics):
    """The store is seeded to a fixed date; anchoring to today would make every answer drift."""
    result = invoke("query_metrics", {"metric": "page_views", "window": "7d"})
    assert result.window_end == "2026-11-01"
    assert result.window_start == "2026-10-25"
    assert any("not on today" in a for a in result.assumptions)


def test_a_narrower_window_returns_a_smaller_total(spans, analytics):
    week = invoke("query_metrics", {"metric": "page_views", "window": "7d"}).total
    month = invoke("query_metrics", {"metric": "page_views", "window": "30d"}).total
    assert 0 < week < month


def test_a_missing_store_explains_how_to_create_one(spans, data_dir):
    with pytest.raises(ToolError, match="analytics/seed.py"):
        invoke("query_metrics", {"metric": "page_views"})


def test_every_metric_is_answerable(spans, analytics):
    """A metric in the catalog that cannot be queried is worse than one that is absent."""
    for name, spec in METRICS.items():
        result = invoke("query_metrics", {"metric": name, "dimensions": [spec.dimensions[0]]})
        assert result.rows, name
        assert result.units == spec.units
