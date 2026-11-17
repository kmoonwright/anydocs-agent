"""`retrieval.run` duration is the search's wall clock, which is what Arize scores."""

from __future__ import annotations

import time

import pytest
from opentelemetry.trace import StatusCode

from anydocs import telemetry
from anydocs.knowledge import retrieval
from anydocs.telemetry import Attr
from anydocs.tools import registry


def test_retrieval_run_span_records_wall_clock_above_the_latency_threshold(
    spans, relayr_corpus, monkeypatch
):
    """A real search, held for 2.6s, must export a non-null retriever duration.

    Arize computes `latency_ms` from the span's start and end. The assertion is that
    interval, not a latency attribute the test could write itself.
    """
    real_open = retrieval.open_active

    def slow_open(*args, **kwargs):
        time.sleep(2.6)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(retrieval, "open_active", slow_open)
    registry.invoke("search_docs", {"query": "max_attempts retry limit", "k": 3})

    # Ingest warm-up can emit its own retrieval.run spans before this call. The delayed
    # search is the one whose input is the tool query.
    matches = [
        span
        for span in spans.find_all(telemetry.RETRIEVAL_RUN)
        if "max_attempts retry limit" in str(span.attributes.get(Attr.INPUT_VALUE))
    ]
    assert len(matches) == 1
    span = matches[0]
    assert span.name == "retrieval.run"
    assert span.start_time is not None and span.end_time is not None
    latency_ms = (span.end_time - span.start_time) / 1e6
    assert latency_ms > 2_500
    assert span.attributes.get(Attr.SPAN_KIND) == "RETRIEVER"
    assert span.status.status_code == StatusCode.OK
    parent = spans._by_id[span.parent.span_id]
    assert parent.name == "search_docs"
    # The tool span keeps the input, output and stage log it had before this span existed.
    assert "max_attempts" in str(spans.attr("search_docs", Attr.INPUT_VALUE))
    assert spans.attr("search_docs", Attr.OUTPUT_VALUE)
    assert "rrf_fusion" in str(spans.attr("search_docs", Attr.RETRIEVAL_STAGES))
    # The retriever span carries the same query plus per-document metadata.
    # input.value has to be a string; Arize drops a string-array attribute.
    recorded_input = span.attributes.get(Attr.INPUT_VALUE)
    assert isinstance(recorded_input, str)
    assert "max_attempts retry limit" in recorded_input
    assert span.attributes.get("retrieval.documents.0.document.id")
    assert span.attributes.get("retrieval.documents.0.document.metadata")


def test_a_failed_retrieval_is_an_error_on_the_retriever_span(spans, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("snapshot unreadable")

    monkeypatch.setattr(retrieval, "open_active", boom)
    with pytest.raises(RuntimeError, match="snapshot unreadable"):
        retrieval.search(["max_attempts"])

    span = spans.find(telemetry.RETRIEVAL_RUN)
    assert span.status.status_code == StatusCode.ERROR
    assert "snapshot unreadable" in (span.status.description or "")
    assert any(event.name == "exception" for event in span.events)
    assert spans.attr(telemetry.RETRIEVAL_RUN, Attr.SPAN_KIND) == "RETRIEVER"
