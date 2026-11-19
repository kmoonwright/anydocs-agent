"""telemetry.py is the only file that names spans — these tests hold that line."""

from __future__ import annotations

import pathlib
import re

import pytest

from anydocs import telemetry
from anydocs.telemetry import Attr

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_turn_context_rides_every_span(spans):
    with telemetry.turn_context(
        turn_id="t1", session_id="s1", user_id="u1", corpus_active=("stripe",)
    ):
        with telemetry.turn_span(message="how do I retry?"):
            with telemetry.plan_span(role="router", model="m", provider="p"):
                pass

    assert spans.names() == [telemetry.PLAN, telemetry.TURN]
    for name in (telemetry.TURN, telemetry.PLAN):
        assert spans.attr(name, Attr.TURN_ID) == "t1"
        assert spans.attr(name, Attr.SESSION_ID) == "s1"
        assert spans.attr(name, Attr.CORPUS_ACTIVE) == ("stripe",)
        # Always present, empty in a clean run, so the harness can set it without a schema change.
        assert spans.attr(name, Attr.FAULT_INJECTED) == ""


def test_span_kinds_match_the_design_doc(spans):
    with telemetry.turn_span(message="q"):
        with telemetry.specialist_span("librarian", budget={}, tools_allowed=["search_docs"]):
            with telemetry.tool_span("search_docs", side_effect="read", span_kind="RETRIEVER"):
                pass

    assert spans.attr(telemetry.TURN, Attr.SPAN_KIND) == "AGENT"
    assert spans.attr("librarian", Attr.SPAN_KIND) == "AGENT"
    assert spans.attr("search_docs", Attr.SPAN_KIND) == "RETRIEVER"
    assert spans.attr("search_docs", Attr.TOOL_SIDE_EFFECT) == "read"
    assert spans.path("search_docs") == [telemetry.TURN, "librarian", "search_docs"]


def test_maintenance_turns_get_their_own_root_name(spans):
    with telemetry.turn_span(message="refresh", maintenance=True):
        pass
    assert spans.has(telemetry.MAINTENANCE)
    assert not spans.has(telemetry.TURN)


def test_unknown_specialist_fails_loudly():
    with pytest.raises(ValueError, match="Unknown specialist"):
        with telemetry.specialist_span("archaeologist", budget={}, tools_allowed=[]):
            pass


def test_unknown_span_kind_fails_loudly():
    with pytest.raises(ValueError, match="not an\nOpenInference|not an OpenInference"):
        with telemetry.tool_span("x", side_effect="read", span_kind="WIZARD"):
            pass


def test_exceptions_land_on_the_span(spans):
    with pytest.raises(RuntimeError):
        with telemetry.turn_span(message="q"):
            raise RuntimeError("crawl blocked")

    span = spans.find(telemetry.TURN)
    assert span.status.status_code.name == "ERROR"
    assert any(e.name == "exception" for e in span.events)


def test_nested_dict_attributes_become_json_not_dropped(spans):
    with telemetry.turn_span(message="q", overrides={"model_roles": {"judge": "haiku"}}):
        pass
    raw = spans.attr(telemetry.TURN, Attr.TURN_OVERRIDES)
    assert isinstance(raw, str) and "judge" in raw


def test_no_other_module_names_a_span():
    """The design doc's one hard rule, checked mechanically.

    Any `start_as_current_span` / `start_span` call outside telemetry.py is a violation.
    """
    offenders = []
    for path in (REPO / "anydocs").rglob("*.py"):
        if path.name == "telemetry.py":
            continue
        text = path.read_text()
        if re.search(r"start_as_current_span|start_span\(", text):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"These files name spans directly: {offenders}"


def test_the_embedding_span_is_emitted_whichever_provider_is_configured(
    spans, data_dir, monkeypatch
):
    """A span that appears or vanishes with configuration is worse than no span at all."""
    from anydocs.knowledge.embeddings import get_embeddings, reset_embeddings

    reset_embeddings()
    try:
        get_embeddings().embed(["one", "two"])
    finally:
        reset_embeddings()

    assert spans.attr(telemetry.EMBED, Attr.SPAN_KIND) == "EMBEDDING"
    assert spans.attr(telemetry.EMBED, "embed.count") == 2


def test_each_design_doc_node_is_exactly_one_span(spans, relayr_corpus, scripted):
    """A chain span wrapping an identically-named LLM span reads as two nodes in the trace and
    breaks any assertion on parentage. Every named node must appear once."""
    from anydocs.orchestrator.synthesizer import synthesize
    from anydocs.specialists.results import LibrarianResult

    scripted({"synthesizer": ["Receipts are kept 90 days [1]."]})
    with telemetry.turn_context(turn_id="t1"):
        with telemetry.turn_span(message="q"):
            synthesize("q", LibrarianResult(draft="d", retrieval_confidence=0.8), None)

    for name in (telemetry.TURN, telemetry.SYNTHESIZE):
        assert len(spans.find_all(name)) == 1, name
    assert spans.parent_name(telemetry.SYNTHESIZE) == telemetry.TURN
    # Present and namespaced, not frozen: bumping a prompt is how a behavior change gets tied
    # to a version, so a test must not make that a breaking change.
    assert str(spans.attr(telemetry.SYNTHESIZE, Attr.PROMPT_VERSION)).startswith("synthesize@")


def test_the_ax_endpoint_names_the_traces_path():
    """The collector answers 200 at `/v1` and discards whatever it is sent; only `/v1/traces`
    stores it. Verified by posting to both — one trace appeared, the other vanished. This is the
    silent-success failure mode, so the path is pinned rather than trusted."""
    from anydocs.config import get_settings

    assert get_settings().arize_endpoint.endswith("/v1/traces")


def test_both_targets_batch_and_neither_needs_a_vendor_wrapper(monkeypatch, data_dir):
    """Spans are batched in every mode; serverless correctness is the per-request flush, so a
    turn never pays one blocking export per span. Phoenix and AX share the stock exporter."""
    import sys

    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from anydocs import config, telemetry

    monkeypatch.setenv("ARIZE_SPACE_ID", "space")
    monkeypatch.setenv("ARIZE_API_KEY", "key")
    monkeypatch.setitem(sys.modules, "phoenix.otel", None)  # importing it would now fail

    def processors_for(target: str, flush_per_request: str) -> list[str]:
        monkeypatch.setenv("TRACE_TARGET", target)
        monkeypatch.setenv("ANYDOCS_FLUSH_SPANS", flush_per_request)
        config.reset_settings_cache()
        telemetry.flush_tracing()
        provider = telemetry.ensure_tracing()
        try:
            active = provider._active_span_processor._span_processors
            return [type(p).__name__ for p in active]
        finally:
            telemetry.flush_tracing()
            config.reset_settings_cache()

    for target in ("ax", "phoenix"):
        for flush in ("true", "false"):
            assert processors_for(target, flush) == [BatchSpanProcessor.__name__]


def test_the_span_attribute_cap_is_raised_above_the_default(monkeypatch):
    """A turn span passes OpenTelemetry's default 128 attributes.

    Past the cap the SDK drops the oldest attribute to make room for each new one and only logs
    it, so the identity block written first — `session.id`, `turn.id`, `corpus.active` — is what
    disappears, and every downstream filter is built on those keys.
    """
    from anydocs import config, telemetry

    monkeypatch.setenv("ARIZE_SPACE_ID", "space")
    monkeypatch.setenv("ARIZE_API_KEY", "key")
    monkeypatch.setenv("TRACE_TARGET", "ax")
    config.reset_settings_cache()
    telemetry.flush_tracing()
    try:
        limits = telemetry.ensure_tracing()._span_limits
        assert limits.max_span_attributes >= 512
        assert limits.max_attributes >= 512
    finally:
        telemetry.flush_tracing()
        config.reset_settings_cache()


def test_effort_is_dropped_for_a_model_that_rejects_it():
    """`--judge-model` pointing a Claude 5 role at a 4.x model must degrade, not fail.

    Anthropic answers a request carrying `effort` to a 4.x model with a 400 — on every call, not
    just the first — so one role override would otherwise take down every turn.
    """
    from anydocs.models.chat import resolve_role, set_role_overrides

    try:
        set_role_overrides({"judge": "claude-haiku-4-5-20251001"})
        downgraded = resolve_role("judge")
        assert downgraded.model == "claude-haiku-4-5-20251001"
        assert downgraded.effort == ""

        set_role_overrides({"judge": "claude-sonnet-5"})
        kept = resolve_role("judge")
        assert kept.effort, "a Claude 5 model should still be asked for an effort level"
    finally:
        set_role_overrides(None)


def test_only_a_tool_loop_asks_for_prompt_caching():
    """A specialist resends its history on every tool call, so it caches; a one-shot call does not.

    On a one-shot call the marker is a cache write at 1.25x input that nothing reads back.
    """
    from anydocs.config import get_settings
    from anydocs.models.chat import _anthropic_kwargs

    cfg = get_settings().role("reader")
    cached = _anthropic_kwargs(cfg, cache_prompt=True)
    assert cached["model_kwargs"] == {"cache_control": {"type": "ephemeral"}}
    assert "model_kwargs" not in _anthropic_kwargs(cfg)


def test_every_claude_5_role_sets_an_effort():
    """Claude 5 rejects a non-default temperature, which is what a role without effort sends."""
    from anydocs.config import get_settings
    from anydocs.models.chat import _EFFORT_CAPABLE

    settings = get_settings()
    for role in ("router", "summarizer", "reader", "judge", "synthesizer"):
        cfg = settings.role(role)
        if cfg.provider == "anthropic" and _EFFORT_CAPABLE.search(cfg.model):
            assert cfg.effort, f"{role} is on {cfg.model} with no effort"
