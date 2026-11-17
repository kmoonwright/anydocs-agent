"""The only file in this codebase that names a span.

Every span name, kind and attribute key lives here as a constant or a factory. Tool and
specialist code opens a context manager from this module; it never passes a string literal to a
tracer. If you find yourself wanting to name a span elsewhere, add a factory here instead — see
CLAUDE.md.

Turn-wide attributes (`session.id`, `user.id`, `corpus.active`, ...) ride a contextvar rather
than being threaded through call signatures, so every span inside a turn carries them without
each caller remembering to.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

MAX_ATTR_CHARS = 8000

# --------------------------------------------------------------------------------------
# Span names. Mirrors the span tree in the design doc, one constant per node.
# --------------------------------------------------------------------------------------

TURN = "anydocs.turn"
MAINTENANCE = "anydocs.maintenance"
PLAN = "orchestrator.plan"
SYNTHESIZE = "orchestrator.synthesize"
APPROVAL_PENDING = "policy.approval_pending"
APPROVAL_DECISION = "policy.approval_decision"
GUARDRAIL_INJECTION = "guardrail.injection_check"
EMBED = "embed"
SESSION_SUMMARY = "session.summarize"
RERANK = "rerank"
# The hybrid search itself, nested under the `search_docs` tool span. Named for the
# retrieval-latency eval, which filters on this name and reads the span's own duration.
RETRIEVAL_RUN = "retrieval.run"

# Specialist agent spans: the specialist's own name is the span name, so a trace reads as a
# list of job titles. Registered here so a typo in a spec.yaml fails loudly.
SPECIALISTS = ("librarian", "scout", "archivist", "analyst", "auditor")

# The LLM calls a specialist makes are named by LangChain, not here: `create_agent` runs the
# loop, so the model call is its `model` node. The design doc sketched fixed names per specialist
# (`librarian.rewrite`, `librarian.draft`), which a variable-length agent loop cannot honour, and
# a framework span with the real model on it beats a name we invent for a call we do not make.
DIGEST_PAGE = "digest_page"

# --------------------------------------------------------------------------------------
# Attribute keys. The doc's "attributes every span carries" plus the per-node extras.
# --------------------------------------------------------------------------------------


class Attr:
    """Attribute keys. Grouped by what writes them."""

    # on every span
    SESSION_ID = "session.id"
    USER_ID = "user.id"
    TURN_ID = "turn.id"
    CORPUS_ACTIVE = "corpus.active"
    SNAPSHOT_ID = "snapshot.id"
    PROMPT_VERSION = "prompt.version"
    MODEL_ROLE = "model.role"
    MODEL_NAME = "model.name"
    MODEL_DEGRADED = "model.degraded"
    POLICY_NAME = "policy.name"
    FAULT_INJECTED = "fault.injected"
    TOOL_SIDE_EFFECT = "tool.side_effect"
    TOOL_OUTCOME = "tool.outcome"  # ran | denied | needs_approval
    OPERATOR_NOTES_VERSION = "operator.notes.version"
    USER_PROFILE_VERSION = "user.profile.version"
    SESSION_SUMMARY = "session.summary"

    # turn
    TURN_CONFIDENCE = "turn.confidence"
    TURN_ANSWER = "turn.answer"
    TURN_CITATIONS = "turn.citations"
    TURN_LEARNED = "turn.learned"
    TURN_OVERRIDES = "turn.overrides"
    TURN_COST_USD = "turn.cost_usd"
    TURN_TOOL_CALLS = "turn.tool_calls"
    TURN_NOTICE = "turn.notice"

    # plan
    PLAN_JSON = "plan"
    PLAN_INTENT_TOOLS = "plan.intent_tools"
    PLAN_SPECIALISTS = "plan.specialists"

    # specialist
    BUDGET = "budget"
    RESULT_CONFIDENCE = "result.confidence"
    RESULT_TRUNCATED = "result.truncated"
    RESULT_NOTES = "result.notes"
    SPECIALIST_NAME = "specialist.name"
    TOOLS_ALLOWED = "tools.allowed"
    TOOLS_CALLED = "tools.called"
    TOOLS_DENIED = "tools.denied"
    TOOLS_FAILED = "tools.failed"

    # retrieval
    RETRIEVAL_QUERIES = "retrieval.queries"
    RETRIEVAL_STAGES = "retrieval.stages"
    RETRIEVAL_GOLD_RANK = "retrieval.gold_chunk_rank"
    RETRIEVAL_SCORE_SPREAD = "retrieval.score_spread"

    # auditor
    AUDIT_VERDICT = "audit.verdict"
    AUDIT_CLAIMS = "audit.claims"
    AUDIT_UNSUPPORTED = "audit.unsupported_count"
    AUDIT_REASONS = "audit.reasons"

    # crawl / fetch
    PAGES_FETCHED = "pages.fetched"
    PAGES_SKIPPED = "pages.skipped"
    PAGES_FAILED = "pages.failed"
    ROBOTS_DECISION = "robots.decision"
    HTTP_STATUS = "http.status_code"
    URL = "url"
    DIGEST_QUALITY = "digest.quality"
    DOC_ID = "doc.id"
    PROVENANCE_ID = "provenance.id"

    # search
    SEARCH_PROVIDER = "search.provider"
    SEARCH_QUERY = "search.query"
    SEARCH_RESULTS = "search.results"

    # policy
    POLICY_ACTION = "policy.action"
    POLICY_MODE = "policy.mode"
    POLICY_DECISION = "policy.decision"
    POLICY_ESTIMATE_PAGES = "policy.estimate.pages"
    POLICY_ESTIMATE_USD = "policy.estimate.usd"

    # openinference passthroughs, so callers never import semconv themselves
    SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
    INPUT_VALUE = SpanAttributes.INPUT_VALUE
    OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
    LLM_MODEL = SpanAttributes.LLM_MODEL_NAME
    LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
    LLM_TOKENS_IN = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
    LLM_TOKENS_OUT = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
    TOOL_NAME = SpanAttributes.TOOL_NAME
    TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION


_KIND = OpenInferenceSpanKindValues

# --------------------------------------------------------------------------------------
# Turn context: attributes stamped on every span inside a turn.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnContext:
    turn_id: str = ""
    session_id: str = ""
    user_id: str = ""
    corpus_active: tuple[str, ...] = ()
    snapshot_id: str = ""
    policy_name: str = "default"
    prompt_version: str = ""
    fault_injected: str = ""
    operator_notes_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            Attr.TURN_ID: self.turn_id,
            Attr.SESSION_ID: self.session_id,
            Attr.USER_ID: self.user_id,
            Attr.POLICY_NAME: self.policy_name,
            # Always present, empty in a clean run — the harness's fault layer sets it.
            Attr.FAULT_INJECTED: self.fault_injected,
        }
        if self.corpus_active:
            attrs[Attr.CORPUS_ACTIVE] = list(self.corpus_active)
        if self.snapshot_id:
            attrs[Attr.SNAPSHOT_ID] = self.snapshot_id
        if self.prompt_version:
            attrs[Attr.PROMPT_VERSION] = self.prompt_version
        if self.operator_notes_version:
            attrs[Attr.OPERATOR_NOTES_VERSION] = self.operator_notes_version
        attrs.update(self.extra)
        return attrs


# Default is None rather than a TurnContext instance: a ContextVar default is shared across
# every context, and TurnContext carries a dict.
_turn_context: ContextVar[TurnContext | None] = ContextVar("anydocs_turn_context", default=None)


def current_turn_context() -> TurnContext:
    return _turn_context.get() or TurnContext()


@contextmanager
def turn_context(**fields: Any) -> Iterator[TurnContext]:
    """Set or extend the turn context for everything inside the block."""
    ctx = replace(current_turn_context(), **fields)
    token = _turn_context.set(ctx)
    try:
        yield ctx
    finally:
        _turn_context.reset(token)


def update_turn_context(**fields: Any) -> TurnContext:
    """Extend the context in place — for facts learned mid-turn, like a new snapshot id."""
    ctx = replace(current_turn_context(), **fields)
    _turn_context.set(ctx)
    return ctx


# --------------------------------------------------------------------------------------
# Tracer provider setup.
# --------------------------------------------------------------------------------------

_provider = None
_instrumented = False


def ensure_tracing():
    """Register the exporter and the LLM instrumentors. Idempotent.

    Branches on TRACE_TARGET only here, and only for where spans go: local Phoenix and deployed
    AX share one exporter. The OTLP exporter fails silently when the collector is unreachable,
    so the agent runs either way.
    """
    global _provider, _instrumented
    if _provider is not None or _instrumented:
        return _provider

    from anydocs.config import get_settings

    settings = get_settings()
    project = settings.project_name
    target = settings.trace_target

    if target == "none":
        _instrumented = True
        return None

    if target == "ax":
        if not settings.arize_space_id or not settings.arize_api_key:
            raise ValueError(
                "TRACE_TARGET=ax needs ARIZE_SPACE_ID and ARIZE_API_KEY. "
                "Source them from `ax profiles` or ~/.arize/profiles/."
            )
        # The full OTLP traces path, spelled out. Arize's own `Endpoint.ARIZE` is `.../v1`
        # without it, and posting there returns 200 and discards the spans.
        endpoint = settings.arize_endpoint
        headers = {
            "authorization": settings.arize_api_key,
            "arize-space-id": settings.arize_space_id,
            # Tells the collector these are OTel spans rather than a pandas log.
            "arize-interface": "otel",
        }
        label = "Arize AX"
    elif target == "phoenix":
        endpoint = settings.phoenix_endpoint.rstrip("/") + "/v1/traces"
        headers = {}
        label = f"Phoenix at {settings.phoenix_endpoint}"
    else:
        raise ValueError(f"TRACE_TARGET must be phoenix, ax or none; got {target!r}")

    _provider = _register_otlp(endpoint, headers, project)
    print(f"Tracing to {label} (project '{project}').")
    _instrument_llm_sdks(_provider, framework_spans=settings.framework_spans)
    _instrumented = True
    return _provider


def _span_limits():
    """Raise OpenTelemetry's per-span attribute cap.

    The default is 128, and a turn span passes it: an answer's citations and an audit's per-claim
    verdicts are published as attributes, and each citation is several of them. Past the cap the
    SDK drops the *oldest* attribute per new one, with only a log line to say so — so the first
    thing lost is `session.id`, `turn.id` and the rest of the identity block, and a trace that
    looks complete is missing exactly the keys everything downstream filters on.
    """
    from opentelemetry.sdk.trace import SpanLimits

    return SpanLimits(max_attributes=512, max_span_attributes=512)


def _register_otlp(endpoint: str, headers: dict[str, str], project: str):
    """Stock OpenTelemetry over OTLP/HTTP, for both Phoenix and AX.

    Neither vendor wrapper is used. `arize.otel.register` crashes on its own HTTP transport (its
    banner reads `exporter._headers`, which that exporter does not define), and its gRPC default
    does not survive a forked or frozen process — exactly the serverless case. `phoenix.otel`
    adds nothing over this beyond a dependency on the OTLP meta-package, which drags in grpcio.

    Always a batch processor: where the host freezes the process after a response, the server's
    per-request `flush_spans()` drains it, so no turn pays one blocking POST per span.
    """
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({ResourceAttributes.PROJECT_NAME: project}),
        span_limits=_span_limits(),
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
    )
    trace.set_tracer_provider(provider)
    return provider


@contextmanager
def remote_parent(headers: Any) -> Iterator[None]:
    """Make the caller's `traceparent`, if any, the parent of every span opened inside.

    This is how a turn started by an AX experiment lands under its `agent.experiment.run` span
    instead of as a new root. With no header the extracted context is empty, and nothing changes.
    """
    from opentelemetry import context, propagate

    token = context.attach(propagate.extract(headers))
    try:
        yield
    finally:
        context.detach(token)


# What to auto-instrument, as (module suffix, instrumentor class name). The provider SDKs cover
# embeddings and the direct SDK paths that sit below LangChain's view; `mcp` carries the trace
# context across the MCP boundary, so an external caller's span is the parent of ours.
_INSTRUMENTORS = (
    ("anthropic", "AnthropicInstrumentor"),
    ("openai", "OpenAIInstrumentor"),
    ("mcp", "MCPInstrumentor"),
)

# LangChain's own view of the run: the graph, each node, each model call. Separate because it is
# the one that can be turned off — see `Settings.framework_spans`.
_FRAMEWORK_INSTRUMENTOR = ("langchain", "LangChainInstrumentor")


def _instrument_llm_sdks(tracer_provider, *, framework_spans: bool = True) -> None:
    """Auto-instrument the libraries that make calls for us. A missing one warns, not crashes."""
    import importlib

    kwargs = {"tracer_provider": tracer_provider} if tracer_provider is not None else {}
    wanted = list(_INSTRUMENTORS)
    if framework_spans:
        wanted.append(_FRAMEWORK_INSTRUMENTOR)
    for suffix, class_name in wanted:
        try:
            module = importlib.import_module(f"openinference.instrumentation.{suffix}")
        except ImportError:
            print(f"Warning: openinference-instrumentation-{suffix} not installed.")
            continue
        getattr(module, class_name)().instrument(**kwargs)


def flush_spans() -> None:
    """Push buffered spans to the collector without tearing the provider down.

    What a serverless request needs before it returns: the process may be frozen immediately
    afterwards and never reach a batch timer, so anything still buffered is lost. Cheap when
    there is nothing to send.
    """
    if _provider is None:
        return
    try:
        _provider.force_flush()
    except Exception as exc:  # a failed flush must not fail the response
        print(f"Warning: span flush failed: {type(exc).__name__}: {exc}")


def flush_tracing() -> None:
    """Flush and shut down so short-lived CLI processes actually export."""
    global _provider, _instrumented
    if _provider is None:
        # `TRACE_TARGET=none` instruments nothing, so there is no provider to flush — but the
        # module is still marked configured, and a flush must leave it unconfigured either way.
        _instrumented = False
        return
    try:
        _provider.force_flush()
    finally:
        try:
            _provider.shutdown()
        except Exception:
            pass
        _provider = None
        _instrumented = False


def tracer() -> trace.Tracer:
    return trace.get_tracer("anydocs")


# --------------------------------------------------------------------------------------
# Attribute helpers
# --------------------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """OTel accepts str/bool/int/float and homogeneous sequences of those. Everything else
    becomes truncated JSON so a span never silently drops an attribute."""
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, (str, bool, int, float)) for v in value
    ):
        return list(value)
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text[:MAX_ATTR_CHARS]


def annotate(**attributes: Any) -> None:
    """Add attributes to whatever span is currently open.

    For code that has something to record but did not open the span it is inside — an adapter
    noticing a provider quirk, say. It names no span, so the one hard rule still holds.
    """
    set_attributes(trace.get_current_span(), attributes)


def set_attributes(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        span.set_attribute(key, _coerce(value))


def _open(name: str, kind: _KIND, attributes: dict[str, Any] | None = None):
    span_attrs = {Attr.SPAN_KIND: kind.value}
    span_attrs.update(current_turn_context().as_attributes())
    if attributes:
        span_attrs.update(attributes)
    return tracer().start_as_current_span(
        name, attributes={k: _coerce(v) for k, v in span_attrs.items()}
    )


@contextmanager
def _span(name: str, kind: _KIND, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
    with _open(name, kind, attributes) as span:
        try:
            yield span
        except Exception as exc:
            # Fail visibly: the exception lands on the span, not just in a log.
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


# --------------------------------------------------------------------------------------
# Span factories, one per node in the doc's span tree.
# --------------------------------------------------------------------------------------


@contextmanager
def turn_span(
    *,
    message: str,
    maintenance: bool = False,
    overrides: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """The root span for one turn. `maintenance=True` renames it so headless traffic can be
    filtered out of user-facing eval scores."""
    name = MAINTENANCE if maintenance else TURN
    attrs: dict[str, Any] = {Attr.INPUT_VALUE: message}
    if overrides:
        attrs[Attr.TURN_OVERRIDES] = overrides
    with _span(name, _KIND.AGENT, attrs) as span:
        yield span


@contextmanager
def plan_span(*, role: str, model: str, provider: str) -> Iterator[Span]:
    """The orchestrator's planning step. One span: the LLM call inside reuses it."""
    with _span(
        PLAN,
        _KIND.LLM,
        {
            Attr.MODEL_ROLE: role,
            Attr.MODEL_NAME: model,
            Attr.LLM_MODEL: model,
            Attr.LLM_PROVIDER: provider,
        },
    ) as span:
        yield span


@contextmanager
def specialist_span(
    name: str, *, budget: dict[str, Any], tools_allowed: list[str]
) -> Iterator[Span]:
    if name not in SPECIALISTS:
        raise ValueError(f"Unknown specialist {name!r}. Add it to telemetry.SPECIALISTS.")
    attrs = {
        Attr.SPECIALIST_NAME: name,
        Attr.BUDGET: budget,
        Attr.TOOLS_ALLOWED: sorted(tools_allowed),
    }
    with _span(name, _KIND.AGENT, attrs) as span:
        yield span


@contextmanager
def llm_span(
    name: str, *, role: str, model: str, provider: str, prompt_version: str = ""
) -> Iterator[Span]:
    attrs = {
        Attr.MODEL_ROLE: role,
        Attr.MODEL_NAME: model,
        Attr.LLM_MODEL: model,
        Attr.LLM_PROVIDER: provider,
    }
    if prompt_version:
        attrs[Attr.PROMPT_VERSION] = prompt_version
    with _span(name, _KIND.LLM, attrs) as span:
        yield span


@contextmanager
def tool_span(
    name: str, *, side_effect: str, span_kind: str = "TOOL", description: str = ""
) -> Iterator[Span]:
    """Open a tool span. Called by the registry wrapper, not by tool authors."""
    try:
        kind = _KIND(span_kind.upper())
    except ValueError:
        raise ValueError(
            f"Tool {name!r} declares span_kind={span_kind!r}, which is not an "
            f"OpenInference span kind."
        ) from None
    attrs = {Attr.TOOL_NAME: name, Attr.TOOL_SIDE_EFFECT: side_effect}
    if description:
        attrs[Attr.TOOL_DESCRIPTION] = description
    with _span(name, kind, attrs) as span:
        yield span


@contextmanager
def synthesize_span(
    *, role: str, model: str, provider: str, prompt_version: str = ""
) -> Iterator[Span]:
    attrs = {
        Attr.MODEL_ROLE: role,
        Attr.MODEL_NAME: model,
        Attr.LLM_MODEL: model,
        Attr.LLM_PROVIDER: provider,
    }
    if prompt_version:
        attrs[Attr.PROMPT_VERSION] = prompt_version
    with _span(SYNTHESIZE, _KIND.LLM, attrs) as span:
        yield span


@contextmanager
def approval_span(
    *, action: str, mode: str, pending: bool, decision: str = "", **extra: Any
) -> Iterator[Span]:
    """The approval moment, visible in the trace: one span for the pause, one for the decision."""
    attrs: dict[str, Any] = {Attr.POLICY_ACTION: action, Attr.POLICY_MODE: mode}
    if decision:
        attrs[Attr.POLICY_DECISION] = decision
    attrs.update(extra)
    name = APPROVAL_PENDING if pending else APPROVAL_DECISION
    with _span(name, _KIND.DECISION, attrs) as span:
        yield span


@contextmanager
def retrieval_span() -> Iterator[Span]:
    """The hybrid search. Open it immediately around `retrieval.search`.

    Arize's `latency_ms` is `(end_time - start_time)` once the span ends. The
    retrieval-latency eval reads that column on a span named `retrieval.run`; the
    time used to live only on `search_docs`, so the eval had no span to score.
    A handwritten latency attribute would be a second clock. Success sets OK,
    because `start_as_current_span` otherwise leaves the span UNSET. `_span`
    records the exception and sets ERROR when the search raises.
    """
    with _span(RETRIEVAL_RUN, _KIND.RETRIEVER) as span:
        yield span
        span.set_status(Status(StatusCode.OK))


@contextmanager
def embed_span(*, model: str, count: int) -> Iterator[Span]:
    with _span(
        EMBED, _KIND.EMBEDDING, {SpanAttributes.EMBEDDING_MODEL_NAME: model, "embed.count": count}
    ) as span:
        yield span


@contextmanager
def rerank_span(*, model: str, candidates: int) -> Iterator[Span]:
    with _span(
        RERANK, _KIND.RERANKER, {Attr.MODEL_NAME: model, "rerank.candidates": candidates}
    ) as span:
        yield span


@contextmanager
def guardrail_span(*, check: str) -> Iterator[Span]:
    with _span(GUARDRAIL_INJECTION, _KIND.GUARDRAIL, {"guardrail.check": check}) as span:
        yield span
