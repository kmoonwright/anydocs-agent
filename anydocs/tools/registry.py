"""One declaration per tool; the registry generates everything else.

A tool file contains exactly four things: a Pydantic `Input`, a Pydantic `Output`, a
`run(inp, ctx) -> Output`, and a `SPEC`. Registration derives the LLM tool schema, the MCP tool
definition, the permission check and the span wrapper from that one declaration — the tool
author writes none of them.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Span
from pydantic import BaseModel, ValidationError

from anydocs import telemetry
from anydocs.budget import Ledger, current_ledger
from anydocs.policy import NeedsApproval, Policy, PolicyDenied, default_policy

SideEffect = str  # read | write | delete


@dataclass(frozen=True)
class ToolContext:
    """What a tool gets besides its validated input.

    `span` is handed over so a tool can record its own domain attributes (retrieval stages,
    robots decisions). It never *names* a span — that stays in telemetry.py.
    """

    span: Span
    policy: Policy
    ledger: Ledger
    requested_by: str = "agent"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    run: Callable[[Any, ToolContext], BaseModel]
    side_effect: SideEffect = "read"
    span_kind: str = "TOOL"
    cost_hint: str = "low"  # low | medium | high
    policy_key: str = "none"  # crawl | write | delete | web_search | none

    def __post_init__(self) -> None:
        if self.side_effect not in ("read", "write", "delete"):
            raise ValueError(f"{self.name}: side_effect must be read, write or delete")
        # `digest_page` is the deliberate exception: a pure function over already-fetched
        # content. Anything else that writes must name the policy key gating it — including a
        # tool whose `run` is still a stub, since the spec is the part of a stub that exists.
        if (
            self.side_effect != "read"
            and self.policy_key == "none"
            and self.name != "digest_page"
        ):
            raise ValueError(f"{self.name}: a non-read tool must declare a policy_key")


class ToolError(RuntimeError):
    """A tool failed in a way the model should see and may route around."""


_REGISTRY: dict[str, ToolSpec] = {}

# Every tool module in this package. Import order does not matter; each registers itself.
TOOL_MODULES = (
    "list_corpora",
    "search_docs",
    "get_doc",
    "lint_code",
    "web_search",
    "fetch_page",
    "digest_page",
    "save_note",
    "crawl_site",
    "diff_page",
    "link_sources",
    "schedule_refresh",
    "forget",
    "query_metrics",
)


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"Tool {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


_loaded = False


def load_tools() -> None:
    """Import every tool module once, so the registry is populated."""
    global _loaded
    if _loaded:
        return
    for mod in TOOL_MODULES:
        importlib.import_module(f"anydocs.tools.{mod}")
    _loaded = True


def all_specs() -> dict[str, ToolSpec]:
    """Every registered tool.

    Loading is tracked by a flag rather than by whether the registry is empty: importing a single
    tool module — which anything may do — leaves it non-empty, and an emptiness check would then
    report that one tool as the whole catalog.
    """
    load_tools()
    return dict(_REGISTRY)


def get_spec(name: str) -> ToolSpec:
    specs = all_specs()
    try:
        return specs[name]
    except KeyError:
        raise ToolError(f"Unknown tool {name!r}. Known: {', '.join(sorted(specs))}") from None


def invoke(
    name: str,
    arguments: dict[str, Any],
    *,
    policy: Policy | None = None,
    requested_by: str = "agent",
    auto_approve: bool = False,
) -> BaseModel:
    """Run one tool: validate, gate, count, trace.

    Raises ToolError for bad input or a tool-level failure, PolicyDenied or NeedsApproval for a
    policy decision. Callers convert the first into a tool result the model sees, and let the
    second two reach the turn.
    """
    spec = get_spec(name)
    pol = policy or default_policy()
    ledger = current_ledger()

    with telemetry.tool_span(
        spec.name,
        side_effect=spec.side_effect,
        span_kind=spec.span_kind,
        description=spec.description.strip().splitlines()[0] if spec.description else "",
    ) as span:
        telemetry.set_attributes(span, {telemetry.Attr.INPUT_VALUE: arguments})

        try:
            inp = spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError(f"{name}: invalid arguments — {exc.errors()}") from None

        # Ledger first: a denied call still consumed a decision, and the cap exists to bound
        # the loop, not to reward it for being refused.
        ledger.record_tool_call(spec.name)

        # The policy gate may raise NeedsApproval, which suspends the turn. Either way the
        # span records that this tool did not run, so `crawl_site` appearing in a trace is not
        # mistaken for a crawl having happened.
        try:
            pol.enforce(
                spec.policy_key,
                action=spec.name,
                payload=_approval_payload(spec, inp),
                auto_approve=auto_approve,
            )
        except NeedsApproval:
            telemetry.set_attributes(span, {telemetry.Attr.TOOL_OUTCOME: "needs_approval"})
            raise
        except PolicyDenied:
            telemetry.set_attributes(span, {telemetry.Attr.TOOL_OUTCOME: "denied"})
            raise
        telemetry.set_attributes(span, {telemetry.Attr.TOOL_OUTCOME: "ran"})

        ctx = ToolContext(span=span, policy=pol, ledger=ledger, requested_by=requested_by)
        result = spec.run(inp, ctx)

        if not isinstance(result, spec.output_model):
            raise ToolError(
                f"{name}: run() returned {type(result).__name__}, "
                f"expected {spec.output_model.__name__}"
            )
        telemetry.set_attributes(
            span, {telemetry.Attr.OUTPUT_VALUE: result.model_dump(mode="json")}
        )
        return result


def _approval_payload(spec: ToolSpec, inp: BaseModel) -> dict[str, Any]:
    """What the client sees when a tool pauses for approval."""
    payload: dict[str, Any] = {
        "tool": spec.name,
        "side_effect": spec.side_effect,
        "cost_hint": spec.cost_hint,
        "arguments": json.loads(inp.model_dump_json()),
    }
    estimate = getattr(inp, "max_pages", None)
    if estimate:
        payload["estimated_pages"] = int(estimate)
    return payload
