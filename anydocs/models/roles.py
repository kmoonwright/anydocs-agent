"""The two call helpers everything else uses.

Both open the named span the design doc's tree asks for, book usage to the turn ledger, and let
the `langchain` instrumentor emit the provider-level span beneath. Callers that already own the
span pass it down, so a node in the tree stays one span.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from opentelemetry.trace import Span
from pydantic import BaseModel, ValidationError

from anydocs import telemetry
from anydocs.budget import current_ledger
from anydocs.config import get_settings
from anydocs.models.chat import chat_model, resolve_role, set_role_overrides

T = TypeVar("T", bound=BaseModel)

__all__ = ["call_role", "call_role_json", "chat_model", "resolve_role", "set_role_overrides"]


def _cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price = get_settings().price_per_mtok(model)
    return (tokens_in / 1e6) * price.get("input", 0.0) + (tokens_out / 1e6) * price.get(
        "output", 0.0
    )


def _book(span: Span, result: Any, model: str) -> None:
    """Record tokens and cost from whatever the model handed back.

    Structured-output calls return the parsed model rather than a message, so usage has to be
    read off the raw response when there is one and skipped when there is not.
    """
    usage = getattr(result, "usage_metadata", None) or {}
    tokens_in = int(usage.get("input_tokens", 0) or 0)
    tokens_out = int(usage.get("output_tokens", 0) or 0)
    telemetry.set_attributes(
        span,
        {
            telemetry.Attr.LLM_TOKENS_IN: tokens_in,
            telemetry.Attr.LLM_TOKENS_OUT: tokens_out,
        },
    )
    current_ledger().record_usage(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=_cost_usd(model, tokens_in, tokens_out),
    )


def call_role(
    span_name: str,
    role: str,
    messages: list[BaseMessage],
    *,
    tools: list[Any] | None = None,
    prompt_version: str = "",
    span: Span | None = None,
) -> AIMessage:
    """One LLM round trip inside a named span.

    Pass `span` when the caller has already opened the span this call belongs to — the planner,
    the reranker and `digest_page` do, so those nodes stay one span rather than a wrapper around
    an identically-named child.
    """
    from contextlib import nullcontext

    model, cfg = chat_model(role)
    ledger = current_ledger()
    ledger.check_deadline()

    opened = (
        nullcontext(span)
        if span is not None
        else telemetry.llm_span(
            span_name,
            role=role,
            model=cfg.model,
            provider=cfg.provider,
            prompt_version=prompt_version,
        )
    )
    with opened as active:
        runnable = model.bind_tools(tools) if tools else model
        reply: AIMessage = runnable.invoke(messages)
        _book(active, reply, cfg.model)
        telemetry.set_attributes(
            active,
            {
                telemetry.Attr.OUTPUT_VALUE: reply.text
                or [c["name"] for c in (reply.tool_calls or [])]
            },
        )
        return reply


def call_role_json(
    span_name: str,
    role: str,
    messages: list[BaseMessage],
    schema: type[T],
    *,
    prompt_version: str = "",
    span: Span | None = None,
) -> T:
    """Structured output, via LangChain's `with_structured_output`.

    It forces a tool call whose schema *is* the Pydantic model, so the provider validates the
    shape before we see it, and LangChain owns the retry and the schema inlining.
    """
    from contextlib import nullcontext

    model, cfg = chat_model(role)
    ledger = current_ledger()
    ledger.check_deadline()

    opened = (
        nullcontext(span)
        if span is not None
        else telemetry.llm_span(
            span_name,
            role=role,
            model=cfg.model,
            provider=cfg.provider,
            prompt_version=prompt_version,
        )
    )
    with opened as active:
        structured = model.with_structured_output(schema, include_raw=True)
        envelope = structured.invoke(messages)
        parsed, raw = envelope.get("parsed"), envelope.get("raw")
        if raw is not None:
            _book(active, raw, cfg.model)
        if parsed is None:
            parsed, repair_error = _repair(schema, raw)
        if parsed is None:
            # The cause goes first and the Pydantic complaint last, because the complaint is
            # misleading on its own — see `_truncation_note`.
            detail = _truncation_note(raw, role) or repair_error
            raise ValueError(
                f"{span_name}: could not obtain valid {schema.__name__}: "
                f"{detail or envelope.get('parsing_error')}"
            )
        telemetry.set_attributes(
            active, {telemetry.Attr.OUTPUT_VALUE: parsed.model_dump(mode="json")}
        )
        return parsed


# Every escape sequence JSON defines, plus a catch-all for the ones a model invents. Matching
# them left to right as whole units is the point: a pattern that only looks for invalid escapes
# can start matching *inside* a valid `\\` pair and corrupt it — turning markdown's `max\\_pages`
# into an invalid `\\_` while fixing a backtick two lines away.
_ESCAPE = re.compile(r'\\(u[0-9a-fA-F]{4}|["\\/bfnrt]|.)', re.S)
_JSON_ESCAPES = '"\\/bfnrt'


def _unescape_invented(text: str) -> str:
    """Drop backslashes before characters JSON does not escape, leaving real escapes alone."""

    def fix(match: re.Match[str]) -> str:
        body = match.group(1)
        valid = len(body) == 5 or body in _JSON_ESCAPES  # `\uXXXX`, or a defined single char
        return match.group(0) if valid else body

    return _ESCAPE.sub(fix, text)


def _decode(value: str) -> tuple[Any, str]:
    """Decode a JSON string, forgiving escapes the model invented.

    A model writing a nested array out as a string escapes whatever looked like markup to it — a
    backtick, an underscore — and the result is a string no JSON parser will take.
    """
    try:
        return json.loads(value), ""
    except ValueError as first:
        cleaned = _unescape_invented(value)
        if cleaned == value:
            return None, str(first)
    try:
        return json.loads(cleaned), ""
    except ValueError as exc:
        return None, str(exc)


def _repair(schema: type[T], raw: BaseMessage | None) -> tuple[T | None, str]:
    """Second attempt at a tool call whose nested lists came back as JSON strings.

    Observed against Haiku on a long page: the call is well-formed and the content is right, but
    `facts` arrives as `"[{...}]"` rather than as a list, with markdown inside it escaped as
    though it were JSON markup. Decoding that costs nothing, where a retry costs another model
    call on the hot ingest path and a 500-page crawl cannot afford one per awkward page.
    """
    calls = getattr(raw, "tool_calls", None) or []
    if not calls:
        return None, "the model returned no tool call to repair"
    args = calls[0].get("args")
    if not isinstance(args, dict):
        return None, "the tool call carried no arguments object"

    repaired: dict[str, Any] = {}
    errors: list[str] = []
    for key, value in args.items():
        if isinstance(value, str) and value.strip()[:1] in ("[", "{"):
            decoded, error = _decode(value)
            if error:
                errors.append(f"{key} is a {len(value)}-character string that is not JSON: {error}")
            else:
                value = decoded
        repaired[key] = value
    try:
        return schema.model_validate(repaired), ""
    except ValidationError as exc:
        errors.append(str(exc))
    return None, "; ".join(errors)


def _truncation_note(raw: BaseMessage | None, role: str) -> str:
    """Whether the model simply ran out of output tokens, said plainly.

    Worth its own check because the symptom does not look like truncation. Anthropic streams tool
    input as partial JSON, so a response cut off at `max_tokens` arrives as a well-formed tool
    call whose unclosed nested value is a raw string — and the error that surfaces is a Pydantic
    "Input should be a valid list", which reads like a prompt or schema problem and is not one.
    """
    stop = (getattr(raw, "response_metadata", None) or {}).get("stop_reason")
    if stop != "max_tokens":
        return ""
    return (
        f"the model hit its output limit, so the tool call was cut off mid-value; "
        f"raise max_tokens for the {role!r} role in config/models.yaml"
    )

