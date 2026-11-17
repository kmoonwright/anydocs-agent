"""Recovering a structured result the provider handed back badly.

Every case here came off a real run against Haiku on a long documentation page, which is the only
way they were ever going to be found: the suite's scripted models return well-formed objects.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from anydocs.knowledge.digest import Digest
from anydocs.models.roles import _decode, _repair, _truncation_note


def _call(args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "Digest", "id": "1", "type": "tool_call", "args": args}],
    )


class _Box(BaseModel):
    items: list[str]


def test_a_nested_list_returned_as_a_json_string_is_recovered():
    """Observed: the call is well-formed and the content is right, but the array is a string."""
    args = {"items": json.dumps(["a", "b"])}
    result, error = _repair(_Box, _call(args))
    assert error == ""
    assert result.items == ["a", "b"]


def test_escapes_the_model_invented_are_dropped():
    """A backtick is not a JSON escape. Haiku escaped markdown as though it were markup."""
    decoded, error = _decode(r'["see \`POST /learn\`"]')
    assert error == ""
    assert decoded == ["see `POST /learn`"]


def test_a_real_escaped_backslash_survives_the_repair():
    """The regression that cost a debugging round.

    A pattern matching only invalid escapes can begin matching inside a valid `\\\\` pair, which
    turns markdown's `max\\_pages` into an invalid `\\_` — breaking a document while fixing it.
    """
    decoded, error = _decode(r'["max\\_pages and \`tick\`"]')
    assert error == ""
    assert decoded == [r"max\_pages and `tick`"]


def test_a_unicode_escape_is_left_alone():
    decoded, error = _decode(r'["café"]')
    assert error == ""
    assert decoded == ["café"]


def test_a_string_that_is_not_json_at_all_reports_why():
    _, error = _decode('["unterminated')
    assert error and "line 1" in error


def test_a_truncated_call_is_reported_as_an_output_limit_not_a_schema_error():
    """`max_tokens` arrives as a Pydantic type error, which reads like a schema problem.

    Anthropic streams tool input as partial JSON, so a cut-off response is a well-formed call
    whose unclosed field is a raw string. Saying so is the difference between raising a limit and
    rewriting a prompt.
    """
    raw = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
    note = _truncation_note(raw, "summarizer")
    assert "output limit" in note and "summarizer" in note

    done = AIMessage(content="", response_metadata={"stop_reason": "tool_use"})
    assert _truncation_note(done, "summarizer") == ""


def test_a_digest_whose_facts_arrived_as_an_escaped_string_validates():
    """The end-to-end shape of the failure, against the real schema."""
    facts = json.dumps(
        [{"claim": "Use `POST /learn`.", "quote": "POST /learn runs workflow 2"}]
    )
    args = {
        "purpose": "p",
        # Re-escape the backticks the way the model did, making the string invalid JSON.
        "facts": facts.replace("`", r"\`"),
        "parameters": [],
        "examples": [],
        "answers": [],
    }
    result, error = _repair(Digest, _call(args))
    assert error == ""
    assert result.facts[0].claim == "Use `POST /learn`."


@pytest.mark.parametrize("raw", [AIMessage(content="no tool call"), None])
def test_nothing_to_repair_says_so(raw):
    result, error = _repair(_Box, raw)
    assert result is None and error
