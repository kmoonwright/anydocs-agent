"""Step 7's acceptance test: the same question over HTTP and over MCP yields matching span
trees.

Driven through the SDK's own client over an in-memory transport, so the assertions cover the
real protocol — tool listing, calls, resources and prompts — rather than the Python functions
behind it.
"""

from __future__ import annotations

import json

import pytest

from anydocs import telemetry
from anydocs.knowledge.digest import doc_id_for
from anydocs.mcp_server import EXPOSED_TOOLS, build_server
from anydocs.telemetry import Attr
from anydocs.tools import registry

RETRIES = "https://docs.relayr.example/retries.md"


@pytest.fixture
def mcp_session(relayr_corpus):
    """An MCP client talking to a fresh server in this process."""
    from tests.conftest import connected_mcp_session

    return connected_mcp_session(build_server())


def _payload(result) -> dict:
    return result.structured_content or json.loads(result.content[0].text)


# ----------------------------------------------------------------------------------------
# The surface
# ----------------------------------------------------------------------------------------


async def test_the_read_tools_are_generated_from_the_registry(mcp_session):
    """A tool added to the registry appears over MCP without anyone editing the MCP server."""
    async with mcp_session as session:
        listed = {t.name: t for t in (await session.list_tools()).tools}

    assert set(EXPOSED_TOOLS) <= set(listed)
    assert {"ask_anydocs", "learn", "approve"} <= set(listed)

    for name in EXPOSED_TOOLS:
        spec = registry.get_spec(name)
        schema = listed[name].input_schema
        # Flat, with no $ref: the schema came from the same Input model the LLM tool uses.
        assert set(schema.get("properties", {})) == set(spec.input_model.model_fields)
        assert "$ref" not in json.dumps(schema)
        assert listed[name].annotations.read_only_hint is (spec.side_effect == "read")


async def test_write_tools_are_not_exposed_raw(mcp_session):
    """An MCP client gets `learn`, which carries the approval semantics, not `crawl_site`."""
    async with mcp_session as session:
        names = {t.name for t in (await session.list_tools()).tools}

    for write_tool in ("crawl_site", "save_note", "forget", "link_sources", "schedule_refresh"):
        assert write_tool not in names


async def test_field_descriptions_survive_into_the_schema(mcp_session):
    """They are the only documentation the client's model gets for each argument."""
    async with mcp_session as session:
        listed = {t.name: t for t in (await session.list_tools()).tools}

    query = listed["search_docs"].input_schema["properties"]["query"]
    assert "Rewrite the user's question" in query["description"]


async def test_resources_let_a_client_pull_a_digest_into_its_own_context(mcp_session):
    doc_id = doc_id_for("relayr", RETRIES)
    async with mcp_session as session:
        resources = {str(r.uri) for r in (await session.list_resources()).resources}
        templates = {
            t.uri_template for t in (await session.list_resource_templates()).resource_templates
        }
        corpora = await session.read_resource("anydocs://corpora")
        doc = await session.read_resource(f"anydocs://docs/{doc_id}")

    assert "anydocs://corpora" in resources
    assert "anydocs://docs/{doc_id}" in templates
    assert "relayr" in json.loads(corpora.contents[0].text)["corpora"][0]["name"]

    payload = json.loads(doc.contents[0].text)
    assert payload["doc"]["doc_id"] == doc_id
    assert payload["doc"]["chunks"]
    assert payload["provenance"][0]["doc_id"] == doc_id
    assert payload["links"] == [] and payload["changelog"] == []


async def test_a_missing_doc_resource_says_so_rather_than_failing(mcp_session):
    async with mcp_session as session:
        result = await session.read_resource("anydocs://docs/not-a-doc")
    assert "No doc" in json.loads(result.contents[0].text)["error"]


async def test_prompts_are_reusable_and_parameterized(mcp_session):
    async with mcp_session as session:
        names = {p.name for p in (await session.list_prompts()).prompts}
        answer = await session.get_prompt("answer-from-docs", {"question": "how do retries work?"})
        compare = await session.get_prompt(
            "compare-versions",
            {"topic": "retries", "from_version": "v2", "to_version": "v3", "corpus": "relayr"},
        )

    assert names == {"answer-from-docs", "compare-versions"}
    text = answer.messages[0].content.text
    assert "how do retries work?" in text
    assert "Inferred:" in text  # the citation discipline travels with the prompt
    assert "`relayr` corpus" in compare.messages[0].content.text


# ----------------------------------------------------------------------------------------
# Calls, and the spans they produce
# ----------------------------------------------------------------------------------------


async def test_a_retrieval_call_produces_the_same_span_as_the_http_path(spans, mcp_session):
    async with mcp_session as session:
        result = await session.call_tool("search_docs", {"query": "max_attempts", "k": 3})

    payload = _payload(result)
    assert payload["hits"]
    assert all(h["corpus"] == "relayr" for h in payload["hits"])

    # Same span name, same kind, same side-effect tag as the HTTP path: the tool layer is shared,
    # so a trace from either door lines up span for span.
    assert spans.attr("search_docs", Attr.SPAN_KIND) == "RETRIEVER"
    assert spans.attr("search_docs", Attr.TOOL_SIDE_EFFECT) == "read"
    assert spans.attr("search_docs", Attr.TOOL_OUTCOME) == "ran"
    assert spans.attr("search_docs", Attr.RETRIEVAL_STAGES)


async def test_a_tool_whose_store_is_missing_says_so_instead_of_erroring(spans, mcp_session):
    """The analytics store is seeded separately, so a deployment without it must get an
    explanation and the command to fix it, not a stack trace."""
    async with mcp_session as session:
        result = await session.call_tool("query_metrics", {"metric": "page_views"})
    payload = _payload(result)
    assert payload["failed"] is True
    assert "analytics/seed.py" in payload["error"]


async def test_a_policy_pause_comes_back_as_a_tool_result_with_a_resume_token(
    spans, mcp_session, monkeypatch
):
    """MCP clients have no standard approval flow, so the pause has to be something the client
    can act on rather than an error it can only report.

    The whole learn turn is covered in test_learn_then_answer.py; what is under test here is the
    translation, so the pause is raised directly.
    """
    from anydocs import mcp_server
    from anydocs.policy import NeedsApproval

    def pause(*args, **kwargs):
        raise NeedsApproval(
            "crawl_site", {"tool": "crawl_site", "estimated_pages": 7}, "resume-token-abc"
        )

    monkeypatch.setattr(mcp_server, "learn_source", pause)

    async with mcp_session as session:
        result = await session.call_tool(
            "learn", {"source": "https://docs.example.com/", "corpus": "x", "max_pages": 3}
        )

    payload = _payload(result)
    assert result.is_error is False
    assert payload["needs_approval"] is True
    assert payload["resume_token"] == "resume-token-abc"
    assert payload["estimated_pages"] == 7


async def test_a_bad_argument_tells_the_client_what_was_wrong(spans, mcp_session):
    """The SDK renders a raised exception as "Error executing tool <name>" with the detail
    stripped, and a model cannot route around a failure it cannot read."""
    async with mcp_session as session:
        result = await session.call_tool("search_docs", {"query": "x", "k": 999})

    payload = _payload(result)
    assert payload["failed"] is True
    assert "invalid arguments" in payload["error"]
    assert "less_than_equal" in payload["error"] or "25" in payload["error"]


async def test_the_whole_turn_is_available_as_one_tool(spans, mcp_session, scripted):
    """`ask_anydocs` is for clients that want the agent rather than its parts."""
    doc_id = doc_id_for("relayr", RETRIES)
    scripted(
        {
            "router": [
                [
                    (
                        "plan",
                        {
                            "specialists": ["librarian", "auditor"],
                            "corpora": ["relayr"],
                            "intent_tools": ["search_docs"],
                            "needs_learning": False,
                            "is_numeric": False,
                            "sub_questions": [],
                            "rationale": "relayr covers this",
                        },
                    )
                ]
            ],
            "reader": [
                [("search_docs", {"query": "max_attempts default", "k": 3})],
                [
                    (
                        "respond",
                        {
                            "draft": "max_attempts defaults to 5 [1].",
                            "citations": [{"doc_id": doc_id, "quote": "defaults to 5"}],
                            "retrieval_confidence": 0.85,
                        },
                    )
                ]
            ],
            "judge": [
                [
                    (
                        "respond",
                        {
                            "verdict": "pass",
                            "claims": [
                                {"claim": "max_attempts defaults to 5.", "verdict": "supported"}
                            ],
                            "confidence": 0.9,
                            "reasons": ["Quoted from the cited page."],
                        },
                    )
                ]
            ],
            "synthesizer": ["max_attempts defaults to 5 [1]."],
        }
    )

    async with mcp_session as session:
        result = await session.call_tool(
            "ask_anydocs", {"question": "what does max_attempts default to?", "corpora": ["relayr"]}
        )

    payload = _payload(result)
    assert "defaults to 5" in payload["answer"]
    assert payload["confidence"] == 0.9
    assert payload["citations"][0]["corpus"] == "relayr"
    assert payload["trace_id"]

    # The same tree the HTTP path produces, from the same turn code.
    assert spans.parent_name("librarian") == telemetry.TURN
    assert spans.parent_name("auditor") == telemetry.TURN
    assert spans.path("search_docs") == [telemetry.TURN, "librarian", "search_docs"]
    assert spans.has(telemetry.SYNTHESIZE)
    assert spans.attr(telemetry.TURN, Attr.USER_ID) == "mcp"


async def test_an_mcp_turn_is_distinguishable_in_the_trace(spans, mcp_session, scripted):
    """Both doors share one tracer provider, so what separates them is structure: an MCP turn
    nests under the transport spans and is tagged `user.id=mcp`."""
    scripted(
        {
            "router": [
                [
                    (
                        "plan",
                        {
                            "specialists": ["librarian"],
                            "corpora": ["relayr"],
                            "intent_tools": [],
                            "needs_learning": False,
                            "is_numeric": False,
                            "sub_questions": [],
                            "rationale": "x",
                        },
                    )
                ]
            ],
            "reader": [[("respond", {"draft": "", "retrieval_confidence": 0.1})]],
            "synthesizer": ["Nothing found."],
        }
    )
    async with mcp_session as session:
        await session.call_tool(
            "ask_anydocs", {"question": "q", "corpora": ["relayr"]}
        )

    assert spans.attr(telemetry.TURN, Attr.USER_ID) == "mcp"
