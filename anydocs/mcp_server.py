"""MCP server: the third door into the same implementation.

Streamable HTTP, mounted at `/mcp` on the same FastAPI app, so there is one process and one tool
layer. The tools are generated from the same `ToolSpec` declarations the HTTP agent uses, so a
tool added to the registry appears here without anyone editing this file.

Mounted in the same process as the HTTP API, so both doors share one tracer provider. What
separates them in a trace is structure, not a project name: an MCP turn nests under
`POST /mcp/{path}` -> `tools/call <name>` and carries `user.id=mcp`, while the `anydocs.turn`
subtree beneath is identical to the HTTP path's, node for node.
"""

from __future__ import annotations

import inspect
import json
from typing import Annotated, Any

from pydantic import Field

from anydocs.ingest import learn_source
from anydocs.knowledge.doc_view import doc_detail
from anydocs.policy import NeedsApproval, PolicyDenied, default_policy
from anydocs.tools import registry

SERVER_NAME = "anydocs"

# The read set, plus the two whole-workflow tools. Write tools are deliberately absent: an MCP
# client gets `learn`, which carries the same approval semantics, rather than raw `crawl_site`
# and `save_note`.
EXPOSED_TOOLS = (
    "search_docs",
    "get_doc",
    "list_corpora",
    "web_search",
    "fetch_page",
    "diff_page",
    "lint_code",
    "query_metrics",
)


def _tool_function(spec: registry.ToolSpec):
    """Build a function whose signature *is* the tool's input model.

    The MCP SDK derives a tool's schema from the function signature, so the signature is
    synthesized from the Pydantic model rather than hand-written. Hand-writing it would mean the
    same declaration living in two places, and the second one drifting.
    """
    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for name, field in spec.input_model.model_fields.items():
        default = (
            inspect.Parameter.empty
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        # Annotated carries the description through, which is what the client's model reads.
        annotation = Annotated[field.annotation, Field(description=field.description or "")]
        parameters.append(
            inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation
            )
        )
        annotations[name] = annotation

    def run(**kwargs: Any) -> dict[str, Any]:
        try:
            result = registry.invoke(
                spec.name, kwargs, policy=default_policy(), requested_by="mcp"
            )
        except NeedsApproval as exc:
            # MCP has no standard approval flow, so the pause comes back as a tool result
            # carrying the resume token rather than as an error the client cannot act on.
            return {
                "needs_approval": True,
                "action": exc.action,
                "resume_token": exc.resume_token,
                **exc.payload,
            }
        except PolicyDenied as exc:
            return {"error": str(exc), "denied": True}
        except registry.ToolError as exc:
            # Returned, not raised: the SDK turns an exception into "Error executing tool <name>"
            # with the detail stripped, and a client's model cannot route around a failure it
            # cannot read.
            return {"error": str(exc), "failed": True}
        return result.model_dump(mode="json")

    run.__name__ = spec.name
    run.__doc__ = spec.description.strip()
    run.__signature__ = inspect.Signature(parameters, return_annotation=dict)
    run.__annotations__ = {**annotations, "return": dict}
    return run


def build_server():
    """Construct the MCP server. Importing the module must not do this — the registry and the
    settings have to be loaded first."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer(
        name=SERVER_NAME,
        title="AnyDocs",
        version="0.1.0",
        instructions=(
            "Answers questions about software documentation from a store of digested pages, "
            "each citation resolving to a real URL and heading. Use `ask_anydocs` for a whole "
            "answer with its citations and confidence; use `search_docs` and `get_doc` to pull "
            "material into your own context. `learn` ingests documentation the store lacks and "
            "may return `needs_approval` with a resume token instead of ingesting immediately."
        ),
        warn_on_duplicate_tools=False,
    )

    for name in EXPOSED_TOOLS:
        spec = registry.get_spec(name)
        server.add_tool(
            _tool_function(spec),
            name=spec.name,
            annotations=ToolAnnotations(
                read_only_hint=spec.side_effect == "read",
                destructive_hint=spec.side_effect == "delete",
            ),
        )

    _add_workflow_tools(server)
    _add_resources(server)
    _add_prompts(server)
    return server


def _add_workflow_tools(server) -> None:
    from mcp.types import ToolAnnotations

    def ask_anydocs(
        question: Annotated[str, Field(description="The documentation question to answer.")],
        corpora: Annotated[
            list[str], Field(description="Corpus names to search. Empty means all.")
        ] = [],  # noqa: B006 — the MCP schema needs a literal default, and it is never mutated
        session_id: Annotated[
            str, Field(description="Reuse across calls so follow-ups keep context.")
        ] = "",
    ) -> dict[str, Any]:
        """Answer a documentation question end to end: plan, retrieve, verify, and reply with
        citations, a confidence score and the reasons behind it. Use this when you want the
        agent rather than its parts."""
        from anydocs.orchestrator.turn import TurnRequest, run_turn

        outcome = run_turn(
            TurnRequest(
                message=question,
                session_id=session_id,
                user_id="mcp",
                corpora=list(corpora),
                requested_by="mcp",
            )
        )
        if outcome.pending_approval:
            return {"needs_approval": True, **outcome.pending_approval}
        answer = outcome.answer
        return {
            "answer": answer.answer if answer else "",
            "citations": [c.model_dump(mode="json") for c in (answer.citations if answer else [])],
            "confidence": answer.confidence if answer else 0.0,
            "reasons": answer.reasons if answer else [],
            "caveats": answer.caveats if answer else [],
            "trace_id": outcome.trace_id,
        }

    def learn(
        source: Annotated[
            str, Field(description="A documentation URL to crawl, or a query to resolve first.")
        ],
        corpus: Annotated[str, Field(description="Corpus name to write into.")] = "",
        max_pages: Annotated[int, Field(description="Cap on pages crawled.", ge=1, le=500)] = 25,
    ) -> dict[str, Any]:
        """Ingest documentation this agent does not have yet. Returns what was added, or
        `needs_approval` with a resume token when the deployment's policy requires a human
        decision before crawling."""
        try:
            return learn_source(source, corpus=corpus, max_pages=max_pages, requested_by="mcp")
        except NeedsApproval as exc:
            return {
                "needs_approval": True,
                "action": exc.action,
                "resume_token": exc.resume_token,
                **exc.payload,
            }
        except PolicyDenied as exc:
            return {"error": str(exc), "denied": True}

    def approve(
        resume_token: Annotated[str, Field(description="From a `needs_approval` result.")],
        approve: Annotated[bool, Field(description="True to proceed, false to decline.")] = True,
        note: Annotated[str, Field(description="Why. Recorded on the decision span.")] = "",
    ) -> dict[str, Any]:
        """Continue a turn that paused for approval. The decision becomes a span."""
        from anydocs.orchestrator.turn import resume_turn

        try:
            outcome = resume_turn(resume_token, approve=approve, note=note)
        except PolicyDenied as exc:
            return {"error": str(exc)}
        return {
            "answer": outcome.answer.answer if outcome.answer else "",
            "confidence": outcome.answer.confidence if outcome.answer else 0.0,
            "trace_id": outcome.trace_id,
        }

    server.add_tool(ask_anydocs, annotations=ToolAnnotations(read_only_hint=True))
    server.add_tool(learn, annotations=ToolAnnotations(read_only_hint=False))
    server.add_tool(approve, annotations=ToolAnnotations(read_only_hint=False))


def _add_resources(server) -> None:
    """Resources let a client pull a digest into its own context instead of asking us to answer."""
    from mcp.server.mcpserver.resources import FunctionResource

    def corpora_resource() -> str:
        result = registry.invoke("list_corpora", {}, requested_by="mcp")
        return json.dumps(result.model_dump(mode="json"), indent=2)

    server.add_resource(
        FunctionResource(
            uri="anydocs://corpora",
            name="corpora",
            title="Ingested corpora",
            description="Every documentation corpus this agent holds, with page counts and how "
            "fresh each one is.",
            mime_type="application/json",
            fn=corpora_resource,
        )
    )

    @server.resource(
        "anydocs://docs/{doc_id}",
        name="doc",
        title="One digested page",
        description="A page's structured digest, its raw chunks, its provenance and its links — "
        "what a citation resolves to.",
        mime_type="application/json",
    )
    def doc_resource(doc_id: str) -> str:
        detail = doc_detail(doc_id)
        if detail is None:
            return json.dumps({"error": f"No doc {doc_id!r} in the active snapshots"})
        return json.dumps(detail, indent=2, default=str)


def _add_prompts(server) -> None:
    """Reusable prompts, so a client's own model can work the way this agent's does."""

    @server.prompt(
        name="answer-from-docs",
        title="Answer from documentation",
        description="Answer a question from ingested documentation, with citations that resolve.",
    )
    def answer_from_docs(question: str, corpus: str = "") -> str:
        scope = f" Search only the `{corpus}` corpus." if corpus else ""
        return (
            f"Answer this question about the documentation: {question}\n{scope}\n\n"
            f"Use `search_docs` to find passages, and `get_doc` when a snippet is not enough to "
            f"quote accurately. Every factual sentence must carry a citation to the doc id and "
            f"heading it came from. A sentence you cannot attribute to a retrieved passage is "
            f"prefixed `Inferred:` — never stated as fact. If the store does not cover the "
            f"question, say what is missing rather than filling the gap from memory."
        )

    @server.prompt(
        name="compare-versions",
        title="Compare two versions",
        description="Find what changed about a topic between two versions of a doc set.",
    )
    def compare_versions(topic: str, from_version: str, to_version: str, corpus: str = "") -> str:
        scope = f" in the `{corpus}` corpus" if corpus else ""
        return (
            f"Compare how `{topic}` is documented in {from_version} versus {to_version}"
            f"{scope}.\n\n"
            f"Search once per version, using `search_docs` with the `version` filter, and read "
            f"the changelog or release-notes pages if the corpus has them. Report: what changed, "
            f"what was removed, what is newly documented, and anything the docs do not say — an "
            f"unstated default is not the same as an unchanged one. Cite the page and heading "
            f"for each difference, and say explicitly if a difference is only implied."
        )
