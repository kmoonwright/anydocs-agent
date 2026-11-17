"""Search the public web for documentation the store lacks.

Gated by the `web_search` policy and capped per turn, because an agent that can search will
search. The provider name goes on the span so result quality is attributable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.search import get_search_provider
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register


class Input(BaseModel):
    query: str = Field(description="What to search for. Include the product name.")
    site: str = Field(default="", description="Restrict to one domain, e.g. 'docs.stripe.com'.")
    recency: str = Field(
        default="", description="Vendor time filter, e.g. 'month' or 'year'. Empty means any."
    )
    k: int = Field(default=8, ge=1, le=20)


class Result(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    domain: str = ""
    authority_hint: float = 0.0
    published: str = ""


class Output(BaseModel):
    provider: str
    query: str
    results: list[Result] = Field(default_factory=list)


def run(inp: Input, ctx: ToolContext) -> Output:
    max_calls = int(ctx.policy.section("web_search").get("max_calls_per_turn", 0))
    ctx.ledger.record_search_call(max_calls)

    try:
        provider = get_search_provider()
        results = provider.search(inp.query, site=inp.site, recency=inp.recency, k=inp.k)
    except RuntimeError as exc:
        # An unconfigured or unreachable provider is a tool failure, not a turn failure: the
        # Scout can report that it could not search, and the answer says the store is all there
        # was.
        raise ToolError(f"web search is unavailable: {exc}") from None

    telemetry.set_attributes(
        ctx.span,
        {
            telemetry.Attr.SEARCH_PROVIDER: provider.name,
            telemetry.Attr.SEARCH_QUERY: inp.query,
            telemetry.Attr.SEARCH_RESULTS: [
                {"url": r.url, "authority": r.authority_hint} for r in results
            ],
        },
    )
    return Output(
        provider=provider.name,
        query=inp.query,
        results=[Result(**r.model_dump()) for r in results],
    )


SPEC = register(
    ToolSpec(
        name="web_search",
        description=(
            "Search the public web for documentation this agent has not ingested. Returns URLs "
            "with titles, snippets and a hint at how authoritative each domain is for docs. Use "
            "it to find a documentation root, not to answer from — pages must be crawled and "
            "digested before they can be cited."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="read",
        span_kind="TOOL",
        cost_hint="medium",
        policy_key="web_search",
    )
)
