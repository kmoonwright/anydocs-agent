"""Turn a fetched page into a structured digest.

A pure function over content that has already been fetched, so no policy guard gates it: it
reads nothing new and writes nothing durable. The quality score it returns is what decides
whether a page is re-digested on a better model.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.digest import Digest
from anydocs.models.roles import call_role_json, resolve_role
from anydocs.prompts import load_prompt
from anydocs.tools._raw_pages import crawled_page
from anydocs.tools.page_text import UNTRUSTED_NOTE
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register

MAX_PAGE_CHARS = 24_000
REDIGEST_THRESHOLD = 0.5


class Input(BaseModel):
    markdown: str = Field(
        default="", description="Cleaned page content from fetch_page. Omit when passing a path."
    )
    page_path: str = Field(
        default="",
        description="A `path` from a crawl_site result, to digest a page the crawl wrote to "
        "disk. Use this instead of markdown after a crawl.",
    )
    url: str = Field(default="", description="Where the page came from, for context.")
    title: str = Field(default="", description="Page title.")
    role: str = Field(
        default="summarizer",
        description="Model role to digest with. The Archivist escalates to `reader` for pages "
        "whose first digest scored poorly.",
    )


class Output(BaseModel):
    digest: Digest
    quality: float
    model_role: str
    model: str = Field(default="", description="Pass this to save_note as `digest_model`.")
    prompt_version: str = Field(
        default="", description="Pass this to save_note as `digest_prompt_version`."
    )
    truncated: bool = False
    needs_redigest: bool = False


def run(inp: Input, ctx: ToolContext) -> Output:
    markdown, url, title = inp.markdown, inp.url, inp.title
    if inp.page_path:
        record = crawled_page(inp.page_path)
        markdown = record["markdown"]
        url = url or record.get("url", "")
        title = title or record.get("title", "")
    if not markdown.strip():
        raise ToolError("digest_page needs either `markdown` or a `page_path` from a crawl")

    prompt = load_prompt("digest_page")
    body = markdown[:MAX_PAGE_CHARS]
    truncated = len(markdown) > MAX_PAGE_CHARS

    digest = call_role_json(
        telemetry.DIGEST_PAGE,
        inp.role,
        [
            SystemMessage(prompt.text),
            HumanMessage(
                f"{UNTRUSTED_NOTE}\n\n"
                f"URL: {url}\nTitle: {title}\n\n"
                f"--- page content starts ---\n{body}\n--- page content ends ---"
            ),
        ],
        Digest,
        prompt_version=prompt.version,
        span=ctx.span,
    )

    quality = digest.quality_score()
    telemetry.set_attributes(
        ctx.span,
        {
            telemetry.Attr.DIGEST_QUALITY: quality,
            telemetry.Attr.PROMPT_VERSION: prompt.version,
            telemetry.Attr.MODEL_ROLE: inp.role,
        },
    )
    return Output(
        digest=digest,
        quality=quality,
        model_role=inp.role,
        model=resolve_role(inp.role).model,
        prompt_version=prompt.version,
        truncated=truncated,
        needs_redigest=quality < REDIGEST_THRESHOLD and inp.role == "summarizer",
    )


SPEC = register(
    ToolSpec(
        name="digest_page",
        description=(
            "Turn a fetched page into a structured digest: purpose, facts with their source "
            "spans, parameters, code examples, version notes, and the questions the page "
            "answers. Pass `page_path` from a crawl_site result to digest a page the crawl "
            "wrote to disk, or `markdown` for a page from fetch_page. Returns a quality score; "
            "a low score means the page should be digested again on a stronger model."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="write",
        span_kind="LLM",
        cost_hint="high",
        policy_key="none",
    )
)
