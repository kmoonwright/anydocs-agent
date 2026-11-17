"""Re-fetch a page and compare it with what the store already holds.

The unit of comparison is a section, not a line: a reader asks "what changed about retries",
not "which bytes moved". Sections are matched on their heading path, which is also what a
citation points at, so a diff and a citation name the same thing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.digest import Chunk, content_hash, utcnow
from anydocs.knowledge.store import StoreError, find_doc
from anydocs.tools import http_client
from anydocs.tools.page_text import to_markdown
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register

# Sections whose text differs by less than this are treated as unchanged: documentation sites
# routinely re-render whitespace and smart quotes without anyone editing anything.
NOISE_RATIO = 0.02


class Input(BaseModel):
    doc_id: str = Field(default="", description="Doc to re-check.")
    url: str = Field(default="", description="URL to re-check, when the doc id is unknown.")
    corpus: str = Field(default="", description="Narrows the lookup.")
    snapshot_id: str = Field(default="", description="Compare against this snapshot.")


class SectionDiff(BaseModel):
    heading: str = ""
    change: str = "changed"  # added | removed | changed
    before: str = ""
    after: str = ""


class Output(BaseModel):
    doc_id: str = ""
    url: str = ""
    changed: bool = False
    old_hash: str = ""
    new_hash: str = ""
    sections: list[SectionDiff] = Field(default_factory=list)
    checked_at: str = ""
    http_status: int = 0
    robots_decision: str = ""
    markdown: str = Field(
        default="", description="The freshly fetched page, for re-digesting without re-fetching."
    )
    sections_added: int = 0
    sections_removed: int = 0
    sections_changed: int = 0


def run(inp: Input, ctx: ToolContext) -> Output:
    try:
        doc, _ = find_doc(inp.corpus, inp.snapshot_id, doc_id=inp.doc_id, url=inp.url)
    except StoreError as exc:
        raise ToolError(str(exc)) from None

    url = doc.canonical_url or doc.url
    try:
        page = http_client.fetch(url, policy=ctx.policy)
    except http_client.RobotsBlocked as exc:
        # A page that has become unreadable is a freshness result, not a crash: the scheduler
        # records it and backs off rather than retrying every cycle.
        telemetry.set_attributes(
            ctx.span, {telemetry.Attr.URL: url, telemetry.Attr.ROBOTS_DECISION: "disallowed"}
        )
        raise ToolError(f"robots.txt now disallows {url}") from exc

    telemetry.set_attributes(
        ctx.span,
        {
            telemetry.Attr.URL: url,
            telemetry.Attr.DOC_ID: doc.doc_id,
            telemetry.Attr.HTTP_STATUS: page.status,
            telemetry.Attr.ROBOTS_DECISION: page.robots_decision,
        },
    )
    if page.status >= 400:
        raise ToolError(f"{url} returned HTTP {page.status}")

    markdown = to_markdown(page.html, page.final_url)
    if not markdown.strip():
        raise ToolError(f"{url} produced no readable content after cleaning")

    new_hash = content_hash(markdown)
    sections = _section_diff(doc.chunks, markdown, doc.doc_id)
    changed = new_hash != doc.content_hash and bool(sections)

    counts = {"added": 0, "removed": 0, "changed": 0}
    for section in sections:
        counts[section.change] += 1

    telemetry.set_attributes(
        ctx.span,
        {
            "diff.changed": changed,
            "diff.sections_added": counts["added"],
            "diff.sections_removed": counts["removed"],
            "diff.sections_changed": counts["changed"],
        },
    )
    return Output(
        doc_id=doc.doc_id,
        url=url,
        changed=changed,
        old_hash=doc.content_hash,
        new_hash=new_hash,
        sections=sections,
        checked_at=utcnow(),
        http_status=page.status,
        robots_decision=page.robots_decision,
        markdown=markdown,
        sections_added=counts["added"],
        sections_removed=counts["removed"],
        sections_changed=counts["changed"],
    )


def _section_diff(old_chunks: list[Chunk], new_markdown: str, doc_id: str) -> list[SectionDiff]:
    """Compare old and new sections by heading path."""
    from anydocs.knowledge.digest import chunk_markdown

    new_chunks = chunk_markdown(new_markdown, doc_id)
    old_by_heading = _merge_by_heading(old_chunks)
    new_by_heading = _merge_by_heading(new_chunks)

    diffs: list[SectionDiff] = []
    for heading, old_text in old_by_heading.items():
        new_text = new_by_heading.get(heading)
        if new_text is None:
            diffs.append(
                SectionDiff(heading=heading, change="removed", before=_clip(old_text), after="")
            )
        elif _materially_different(old_text, new_text):
            diffs.append(
                SectionDiff(
                    heading=heading,
                    change="changed",
                    before=_clip(old_text),
                    after=_clip(new_text),
                )
            )
    for heading, new_text in new_by_heading.items():
        if heading not in old_by_heading:
            diffs.append(
                SectionDiff(heading=heading, change="added", before="", after=_clip(new_text))
            )
    return diffs


def _merge_by_heading(chunks: list[Chunk]) -> dict[str, str]:
    """One entry per heading path. A long section split across chunks is still one section."""
    merged: dict[str, str] = {}
    for chunk in chunks:
        key = chunk.heading or "(page)"
        merged[key] = f"{merged[key]}\n\n{chunk.text}" if key in merged else chunk.text
    return merged


def _materially_different(old: str, new: str) -> bool:
    """Ignore re-rendering noise: whitespace, and changes smaller than NOISE_RATIO of the text."""
    left, right = " ".join(old.split()), " ".join(new.split())
    if left == right:
        return False
    import difflib

    ratio = difflib.SequenceMatcher(None, left, right).ratio()
    return (1.0 - ratio) > NOISE_RATIO


def _clip(text: str, limit: int = 600) -> str:
    stripped = text.strip()
    return stripped if len(stripped) <= limit else stripped[:limit] + "…"


SPEC = register(
    ToolSpec(
        name="diff_page",
        description=(
            "Re-fetch a page and compare it with the stored digest: returns whether it changed "
            "and which sections were added, removed or rewritten, matched by heading. Also "
            "returns the fresh page content, so a changed page can be re-digested without "
            "fetching it twice."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="read",
        span_kind="TOOL",
        cost_hint="medium",
    )
)
