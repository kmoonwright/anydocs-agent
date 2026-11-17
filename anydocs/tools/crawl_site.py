"""Polite BFS crawl. Writes raw pages into a new snapshot folder.

Pages land on disk rather than in the tool result: a 500-page crawl must not try to pass itself
through a model's context. The Archivist walks the returned manifest and digests each page.
"""

from __future__ import annotations

import fnmatch
import json
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.digest import content_hash, extract_version_markers, utcnow
from anydocs.knowledge.store import create_snapshot, open_snapshot
from anydocs.tools import http_client
from anydocs.tools.page_text import same_site_links, title_from_html, to_markdown
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register


class Input(BaseModel):
    root_url: str = Field(description="Where the crawl starts. Usually a documentation root.")
    corpus: str = Field(description="Corpus to crawl into; created if it does not exist.")
    include: list[str] = Field(
        default_factory=list,
        description="Glob patterns a URL path must match, e.g. '/docs/*'. Empty means any path "
        "under the root.",
    )
    exclude: list[str] = Field(
        default_factory=list, description="Glob patterns to skip, e.g. '*/changelog/*'."
    )
    max_pages: int = Field(default=50, ge=1, description="Hard cap on pages fetched.")
    max_depth: int = Field(default=3, ge=0, description="Link hops from the root.")
    snapshot_id: str = Field(default="", description="Crawl into this snapshot; new if empty.")


class CrawledPage(BaseModel):
    url: str
    title: str = ""
    content_hash: str = ""
    path: str = ""
    depth: int = 0
    status: int = 0
    bytes: int = 0


class Output(BaseModel):
    corpus: str
    snapshot_id: str
    root_url: str
    pages: list[CrawledPage] = Field(default_factory=list)
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    blocked: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def run(inp: Input, ctx: ToolContext) -> Output:
    _, hard_cap = ctx.policy.crawl_caps()
    if inp.max_pages > hard_cap:
        raise ToolError(
            f"max_pages={inp.max_pages} exceeds policy max_pages_per_crawl={hard_cap}"
        )

    snapshot = (
        open_snapshot(inp.corpus, inp.snapshot_id)
        if inp.snapshot_id
        else create_snapshot(inp.corpus, source=inp.root_url)
    )
    raw_dir = snapshot.root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    root = urlparse(inp.root_url)
    root_prefix = root.path.rsplit("/", 1)[0] or "/"

    queue: deque[tuple[str, int]] = deque([(inp.root_url, 0)])
    seen: set[str] = {inp.root_url}
    seen_hashes: set[str] = set()
    pages: list[CrawledPage] = []
    blocked: list[str] = []
    errors: list[str] = []
    skipped = 0
    failed = 0

    while queue and len(pages) < inp.max_pages:
        url, depth = queue.popleft()
        try:
            page = http_client.fetch(url, policy=ctx.policy)
        except http_client.RobotsBlocked:
            blocked.append(url)
            continue
        except Exception as exc:  # network error, unsafe redirect, timeout
            failed += 1
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            continue

        crawled, problem = _store_page(page, url, depth, raw_dir, seen_hashes)
        if crawled is None:
            if problem == "duplicate":
                skipped += 1
            else:
                failed += 1
                errors.append(f"{url}: {problem}")
            continue
        pages.append(crawled)

        if depth >= inp.max_depth:
            continue
        for link in same_site_links(page.html, page.final_url):
            if link in seen or len(seen) >= inp.max_pages * 8:
                continue
            if not _wanted(link, root_prefix, inp.include, inp.exclude):
                continue
            seen.add(link)
            queue.append((link, depth + 1))

    snapshot.write_manifest(
        crawl={
            "root_url": inp.root_url,
            "fetched": len(pages),
            "skipped": skipped,
            "failed": failed,
            "blocked": len(blocked),
            "max_pages": inp.max_pages,
            "max_depth": inp.max_depth,
            "finished_at": utcnow(),
        }
    )

    telemetry.set_attributes(
        ctx.span,
        {
            telemetry.Attr.PAGES_FETCHED: len(pages),
            telemetry.Attr.PAGES_SKIPPED: skipped,
            telemetry.Attr.PAGES_FAILED: failed,
            telemetry.Attr.SNAPSHOT_ID: snapshot.snapshot_id,
            telemetry.Attr.URL: inp.root_url,
        },
    )
    return Output(
        corpus=inp.corpus,
        snapshot_id=snapshot.snapshot_id,
        root_url=inp.root_url,
        pages=pages,
        fetched=len(pages),
        skipped=skipped,
        failed=failed,
        blocked=blocked,
        errors=errors[:20],
    )


def _store_page(
    page, requested_url: str, depth: int, raw_dir: Path, seen_hashes: set[str]
) -> tuple[CrawledPage | None, str]:
    """Clean one fetched page and write it into the snapshot's raw folder.

    Returns (page, "") on success, or (None, reason) — "duplicate" when the same content has
    already been seen under another URL, which is a skip rather than a failure.
    """
    if page.status >= 400:
        return None, f"HTTP {page.status}"

    markdown = to_markdown(page.html, page.final_url)
    if not markdown.strip():
        return None, "no readable content after cleaning"

    page_hash = content_hash(markdown)
    if page_hash in seen_hashes:
        # Same content under a second URL — a canonical alias or a printer-friendly copy.
        return None, "duplicate"
    seen_hashes.add(page_hash)

    title = title_from_html(page.html)
    path = raw_dir / f"{page_hash}.json"
    path.write_text(
        json.dumps(
            {
                "url": page.final_url,
                "requested_url": requested_url,
                "title": title,
                "markdown": markdown,
                "status": page.status,
                "content_type": page.content_type,
                "last_modified": page.last_modified,
                "robots_decision": page.robots_decision,
                "version_markers": extract_version_markers(f"{title}\n{markdown[:4000]}"),
                "content_hash": page_hash,
                "fetched_at": utcnow(),
                "depth": depth,
            }
        )
    )
    return (
        CrawledPage(
            url=page.final_url,
            title=title,
            content_hash=page_hash,
            path=str(path),
            depth=depth,
            status=page.status,
            bytes=len(markdown),
        ),
        "",
    )


def _wanted(url: str, root_prefix: str, include: list[str], exclude: list[str]) -> bool:
    path = urlparse(url).path or "/"
    if any(fnmatch.fnmatch(path, pattern) for pattern in exclude):
        return False
    if include:
        return any(fnmatch.fnmatch(path, pattern) for pattern in include)
    # No include globs: stay under the root's directory, so a docs crawl does not wander into
    # the marketing site.
    return path.startswith(root_prefix)


SPEC = register(
    ToolSpec(
        name="crawl_site",
        description=(
            "Crawl a documentation site from a root URL, breadth first, honoring robots.txt and "
            "a per-domain rate limit. Writes raw pages into a new snapshot and returns the "
            "manifest: what was fetched, what was skipped as a duplicate, what failed and what "
            "robots blocked. Digest and save the pages afterwards."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="write",
        span_kind="TOOL",
        cost_hint="high",
        policy_key="crawl",
    )
)
