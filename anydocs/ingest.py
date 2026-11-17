"""Ingest pipelines shared by the CLI, the HTTP API and the Archivist.

A crawl from a laptop and a crawl triggered by a user run the same functions, so the two produce
identical traces — which is the point of the CLI calling into here rather than shelling out to
the server.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anydocs import telemetry
from anydocs.budget import DailySpend, current_ledger, set_ledger
from anydocs.knowledge.digest import (
    extract_version_markers,
    heuristic_digest,
    utcnow,
)
from anydocs.knowledge.store import create_snapshot, open_snapshot, publish_snapshot
from anydocs.policy import Policy, default_policy
from anydocs.tools import registry
from anydocs.tools._raw_pages import load_raw_page


@dataclass
class IngestReport:
    corpus: str
    snapshot_id: str
    pages_added: int = 0
    pages_unchanged: int = 0
    pages_failed: int = 0
    indexed_nodes: int = 0
    contradictions_found: int = 0
    average_quality: float = 0.0
    cost_usd: float = 0.0
    best_questions: list[str] = field(default_factory=list)
    warm_up: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "snapshot_id": self.snapshot_id,
            "pages_added": self.pages_added,
            "pages_unchanged": self.pages_unchanged,
            "pages_failed": self.pages_failed,
            "indexed_nodes": self.indexed_nodes,
            "contradictions_found": self.contradictions_found,
            "average_quality": round(self.average_quality, 3),
            "cost_usd": round(self.cost_usd, 4),
            "best_questions": self.best_questions,
            "warm_up": self.warm_up,
            "errors": self.errors,
        }


@dataclass
class PageBatch:
    """What a front end hands the pipeline: where to write, and what to write there."""

    pages: Iterable[SourcePage]
    snapshot_id: str = ""
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class SourcePage:
    """One page on its way into the store, however it was obtained.

    Both ingest paths normalize to this before anything is written, which is what makes the
    design doc's claim true by construction: a crawl from a laptop and a crawl triggered by a
    user write identical snapshots because they run the same code from here on.
    """

    url: str
    markdown: str
    title: str = ""
    version_markers: list[str] = field(default_factory=list)
    fetched_at: str = ""
    http_status: int = 0
    content_type: str = ""
    last_modified: str = ""
    robots_decision: str = ""

    def save_arguments(self, corpus: str, snapshot_id: str, digest, quality, model, version):
        return {
            "corpus": corpus,
            "snapshot_id": snapshot_id,
            "url": self.url,
            "markdown": self.markdown,
            "title": self.title,
            "digest": digest.model_dump(mode="json"),
            "version_markers": self.version_markers or extract_version_markers(self.markdown),
            "fetched_at": self.fetched_at or utcnow(),
            "http_status": self.http_status,
            "content_type": self.content_type,
            "last_modified": self.last_modified,
            "robots_decision": self.robots_decision,
            "digest_model": model,
            "digest_prompt_version": version,
            "digest_quality": quality,
            # The pipeline checks conflicts once over everything it wrote, so doing it per page
            # here would duplicate the work and spend the Archivist's budget on it.
            "check_contradictions": False,
        }


def ingest_folder(
    folder: Path,
    corpus: str,
    *,
    base_url: str = "",
    digest: bool = True,
    publish: bool = True,
    snapshot_id: str = "",
    policy: Policy | None = None,
) -> IngestReport:
    """Ingest local markdown files as a corpus.

    `digest=False` fills each digest from page structure instead of calling a model — the
    offline path, used by the test fixtures and by anyone who wants a corpus without an API key.
    """
    files = sorted(p for p in folder.rglob("*.md") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .md files under {folder}")

    def read_files(_: Policy) -> PageBatch:
        def pages():
            for path in files:
                markdown = path.read_text()
                relative = path.relative_to(folder).as_posix()
                yield SourcePage(
                    url=(
                        f"{base_url.rstrip('/')}/{relative}"
                        if base_url
                        else path.resolve().as_uri()
                    ),
                    markdown=markdown,
                    title=_title_of(markdown, relative),
                    content_type="text/markdown",
                    robots_decision="not_checked",
                )

        return PageBatch(snapshot_id=snapshot_id, pages=pages())

    return _ingest(
        read_files,
        corpus,
        label=f"ingest folder {folder}",
        source=str(folder),
        digest=digest,
        publish=publish,
        policy=policy,
        requested_by="operator",
    )


def crawl_and_ingest(
    root_url: str,
    corpus: str,
    *,
    max_pages: int = 50,
    max_depth: int = 3,
    exclude: list[str] | None = None,
    snapshot_id: str = "",
    digest: bool = True,
    publish: bool = True,
    policy: Policy | None = None,
    requested_by: str = "operator",
) -> IngestReport:
    """Crawl a site, digest every page, save each with provenance, then index.

    The Archivist does the same three steps through its own agent loop; this is the direct path
    the CLI uses.
    """
    pol = policy or default_policy()

    def crawl_then_read(_: Policy) -> PageBatch:
        set_ledger(pol.new_ledger())
        crawl = registry.invoke(
            "crawl_site",
            {
                "root_url": root_url,
                "corpus": corpus,
                "max_pages": max_pages,
                "max_depth": max_depth,
                "include": [],
                "exclude": exclude or [],
                "snapshot_id": snapshot_id,
            },
            policy=pol,
            requested_by=requested_by,
        )

        def pages():
            for page in crawl.pages:
                raw = load_raw_page(page.path)
                yield SourcePage(
                    url=raw["url"],
                    markdown=raw["markdown"],
                    title=raw.get("title", ""),
                    version_markers=raw.get("version_markers", []),
                    fetched_at=raw.get("fetched_at", ""),
                    http_status=int(raw.get("status", 0) or 0),
                    content_type=raw.get("content_type", ""),
                    last_modified=raw.get("last_modified", ""),
                    robots_decision=raw.get("robots_decision", ""),
                )

        return PageBatch(
            snapshot_id=crawl.snapshot_id,
            pages=pages(),
            failed=crawl.failed,
            errors=list(crawl.errors),
        )

    return _ingest(
        crawl_then_read,
        corpus,
        label=f"crawl {root_url}",
        source=root_url,
        digest=digest,
        publish=publish,
        policy=pol,
        requested_by=requested_by,
    )


def _ingest(
    prepare: Callable[[Policy], PageBatch],
    corpus: str,
    *,
    label: str,
    source: str,
    digest: bool,
    publish: bool,
    policy: Policy | None,
    requested_by: str,
) -> IngestReport:
    """Digest, save, index, check conflicts and warm up. The one pipeline both front ends use.

    `prepare` is called *inside* the maintenance span rather than before it, so a crawl that
    produces the pages appears under the same root as the saves — which is the trace shape the
    design doc specifies and what `tests/integration/test_archivist.py` asserts.
    """
    pol = policy or default_policy()
    report = IngestReport(corpus=corpus, snapshot_id="")
    qualities: list[float] = []
    questions: list[str] = []
    added: list[str] = []

    with telemetry.turn_context(
        turn_id=uuid.uuid4().hex[:12],
        user_id="system",
        corpus_active=(corpus,),
        policy_name=pol.name,
    ):
        with telemetry.turn_span(message=label, maintenance=True):
            batch = prepare(pol)
            snapshot = (
                open_snapshot(corpus, batch.snapshot_id)
                if batch.snapshot_id
                else create_snapshot(corpus, source=source)
            )
            report.snapshot_id = snapshot.snapshot_id
            report.pages_failed = batch.failed
            report.errors = list(batch.errors)
            telemetry.update_turn_context(snapshot_id=snapshot.snapshot_id)

            for page in batch.pages:
                # One ledger per page. `config/policy.default.yaml` bounds a *turn* — 24 tool
                # calls, 60k tokens, 90 seconds — because a user is waiting. A 200-page crawl
                # exceeds every one of them legitimately, so each page gets its own budget and
                # the daily USD cap carries the job-level limit.
                set_ledger(pol.new_ledger())
                try:
                    digest_obj, quality, model, version = _digest_for(
                        page.markdown, page.url, page.title, digest, pol
                    )
                    saved = registry.invoke(
                        "save_note",
                        page.save_arguments(
                            corpus, snapshot.snapshot_id, digest_obj, quality, model, version
                        ),
                        policy=pol,
                        requested_by=requested_by,
                    )
                except Exception as exc:  # one bad page must not abort the corpus
                    report.pages_failed += 1
                    report.errors.append(f"{page.url}: {type(exc).__name__}: {exc}")
                    continue
                finally:
                    report.cost_usd += current_ledger().cost_usd

                if saved.duplicate:
                    report.pages_unchanged += 1
                else:
                    report.pages_added += 1
                    added.append(saved.doc_id)
                    qualities.append(quality)
                    questions.extend(q.question for q in digest_obj.answers)

            set_ledger(pol.new_ledger())
            report.indexed_nodes = snapshot.reindex()
            report.contradictions_found = _check_contradictions(snapshot, added, digest)
            report.cost_usd += current_ledger().cost_usd

    if publish:
        publish_snapshot(corpus, snapshot.snapshot_id)
    DailySpend().add(report.cost_usd)
    report.average_quality = sum(qualities) / len(qualities) if qualities else 0.0
    report.best_questions = _best_questions(questions)
    report.warm_up = warm_up(corpus, report.best_questions)
    return report


def _check_contradictions(snapshot, doc_ids: list[str], digest: bool) -> int:
    """Find conflicts among the pages this run just wrote.

    Skipped when `digest` is off, because that mode promises no model calls at all. A failure
    here must not fail the ingest: the pages are already saved and correct, and a missing
    conflict edge is a gap, not a corruption.
    """
    if not digest or not doc_ids:
        return 0
    from anydocs.knowledge.links import detect_across

    try:
        return len(detect_across(snapshot, doc_ids))
    except Exception:
        return 0


def _digest_for(
    markdown: str, url: str, title: str, use_model: bool, policy: Policy
) -> tuple[Any, float, str, str]:
    if not use_model:
        digest = heuristic_digest(markdown, title)
        return digest, digest.quality_score(), "heuristic", "heuristic@1"

    from anydocs.models.roles import resolve_role

    result = registry.invoke(
        "digest_page",
        {"markdown": markdown, "url": url, "title": title},
        policy=policy,
        requested_by="operator",
    )
    if result.needs_redigest:
        # One escalation to the stronger reader model, not a loop.
        result = registry.invoke(
            "digest_page",
            {"markdown": markdown, "url": url, "title": title, "role": "reader"},
            policy=policy,
            requested_by="operator",
        )
    return (
        result.digest,
        result.quality,
        resolve_role(result.model_role).model,
        result.prompt_version,
    )


def _title_of(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _best_questions(questions: list[str], limit: int = 3) -> list[str]:
    """The most specific, least overlapping questions the corpus answers.

    Longer questions are more specific; a question sharing most of its words with one already
    chosen adds nothing a user would read as proof the corpus is worth having.
    """
    chosen: list[str] = []
    for question in sorted(dict.fromkeys(questions), key=len, reverse=True):
        words = _distinctive(question)
        if any(_overlap(words, _distinctive(c)) > 0.6 for c in chosen):
            continue
        chosen.append(question)
        if len(chosen) >= limit:
            break
    return chosen


_QUESTION_STOPWORDS = frozenset(
    """
    what how why when where which who does do did is are was were the a an of to for in on with
    this that page say about can i you it its use using set configure configured
    """.split()
)


def _distinctive(question: str) -> set[str]:
    """The words that make one question different from another.

    Without this, a set of questions sharing a stock phrasing — which is exactly what the
    structure-only digest produces — all look like duplicates of each other.
    """
    return {w.strip("?.,:") for w in question.lower().split()} - _QUESTION_STOPWORDS


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def warm_up(corpus: str, questions: list[str]) -> list[str]:
    """Run the new corpus's own best questions once.

    Two reasons: the first real user is not the first query the index has ever served, and a
    question the corpus claims to answer that returns nothing is a sign the ingest went wrong —
    better to see that in the ingest report than in an answer.
    """
    from anydocs.knowledge import retrieval

    served: list[str] = []
    for question in questions:
        try:
            hits = len(retrieval.search([question], corpora=[corpus], k=4, rerank=False).hits)
        except Exception as exc:
            served.append(f"{question}: FAILED ({type(exc).__name__})")
            continue
        served.append(f"{question}: {hits} hits")
    return served


def learn_source(
    source: str,
    *,
    corpus: str = "",
    max_pages: int = 50,
    requested_by: str = "operator",
) -> dict[str, Any]:
    """Workflow 2: learn on request, without a chat turn.

    `source` is a URL to crawl directly, or a query the Scout resolves to a doc root first.
    """
    from anydocs.orchestrator.turn import TurnRequest, run_turn

    pol = default_policy()
    is_url = source.startswith(("http://", "https://"))
    message = (
        f"Learn the documentation at {source}."
        if is_url
        else f"Find and learn the documentation for: {source}"
    )
    request = TurnRequest(
        message=message,
        session_id=f"learn-{uuid.uuid4().hex[:8]}",
        user_id="system",
        corpora=[corpus] if corpus else [],
        overrides={
            "policy": {
                "crawl": pol.section("crawl") | {"max_pages_per_crawl": max_pages},
                # Headless: no user is waiting, so the interactive turn budget does not apply.
                # The daily USD cap still bounds it.
                "budget": pol.budget
                | {
                    "max_wall_ms": max(pol.budget.get("max_wall_ms", 45_000), 15 * 60_000),
                    "max_tokens_per_turn": max(
                        pol.budget.get("max_tokens_per_turn", 60_000), 60_000 * max(1, max_pages)
                    ),
                    "max_tool_calls_per_turn": max(
                        pol.budget.get("max_tool_calls_per_turn", 24), 4 * max(1, max_pages) + 8
                    ),
                },
            }
        },
        maintenance=True,
        allow_learning=True,
        requested_by=requested_by,
    )
    outcome = run_turn(request)
    return {
        "turn_id": outcome.turn_id,
        "trace_id": outcome.trace_id,
        "pending_approval": outcome.pending_approval,
        "report": outcome.answer.answer if outcome.answer else "",
        "specialists": outcome.specialists,
        "cost_usd": outcome.cost_usd,
    }

