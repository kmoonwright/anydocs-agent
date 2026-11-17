"""What the agent knows and how fresh it is."""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs.knowledge.store import CorpusRegistry, open_snapshot
from anydocs.tools.registry import ToolContext, ToolSpec, register


class Input(BaseModel):
    pass


class CorpusSummary(BaseModel):
    name: str
    default_snapshot: str
    snapshots: int
    page_count: int
    embedding_model: str = ""
    last_refresh: str = ""
    source: str = ""


class Output(BaseModel):
    corpora: list[CorpusSummary] = Field(default_factory=list)


def run(inp: Input, ctx: ToolContext) -> Output:
    registry = CorpusRegistry()
    summaries: list[CorpusSummary] = []
    for entry in registry.list():
        stats = {}
        if entry.default_snapshot:
            try:
                stats = open_snapshot(entry.name, entry.default_snapshot).stats()
            except Exception:
                # A corpus whose snapshot will not open is still worth listing — the agent
                # should be able to say it exists and is broken.
                stats = {}
        summaries.append(
            CorpusSummary(
                name=entry.name,
                default_snapshot=entry.default_snapshot,
                snapshots=len(entry.snapshots),
                page_count=int(stats.get("page_count", 0)),
                embedding_model=str(stats.get("embedding_model", "")),
                last_refresh=str(stats.get("updated_at", "")),
                source=entry.source,
            )
        )
    return Output(corpora=summaries)


SPEC = register(
    ToolSpec(
        name="list_corpora",
        description=(
            "List the documentation corpora this agent has ingested, with their default "
            "snapshot, page count and last refresh time. Call this when you need to know what "
            "is available before searching, or when the user asks what you know about."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="read",
        span_kind="TOOL",
        cost_hint="low",
    )
)
