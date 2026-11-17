"""Hybrid search over the active snapshots. The hot path's RETRIEVER span."""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.retriever import AnyDocsRetriever
from anydocs.tools.registry import ToolContext, ToolSpec, register


class Input(BaseModel):
    query: str = Field(description="One retrieval query. Rewrite the user's question first.")
    corpora: list[str] = Field(
        default_factory=list, description="Corpus names to search; empty means all active corpora."
    )
    k: int = Field(default=8, ge=1, le=25, description="How many passages to return.")
    version: str = Field(default="", description="Only match pages mentioning this version.")
    section: str = Field(default="", description="Only match passages under this heading.")
    gold_chunk_id: str = Field(
        default="", description="Eval use only: the chunk a dataset example expects."
    )


class Hit(BaseModel):
    node_id: str
    doc_id: str
    corpus: str
    kind: str
    url: str
    title: str = ""
    heading: str = ""
    versions: str = ""
    text: str
    score: float
    snapshot_id: str = ""


class Output(BaseModel):
    hits: list[Hit] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    score_spread: float = 0.0
    searched_corpora: list[str] = Field(default_factory=list)


def run(inp: Input, ctx: ToolContext) -> Output:
    result = AnyDocsRetriever(
        corpora=inp.corpora,
        k=inp.k,
        version=inp.version,
        section=inp.section,
        gold_chunk_id=inp.gold_chunk_id,
    ).search([inp.query])

    # Every stage's candidates and scores, so a retrieval eval can see where a good chunk
    # was lost rather than only that the answer was wrong.
    telemetry.set_attributes(
        ctx.span,
        {
            telemetry.Attr.RETRIEVAL_QUERIES: result.queries,
            telemetry.Attr.RETRIEVAL_STAGES: result.stage_attribute(),
            telemetry.Attr.RETRIEVAL_SCORE_SPREAD: result.score_spread,
            telemetry.Attr.RETRIEVAL_GOLD_RANK: result.gold_rank,
        },
    )

    hits = [
        Hit(
            node_id=h["node_id"],
            doc_id=h["doc_id"],
            corpus=h["corpus"],
            kind=h["kind"],
            url=h.get("url", ""),
            title=h.get("title", ""),
            heading=h.get("heading", ""),
            versions=h.get("versions", ""),
            text=h["text"],
            score=h["score"],
            snapshot_id=h.get("snapshot_id", ""),
        )
        for h in result.hits
    ]
    corpora = sorted({h.corpus for h in hits}) or inp.corpora
    if hits:
        telemetry.update_turn_context(snapshot_id=hits[0].snapshot_id)
    return Output(
        hits=hits,
        queries=result.queries,
        score_spread=result.score_spread,
        searched_corpora=corpora,
    )


SPEC = register(
    ToolSpec(
        name="search_docs",
        description=(
            "Search the ingested documentation. Runs dense and keyword search together, fuses "
            "the results and reranks them. Returns passages with the doc id, heading path and "
            "URL needed to cite them. Use one focused query per call."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="read",
        span_kind="RETRIEVER",
        cost_hint="medium",
    )
)
