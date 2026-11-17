"""The structured results specialists hand back to the orchestrator.

Free text appears only in the final synthesis; everything crossing a specialist boundary is one
of these, so the orchestrator never parses prose.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from anydocs.telemetry import Attr


class SpecialistResultModel(BaseModel):
    """Base for what a specialist hands back.

    `eval_attributes` is how a result says what an evaluator needs on the span. The design doc's
    rule is that the agent writes what evaluators need instead of making them infer it — so a
    groundedness eval reads `audit.claims` directly rather than parsing `output.value`.
    """

    def eval_attributes(self) -> dict[str, Any]:
        return {}


class Citation(BaseModel):
    """Resolvable by construction: corpus, doc, URL and heading all come from the store."""

    doc_id: str
    node_id: str = ""
    url: str = ""
    title: str = ""
    heading: str = ""
    corpus: str = ""
    snapshot_id: str = ""
    quote: str = Field(default="", description="The span of text this citation supports.")


class LibrarianResult(SpecialistResultModel):
    draft: str = Field(description="Draft answer with inline [1]-style citation markers.")
    citations: list[Citation] = Field(default_factory=list)
    retrieval_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How well the store covered the question."
    )
    unanswered: list[str] = Field(
        default_factory=list, description="Sub-questions the store could not answer."
    )
    corpora_searched: list[str] = Field(default_factory=list)

    def eval_attributes(self) -> dict[str, Any]:
        """What a retrieval eval reads: how well the store covered the question, and what it
        could not answer."""
        return {
            Attr.RESULT_CONFIDENCE: self.retrieval_confidence,
            "retrieval.unanswered": self.unanswered,
            "retrieval.citation_count": len(self.citations),
        }


class SourceCandidate(BaseModel):
    url: str
    title: str = ""
    kind: str = Field(default="docs", description="docs | blog | forum | other")
    reason: str = Field(description="Why this is or is not authoritative documentation.")
    estimated_pages: int = Field(default=0, ge=0)
    authority: float = Field(default=0.0, ge=0.0, le=1.0)


class ScoutResult(SpecialistResultModel):
    candidates: list[SourceCandidate] = Field(default_factory=list)
    recommended_root: str = Field(default="", description="The doc root worth crawling, if any.")
    recommended_corpus: str = Field(default="", description="Suggested corpus name.")
    notes: str = ""


class ArchivistResult(SpecialistResultModel):
    corpus: str = ""
    snapshot_id: str = ""
    pages_added: int = 0
    pages_unchanged: int = 0
    pages_failed: int = 0
    pages_blocked: int = 0
    average_digest_quality: float = 0.0
    best_questions: list[str] = Field(
        default_factory=list,
        description="The three questions this new corpus is best at answering.",
    )
    notes: str = ""


class ClaimVerdict(BaseModel):
    claim: str
    verdict: str = Field(description="supported | unsupported | contradicted | unverifiable")
    evidence: str = Field(default="", description="The span of source text that supports it.")
    doc_id: str = ""


class AuditorResult(SpecialistResultModel):
    verdict: str = Field(description="pass | revise")
    claims: list[ClaimVerdict] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    revisions: list[str] = Field(
        default_factory=list, description="Specific edits to make, when the verdict is revise."
    )
    code_findings: list[str] = Field(default_factory=list)
    version_mismatch: bool = False
    contradictions: list[str] = Field(
        default_factory=list,
        description="Conflicts between cited sources that the answer must disclose, each naming "
        "both readings.",
    )

    def eval_attributes(self) -> dict[str, Any]:
        """What a groundedness eval reads, without re-deriving it from the answer text."""
        return {
            Attr.AUDIT_VERDICT: self.verdict,
            Attr.AUDIT_CLAIMS: [c.model_dump(mode="json") for c in self.claims],
            Attr.AUDIT_UNSUPPORTED: sum(
                1 for c in self.claims if c.verdict in ("unsupported", "contradicted")
            ),
            Attr.AUDIT_REASONS: self.reasons,
        }


class AnalystResult(SpecialistResultModel):
    answer: str = ""
    units: str = ""
    rows: list[dict] = Field(default_factory=list)
    query: str = ""
    assumptions: list[str] = Field(default_factory=list)


RESULT_SCHEMAS: dict[str, type[BaseModel]] = {
    "LibrarianResult": LibrarianResult,
    "ScoutResult": ScoutResult,
    "ArchivistResult": ArchivistResult,
    "AuditorResult": AuditorResult,
    "AnalystResult": AnalystResult,
}
