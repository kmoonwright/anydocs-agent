"""The final answer: the Librarian's draft, the Auditor's verdict, and the caveats.

The orchestrator applies at most one round of revisions. When confidence stays low, the answer
ships with the Auditor's caveats inline rather than being withheld or quietly cleaned up.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.config import get_settings
from anydocs.models.roles import call_role, resolve_role
from anydocs.prompts import load_prompt
from anydocs.specialists.results import AuditorResult, Citation, LibrarianResult

LOW_CONFIDENCE = 0.5


class FinalAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    notice: str = Field(default="", description="Budget or policy notice shown to the user.")


def synthesize(
    message: str,
    librarian: LibrarianResult | None,
    auditor: AuditorResult | None,
    *,
    learned: dict | None = None,
    notes: list[str] | None = None,
    notice: str = "",
    contradictions: list[dict[str, str]] | None = None,
) -> FinalAnswer:
    """Write the user-facing answer.

    This is the only place free text is produced. Everything it is built from — the draft, the
    citations, the per-claim verdicts — crossed a specialist boundary as structured data.
    """
    settings = get_settings()
    role_cfg = resolve_role("synthesizer")

    citations = librarian.citations if librarian else []
    if auditor:
        confidence = auditor.confidence
    else:
        confidence = librarian.retrieval_confidence if librarian else 0.0
    reasons = list(auditor.reasons) if auditor else []
    caveats = _caveats(librarian, auditor, notes or [], contradictions or [])

    prompt = load_prompt("synthesize")
    with telemetry.synthesize_span(
        role="synthesizer",
        model=role_cfg.model,
        provider=role_cfg.provider,
        prompt_version=prompt.version,
    ) as span:
        completion = call_role(
            telemetry.SYNTHESIZE,
            "synthesizer",
            [
                SystemMessage(_system(settings.operator_notes)),
                HumanMessage(
                    _brief(
                        message,
                        librarian,
                        auditor,
                        learned,
                        caveats,
                        notice,
                        contradictions or [],
                    )
                ),
            ],
            prompt_version=prompt.version,
            # One span for this node: the span is already open, so the call reuses it.
            span=span,
        )
        answer = completion.text.strip()
        telemetry.set_attributes(
            span,
            {
                telemetry.Attr.TURN_ANSWER: answer,
                telemetry.Attr.TURN_CITATIONS: [c.model_dump(mode="json") for c in citations],
                telemetry.Attr.TURN_CONFIDENCE: confidence,
            },
        )

    return FinalAnswer(
        answer=answer,
        citations=citations,
        confidence=round(float(confidence), 3),
        reasons=reasons,
        caveats=caveats,
        notice=notice,
    )


def _system(operator_notes: str) -> str:
    prompt = load_prompt("synthesize")
    blocks = [prompt.text]
    if operator_notes:
        blocks.append("## Operator notes (house style for this deployment)\n\n" + operator_notes)
    return "\n\n".join(blocks)


def _brief(
    message: str,
    librarian: LibrarianResult | None,
    auditor: AuditorResult | None,
    learned: dict | None,
    caveats: list[str],
    notice: str,
    contradictions: list[dict[str, str]],
) -> str:
    parts = [f"User's question: {message}"]

    if librarian and librarian.draft.strip():
        parts.append(f"Draft answer:\n{librarian.draft}")
        if librarian.citations:
            listing = "\n".join(
                f"[{i + 1}] {c.title or c.doc_id} — {c.heading or '(page)'} — {c.url}"
                for i, c in enumerate(librarian.citations)
            )
            parts.append(f"Citations:\n{listing}")
        if librarian.unanswered:
            parts.append("Not answered by the sources:\n- " + "\n- ".join(librarian.unanswered))
    else:
        parts.append(
            "There is no draft: the store held nothing relevant. Say plainly what was searched "
            "and what was not found."
        )

    if auditor:
        verdicts = "\n".join(
            f"- [{c.verdict}] {c.claim}"
            + (f"\n    evidence: {c.evidence[:200]}" if c.evidence else "")
            for c in auditor.claims
        )
        parts.append(f"Auditor verdict: {auditor.verdict}\nPer-claim:\n{verdicts}")
        if auditor.revisions:
            parts.append("Required edits:\n- " + "\n- ".join(auditor.revisions))
        if auditor.code_findings:
            parts.append("Code sample findings:\n- " + "\n- ".join(auditor.code_findings))

    if contradictions:
        # Both readings, with their versions, so the answer can show the disagreement rather
        # than resolve it silently.
        listing = "\n".join(
            f"- {c['from_title']} ({c['from_versions'] or 'version unstated'}) vs "
            f"{c['to_title']} ({c['to_versions'] or 'version unstated'}): {c['evidence']}"
            for c in contradictions
        )
        parts.append(
            "Sources that disagree. Say so explicitly, show both readings with their versions, "
            f"and do not pick one silently:\n{listing}"
        )

    if learned:
        parts.append(
            f"This turn learned new material before answering: "
            f"{learned.get('pages_added', 0)} pages added to corpus "
            f"{learned.get('corpus', '?')}. Mention this in one short sentence."
        )
    if caveats:
        parts.append("Caveats to work in:\n- " + "\n- ".join(caveats))
    if notice:
        parts.append(f"Notice to include verbatim at the end: {notice}")
    return "\n\n".join(parts)


def _caveats(
    librarian: LibrarianResult | None,
    auditor: AuditorResult | None,
    notes: list[str],
    contradictions: list[dict[str, str]] | None = None,
) -> list[str]:
    """Everything the answer must admit to. Failing visibly is the point."""
    caveats: list[str] = []
    if contradictions:
        caveats.append(
            f"Sources disagree on {len(contradictions)} point(s); both readings are shown."
        )
    if auditor:
        unsupported = [c for c in auditor.claims if c.verdict in ("unsupported", "contradicted")]
        if unsupported:
            caveats.append(
                f"{len(unsupported)} claim(s) could not be verified against the cited sources."
            )
        if auditor.version_mismatch:
            caveats.append("The cited sources may not apply to the version asked about.")
        if auditor.contradictions:
            caveats.append("The cited sources do not agree with each other.")
        if auditor.confidence < LOW_CONFIDENCE:
            caveats.append("Confidence in this answer is low; treat it as a starting point.")
    if librarian and librarian.unanswered:
        caveats.append("Some parts of the question were not covered by any source.")
    for note in notes:
        lowered = note.lower()
        if "robots" in lowered:
            caveats.append("A source could not be read because robots.txt disallowed it.")
        elif "budget exceeded" in lowered or "truncated" in lowered:
            caveats.append("This turn ran out of budget before finishing its checks.")
        elif "denied" in lowered:
            caveats.append("An action this answer needed was not permitted by policy.")
        elif "citation(s) not found" in lowered:
            caveats.append(
                "One or more sources in the draft could not be found in the store and were "
                "removed."
            )
    # Order-preserving dedupe: the first phrasing of a caveat is the one that stays.
    return list(dict.fromkeys(caveats))
