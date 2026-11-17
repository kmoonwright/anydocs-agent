"""The digest schema and the chunker.

A digest — not the raw page — is the unit of knowledge. Chunks are made from the raw page for
citation fidelity and are indexed alongside digests with a `kind` field, so retrieval can weight
them differently.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

MAX_CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 120


def join_headings(path: list[str]) -> str:
    """A heading path as one readable string, the way a citation shows it."""
    return " > ".join(path) if path else ""


class Fact(BaseModel):
    """One claim, tied to the span of source text that supports it.

    `source_span` is quoted verbatim from the page so the Auditor can verify a claim by string
    containment before spending a judge call on it.
    """

    claim: str
    source_span: str = ""


class Parameter(BaseModel):
    name: str
    type: str = ""
    default: str = ""
    constraint: str = ""
    description: str = ""


class Example(BaseModel):
    language: str = "text"
    code: str
    symbols: list[str] = Field(default_factory=list, description="API symbols the sample uses")
    caption: str = ""


class VersionNotes(BaseModel):
    applies_to: list[str] = Field(default_factory=list)
    deprecated_in: list[str] = Field(default_factory=list)
    notes: str = ""


class QAPair(BaseModel):
    question: str
    answer: str


class Digest(BaseModel):
    """The fixed schema the summarizer fills. Every field is load-bearing somewhere:
    `answers` feeds retrieval and corpus warm-up, `parameters` and `examples` feed `lint_code`,
    `versions` feeds the Auditor's version-mismatch check.
    """

    purpose: str = Field(description="One sentence: what this page is for.")
    facts: list[Fact] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    versions: VersionNotes = Field(default_factory=VersionNotes)
    answers: list[QAPair] = Field(
        default_factory=list, description="3-5 natural questions this page answers"
    )

    def quality_score(self) -> float:
        """Schema completeness weighted by fact-to-source-span coverage.

        Crawl cost scales with pages, so a cheap model fills these; the score is what decides
        whether a page gets re-digested on the reader model.
        """
        answered = [qa for qa in self.answers if qa.question.strip() and qa.answer.strip()]
        checks = [
            bool(self.purpose.strip()),
            len(self.facts) >= 2,
            len(answered) >= 3,
            bool(self.parameters),
            bool(self.examples or self.versions.applies_to),
        ]
        completeness = sum(checks) / len(checks)
        if self.facts:
            grounded = sum(1 for f in self.facts if f.source_span.strip()) / len(self.facts)
        else:
            grounded = 0.0
        return round(0.6 * completeness + 0.4 * grounded, 3)

    def searchable_text(self) -> str:
        """What gets embedded for the digest row: purpose, claims and the questions it answers."""
        parts = [self.purpose]
        parts += [f.claim for f in self.facts]
        parts += [qa.question for qa in self.answers]
        parts += [p.name for p in self.parameters]
        return "\n".join(p for p in parts if p.strip())

    def api_surface(self) -> set[str]:
        """Symbols this page documents, for the Auditor's import and symbol checks."""
        surface = {p.name for p in self.parameters if p.name}
        for ex in self.examples:
            surface.update(s for s in ex.symbols if s)
        return surface


class Chunk(BaseModel):
    """A slice of the raw page, carrying the heading path that makes a citation resolvable."""

    chunk_id: str
    doc_id: str
    heading_path: list[str] = Field(default_factory=list)
    text: str
    position: int = 0

    @computed_field  # serialized, so a citation read back from the API carries the heading
    @property
    def heading(self) -> str:
        """Where in the page this chunk sits, as a citation names it."""
        return join_headings(self.heading_path)


class Doc(BaseModel):
    """A digested page: provenance keys, the digest, and the chunks cited against it."""

    doc_id: str
    corpus: str
    url: str
    canonical_url: str = ""
    title: str = ""
    heading_path: list[str] = Field(default_factory=list)
    version_markers: list[str] = Field(default_factory=list)
    content_hash: str = ""
    fetched_at: str = ""
    digest: Digest
    chunks: list[Chunk] = Field(default_factory=list)
    digest_quality: float = 0.0
    tombstoned: bool = False
    tombstone_reason: str = ""

    def citation(self, chunk_id: str = "") -> dict[str, Any]:
        """The citation footer's payload: corpus, snapshot, URL and heading all resolve."""
        heading = ""
        if chunk_id:
            for chunk in self.chunks:
                if chunk.chunk_id == chunk_id:
                    heading = chunk.heading
                    break
        return {
            "doc_id": self.doc_id,
            "corpus": self.corpus,
            "url": self.canonical_url or self.url,
            "title": self.title,
            "heading": heading or self.heading,
            "versions": self.version_markers,
        }

    @computed_field
    @property
    def heading(self) -> str:
        """The page's own position in the doc set, as distinct from a chunk's within the page."""
        return join_headings(self.heading_path)


# ----------------------------------------------------------------------------------------
# Identity and chunking
# ----------------------------------------------------------------------------------------


def content_hash(text: str) -> str:
    """Stable hash of page content. Drives duplicate-skip on crawl and change detection on
    refresh, so it hashes the cleaned text, not the raw HTML."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def doc_id_for(corpus: str, url: str) -> str:
    """Deterministic per (corpus, URL), so re-crawling a page updates it instead of forking it."""
    digest = hashlib.sha256(f"{corpus}::{url}".encode()).hexdigest()[:12]
    return f"{corpus}-{digest}"


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^```")


def chunk_markdown(text: str, doc_id: str) -> list[Chunk]:
    """Split on heading boundaries, carrying the heading path down.

    Headings are the unit a citation points at, so a chunk never spans two of them. Oversized
    sections split on blank lines; code fences are never cut mid-block.
    """
    lines = text.splitlines()
    sections: list[tuple[list[str], list[str]]] = []
    path: list[str] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        if any(line.strip() for line in buffer):
            sections.append((list(path), list(buffer)))
        buffer.clear()

    for line in lines:
        if _FENCE.match(line.strip()):
            in_fence = not in_fence
            buffer.append(line)
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            path = path[: level - 1]
            while len(path) < level - 1:
                path.append("")
            path.append(title)
            continue
        buffer.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading_path, body in sections:
        body_text = "\n".join(body).strip()
        clean_path = [p for p in heading_path if p]
        for piece in _split_oversized(body_text):
            too_small = len(piece.strip()) < MIN_CHUNK_CHARS
            # Fold a fragment into the previous chunk only when it sits under the same heading.
            # Folding across a heading boundary would erase the heading a citation points at,
            # and a short section is still worth citing.
            if too_small and chunks and chunks[-1].heading_path == clean_path:
                chunks[-1].text = f"{chunks[-1].text}\n\n{piece.strip()}"
                continue
            position = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{position:03d}",
                    doc_id=doc_id,
                    heading_path=clean_path,
                    text=piece.strip(),
                    position=position,
                )
            )
    return chunks


def _split_oversized(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text] if text else []
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    in_fence = False
    for para in text.split("\n\n"):
        fences = para.count("```")
        if size + len(para) > MAX_CHUNK_CHARS and current and not in_fence:
            pieces.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
        if fences % 2 == 1:
            in_fence = not in_fence
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def extract_version_markers(text: str) -> list[str]:
    """Version strings a page mentions, used to filter retrieval and to catch an answer that
    cites v2 docs for a v3 question."""
    patterns = [
        r"\bv(\d+(?:\.\d+)*)\b",
        r"\bversion\s+(\d+(?:\.\d+)*)",
        r"\b(\d{4}-\d{2}-\d{2})\b",  # date-style API versions, e.g. Stripe's
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            marker = match.group(1)
            if marker not in found:
                found.append(marker)
    return found[:10]


def heuristic_digest(markdown: str, title: str = "") -> Digest:
    """A digest built from structure alone, with no model call.

    Used by the offline folder ingest and by tests, so the suite can exercise retrieval and the
    Auditor without an API key. It is deliberately worse than the model's: facts quote whole
    bullet lines rather than the precise span, and its `quality_score` reflects that, so a
    heuristic digest is visibly second-class in the store rather than passing as a real one.
    """
    lines = markdown.splitlines()
    paragraphs = [p.strip() for p in markdown.split("\n\n") if p.strip()]
    first_prose = next(
        (p for p in paragraphs if not p.startswith(("#", "```", "|", "-", "*"))), ""
    )

    facts: list[Fact] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")) and len(stripped) > 20:
            claim = stripped[2:].strip()
            facts.append(Fact(claim=claim, source_span=claim))
        if len(facts) >= 12:
            break

    answers: list[QAPair] = []
    for line in lines:
        match = _HEADING.match(line.strip())
        if match and len(match.group(2)) > 3:
            heading = match.group(2).strip()
            answers.append(QAPair(question=f"What does this page say about {heading}?", answer=""))
        if len(answers) >= 5:
            break

    examples: list[Example] = []
    for block in re.finditer(r"```(\w*)\n(.*?)```", markdown, re.DOTALL):
        examples.append(
            Example(language=block.group(1) or "text", code=block.group(2).strip(), symbols=[])
        )
        if len(examples) >= 5:
            break

    return Digest(
        purpose=(first_prose[:300] or title or "Documentation page."),
        facts=facts,
        examples=examples,
        versions=VersionNotes(applies_to=extract_version_markers(markdown)),
        answers=answers,
    )
