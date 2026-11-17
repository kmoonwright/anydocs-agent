"""Related-doc edges, and the contradiction detection that produces them.

A link is how two pages are related: one supersedes another, one is an example of another, or
two of them disagree. Links do two jobs — they expand retrieval by one hop, and they are how an
answer comes to say "sources disagree" instead of silently picking a side.

Contradictions are found at ingest rather than at answer time. The comparison costs one cheap
model call per overlapping pair, paid once when a page is saved, and the answer path then only
has to read the edges.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.store import Snapshot

LINK_KINDS = ("supersedes", "see-also", "contradicts", "example-of")
# Kinds where direction carries no meaning: "A contradicts B" is the same fact as "B contradicts
# A". Stored unordered, or one disagreement would be reported as two.
SYMMETRIC_KINDS = ("contradicts", "see-also")
# Kinds that pull a related doc into the result set. `contradicts` deliberately does not: a
# conflicting page is surfaced as a conflict, not promoted as if it were more evidence.
EXPANDING_KINDS = ("see-also", "supersedes")
MAX_EXPANSION = 6


class Link(BaseModel):
    from_doc_id: str
    to_doc_id: str
    kind: str
    evidence: str = Field(default="", description="Why this edge exists, quoted where possible.")
    recorded_at: str = ""
    recorded_by: str = "agent"


def record(snapshot: Snapshot, link: Link) -> Link:
    """Append an edge. Idempotent on (from, to, kind): re-recording does not duplicate."""
    stamped = link.model_copy(
        update={"recorded_at": link.recorded_at or datetime.now(UTC).isoformat(timespec="seconds")}
    )
    for other in load_links(snapshot):
        if other.kind != stamped.kind:
            continue
        same = (other.from_doc_id, other.to_doc_id) == (stamped.from_doc_id, stamped.to_doc_id)
        mirrored = (other.from_doc_id, other.to_doc_id) == (
            stamped.to_doc_id,
            stamped.from_doc_id,
        )
        if same or (mirrored and stamped.kind in SYMMETRIC_KINDS):
            return other
    snapshot.append_jsonl(snapshot.links_path, stamped.model_dump())
    return stamped


def load_links(snapshot: Snapshot) -> list[Link]:
    return [Link.model_validate(row) for row in snapshot.read_jsonl(snapshot.links_path)]


def links_for(snapshot: Snapshot, doc_ids: list[str], *, kinds: tuple[str, ...] = ()) -> list[Link]:
    """Every edge touching one of these docs, in either direction.

    Direction-agnostic on purpose: "A contradicts B" is the same fact as "B contradicts A", and
    an answer citing only B should still learn about the conflict.
    """
    wanted = set(doc_ids)
    return [
        link
        for link in load_links(snapshot)
        if (link.from_doc_id in wanted or link.to_doc_id in wanted)
        and (not kinds or link.kind in kinds)
    ]


def contradictions_for(
    snapshots: list[Snapshot], doc_ids: list[str]
) -> list[dict[str, str]]:
    """Conflicts involving the cited docs, as the synthesizer and Auditor need them."""
    out: list[dict[str, str]] = []
    for snapshot in snapshots:
        docs = snapshot.docs()
        for link in links_for(snapshot, doc_ids, kinds=("contradicts",)):
            left, right = docs.get(link.from_doc_id), docs.get(link.to_doc_id)
            out.append(
                {
                    "from_doc_id": link.from_doc_id,
                    "to_doc_id": link.to_doc_id,
                    "from_title": left.title if left else link.from_doc_id,
                    "to_title": right.title if right else link.to_doc_id,
                    "from_url": (left.canonical_url or left.url) if left else "",
                    "to_url": (right.canonical_url or right.url) if right else "",
                    "from_versions": " ".join(left.version_markers) if left else "",
                    "to_versions": " ".join(right.version_markers) if right else "",
                    "evidence": link.evidence,
                }
            )
    return out


def expand(snapshot: Snapshot, doc_ids: list[str], *, limit: int = MAX_EXPANSION) -> list[str]:
    """One hop of query expansion: doc ids reachable by a `see-also` or `supersedes` edge.

    One hop, not transitive closure — two hops through a doc set with dense see-also edges pulls
    in the whole corpus and the retriever stops discriminating.
    """
    seen = set(doc_ids)
    found: list[str] = []
    for link in links_for(snapshot, doc_ids, kinds=EXPANDING_KINDS):
        for candidate in (link.from_doc_id, link.to_doc_id):
            if candidate not in seen:
                seen.add(candidate)
                found.append(candidate)
            if len(found) >= limit:
                return found
    return found


# ----------------------------------------------------------------------------------------
# Contradiction detection, run at ingest
# ----------------------------------------------------------------------------------------


class _Conflict(BaseModel):
    """The comparison's answer.

    `shared_subject` comes first and is answered before the verdict on purpose: the commonest
    false positive is comparing two statements about different things, and naming the subject
    out loud is what catches it.
    """

    shared_subject: str = Field(
        description="The one thing both pages make a claim about — a parameter, a limit, a "
        "behaviour. Empty if there isn't one, in which case there is no conflict."
    )
    quote_a: str = Field(default="", description="The claim from page A, verbatim.")
    quote_b: str = Field(default="", description="The claim from page B, verbatim.")
    same_version: bool = Field(
        default=False,
        description="True when both claims are about the same version. Two pages describing "
        "different versions are a changelog, not a conflict.",
    )
    conflicts: bool = Field(
        description="True only when both quotes describe the same subject and the same version, "
        "and cannot both be true."
    )
    explanation: str = Field(default="", description="One sentence on what disagrees.")


# Words that look like something a doc set names rather than something English does: a
# snake_case or camelCase identifier, or a word in backticks. Two pages documenting the same
# parameter share these; they rarely share a section heading, because they describe it from
# different angles.
_IDENTIFIER = re.compile(
    r"`([^`\n]{2,40})`"  # anything the docs put in backticks
    r"|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b"  # snake_case
    r"|\b([a-z]+[A-Z]\w+)\b"  # camelCase
)
# Common across every page of a doc set, so they say nothing about whether two pages overlap.
_TOO_COMMON = frozenset(
    "api_key sdk_key client_id user_id true false none null str int bool dict list".split()
)


def _signals(doc) -> set[str]:
    """What a page is about, as tokens two pages can be compared on."""
    text = " ".join(
        [doc.title or "", doc.digest.purpose]
        + [f.claim for f in doc.digest.facts]
        + [p.name for p in doc.digest.parameters]
        + [h for chunk in doc.chunks for h in chunk.heading_path]
    )
    tokens = {
        (a or b or c).strip().lower() for a, b, c in _IDENTIFIER.findall(text) if (a or b or c)
    }
    tokens |= {s.lower() for s in doc.digest.api_surface()}
    return {t for t in tokens if t and t not in _TOO_COMMON}


def candidates_for(snapshot: Snapshot, doc_id: str, *, limit: int = 4) -> list[str]:
    """Docs worth comparing a newly saved page against.

    Overlap on the identifiers a page documents, not on its headings: two pages that both
    document `max_attempts` are where a disagreement about `max_attempts` would live, and they
    almost never share a section title. Comparing every pair is quadratic, so this is what keeps
    the ingest cost linear in practice.
    """
    docs = snapshot.docs()
    target = docs.get(doc_id)
    if target is None:
        return []

    target_signals = _signals(target)
    if not target_signals:
        return []

    scored: list[tuple[int, str]] = []
    for other in snapshot.live_docs():
        if other.doc_id == doc_id:
            continue
        overlap = len(target_signals & _signals(other))
        if overlap:
            scored.append((overlap, other.doc_id))
    scored.sort(reverse=True)
    return [doc_id_ for _, doc_id_ in scored[:limit]]


def detect_contradictions(
    snapshot: Snapshot, doc_id: str, *, role: str = "summarizer", limit: int = 4
) -> list[Link]:
    """Compare a newly saved digest against overlapping ones and record what disagrees.

    Returns the edges it recorded. A page that contradicts nothing records nothing, which is the
    common case and costs one model call per overlapping candidate.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from anydocs.models.roles import call_role_json
    from anydocs.prompts import load_prompt

    docs = snapshot.docs()
    target = docs.get(doc_id)
    if target is None:
        return []

    prompt = load_prompt("detect_contradiction")
    recorded: list[Link] = []

    for candidate_id in candidates_for(snapshot, doc_id, limit=limit):
        other = docs.get(candidate_id)
        if other is None:
            continue
        verdict = call_role_json(
            "detect_contradiction",
            role,
            [SystemMessage(prompt.text), HumanMessage(_comparison(target, other))],
            _Conflict,
            prompt_version=prompt.version,
        )
        rejected = _implausible(verdict)
        if rejected:
            # The model said conflict; the quotes say otherwise. Recorded rather than silently
            # dropped, because a detector that is wrong a lot is worth knowing about.
            telemetry.annotate(**{"contradiction.rejected": f"{candidate_id}: {rejected}"})
            continue
        evidence = f"Page A: {verdict.quote_a!r} | Page B: {verdict.quote_b!r}"
        recorded.append(
            record(
                snapshot,
                Link(
                    from_doc_id=doc_id,
                    to_doc_id=candidate_id,
                    kind="contradicts",
                    evidence=evidence[:600],
                    recorded_by="archivist",
                ),
            )
        )
    return recorded


MAX_COMPARISONS = 60


def detect_across(
    snapshot: Snapshot,
    doc_ids: list[str],
    *,
    role: str = "summarizer",
    max_comparisons: int = MAX_COMPARISONS,
) -> list[Link]:
    """Check a batch of newly written docs for conflicts, with a hard cap on model calls.

    Run as a post-pass rather than inside `save_note`, for two reasons. A 500-page crawl would
    otherwise make thousands of nested comparison calls, and those calls would come out of the
    Archivist's own per-turn budget and truncate it. A cap here means a large ingest checks the
    pages most likely to conflict and says how many it skipped, instead of either blowing the
    budget or silently checking nothing.
    """
    found: list[Link] = []
    spent = 0
    for doc_id in doc_ids:
        if spent >= max_comparisons:
            break
        candidates = candidates_for(snapshot, doc_id)
        allowance = max(0, max_comparisons - spent)
        found.extend(
            detect_contradictions(
                snapshot, doc_id, role=role, limit=min(len(candidates), allowance)
            )
        )
        spent += min(len(candidates), allowance)
    # Both pages of a pair get compared, and `record` returns the existing edge for the second
    # one. Counting both would report one disagreement as two.
    return _unique(found)


def _unique(found: list[Link]) -> list[Link]:
    seen: set[tuple] = set()
    out: list[Link] = []
    for link in found:
        endpoints = (
            frozenset((link.from_doc_id, link.to_doc_id))
            if link.kind in SYMMETRIC_KINDS
            else (link.from_doc_id, link.to_doc_id)
        )
        key = (link.kind, endpoints)
        if key not in seen:
            seen.add(key)
            out.append(link)
    return out


def _implausible(verdict: _Conflict) -> str:
    """Reject a claimed conflict the quotes do not support.

    A cheap guard over a cheap model. Every check here corresponds to a false positive observed
    on real documentation: two statements about different subjects, a version difference
    reported as a disagreement, and two quotes that state the same number.
    """
    if not verdict.conflicts:
        return "model reported no conflict"
    if not verdict.quote_a.strip() or not verdict.quote_b.strip():
        return "no quotes supplied"
    if not verdict.shared_subject.strip():
        return "no shared subject named"
    if not verdict.same_version:
        return "the claims describe different versions, which is a changelog not a conflict"

    subject_tokens = {t for t in re.split(r"\W+", verdict.shared_subject.lower()) if len(t) > 3}
    a_tokens = {t for t in re.split(r"\W+", verdict.quote_a.lower()) if len(t) > 3}
    b_tokens = {t for t in re.split(r"\W+", verdict.quote_b.lower()) if len(t) > 3}
    if subject_tokens and not (subject_tokens & a_tokens and subject_tokens & b_tokens):
        return "neither quote mentions the subject they are supposed to disagree about"

    # Two numeric claims where one quote's values are contained in the other's are the same
    # claim at different levels of detail: "maximum 3600" against "default 600, maximum 3600".
    # A real numeric conflict states a value the other quote does not mention at all.
    numbers_a = set(re.findall(r"\d+(?:\.\d+)?", verdict.quote_a))
    numbers_b = set(re.findall(r"\d+(?:\.\d+)?", verdict.quote_b))
    if numbers_a and numbers_b and (numbers_a <= numbers_b or numbers_b <= numbers_a):
        return "one quote's numbers are contained in the other's, so they agree"

    return ""


def _comparison(left, right) -> str:
    def side(doc) -> str:
        facts = "\n".join(f"- {f.claim}" for f in doc.digest.facts[:12])
        params = "\n".join(
            f"- {p.name}: type={p.type or '?'} default={p.default or '?'} {p.constraint}".rstrip()
            for p in doc.digest.parameters[:12]
        )
        return (
            f"Title: {doc.title}\nURL: {doc.canonical_url or doc.url}\n"
            f"Applies to: {', '.join(doc.version_markers) or 'unstated'}\n"
            f"Purpose: {doc.digest.purpose}\n"
            f"Facts:\n{facts or '(none)'}\n"
            f"Parameters:\n{params or '(none)'}"
        )

    return f"--- PAGE A ---\n{side(left)}\n\n--- PAGE B ---\n{side(right)}"
