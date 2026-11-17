"""Content guardrails.

Two shipped here: the secrets scanner that runs before anything durable is written, and the
untrusted-content marking that `fetch_page` and `digest_page` rely on. The prompt-injection
check on the user turn runs in two stages: a free pattern prefilter, then the cheap router
model on anything it matches.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from anydocs import telemetry

# Patterns deliberately biased toward false positives: redacting a string that looked like a key
# costs a little retrieval quality, while saving a real credential into a shared snapshot is not
# recoverable.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[abps]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    (
        "private_key_block",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
    ),
    ("bearer_token", re.compile(r"\b[Bb]earer\s+[A-Za-z0-9._~+/-]{24,}={0,2}")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*"
            r"[\"']?([A-Za-z0-9._~+/-]{16,})[\"']?"
        ),
    ),
]

REDACTION = "[REDACTED]"


def scan_secrets(text: str) -> tuple[str, list[str]]:
    """Redact credentials and report what kinds were found.

    Returns the cleaned text and the list of pattern names that matched, which goes into the
    provenance record so the redaction is on the record rather than invisible.
    """
    found: list[str] = []
    cleaned = text
    for name, pattern in _SECRET_PATTERNS:
        if not pattern.search(cleaned):
            continue
        found.append(name)
        cleaned = pattern.sub(
            lambda m: m.group(0).replace(m.group(1), REDACTION) if m.groups() else REDACTION,
            cleaned,
        )
    return cleaned, found


# Phrasings that try to redirect an agent rather than document a product. A hit downgrades the
# turn to read-only tools; it never silently rewrites the user's question.
_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (?:all |any )?(?:previous|prior|above) instructions"),
    re.compile(r"(?i)disregard (?:your|the) (?:system )?(?:prompt|instructions|rules)"),
    re.compile(r"(?i)you are now (?:a|an|in) \w+"),
    re.compile(r"(?i)reveal (?:your|the) (?:system )?prompt"),
    re.compile(r"(?i)print (?:your|the) (?:instructions|system prompt|api key)"),
    re.compile(r"(?i)\b(?:exfiltrate|send (?:the )?(?:contents|data) to)\b"),
]


class _InjectionVerdict(BaseModel):
    attempt: bool = Field(
        description="True only when the message is addressed to the agent as instructions."
    )
    evidence: str = Field(
        default="", description="The phrase that decided it, quoted. Empty means no attempt."
    )


def check_injection(text: str) -> tuple[bool, str]:
    """Return (hit, reason) for the user's turn.

    Two stages. The patterns are a free prefilter: a message matching none of them is let
    through without a model call, which is almost every message. A match goes to the cheap router
    model, because the patterns cannot tell "ignore your instructions" addressed to the agent
    from a question *about* prompt injection — and a documentation agent that refuses to discuss
    prompt injection is useless to the people most likely to ask.

    A hit downgrades the turn to read-only tools and tags the span. It never refuses: the
    question still gets answered, from the store only.
    """
    with telemetry.guardrail_span(check="injection") as span:
        matched = next((p for p in _INJECTION_PATTERNS if p.search(text)), None)
        if matched is None:
            telemetry.set_attributes(
                span, {"guardrail.injection": False, "guardrail.stage": "patterns"}
            )
            return False, ""

        pattern_reason = f"matched {matched.pattern[:48]!r}"
        try:
            verdict = _judge_injection(text)
        except Exception as exc:
            # The prefilter already fired. If the second opinion is unavailable, keep the
            # cautious answer rather than upgrading the turn's permissions on a failure.
            telemetry.set_attributes(
                span,
                {
                    "guardrail.injection": True,
                    "guardrail.stage": "patterns",
                    "guardrail.reason": pattern_reason,
                    "guardrail.model_failed": f"{type(exc).__name__}: {exc}",
                },
            )
            return True, pattern_reason

        reason = (
            f"{pattern_reason}; the router model agreed: {verdict.evidence[:120]}"
            if verdict.attempt
            else ""
        )
        telemetry.set_attributes(
            span,
            {
                "guardrail.injection": verdict.attempt,
                "guardrail.stage": "model",
                "guardrail.reason": reason,
                "guardrail.pattern": pattern_reason,
            },
        )
        return verdict.attempt, reason


def _judge_injection(text: str) -> _InjectionVerdict:
    from langchain_core.messages import HumanMessage, SystemMessage

    from anydocs.models.roles import call_role_json
    from anydocs.prompts import load_prompt

    prompt = load_prompt("injection_check")
    return call_role_json(
        "guardrail.injection_judge",
        "router",
        [SystemMessage(prompt.text), HumanMessage(f"Message to classify:\n\n{text}")],
        _InjectionVerdict,
        prompt_version=prompt.version,
    )


def mirrors_imperative_content(answer: str, sources: list[str]) -> bool:
    """True when the answer repeats imperative sentences lifted from a source.

    The Auditor uses this to flag an answer that may be echoing instructions found in a fetched
    page rather than describing what the page documents.
    """
    imperatives = []
    for source in sources:
        for line in source.splitlines():
            stripped = line.strip(" -*\t")
            if len(stripped) < 20:
                continue
            first = stripped.split()[0].lower() if stripped.split() else ""
            if first in {"ignore", "disregard", "forget", "reveal", "send", "execute", "run"}:
                imperatives.append(stripped.lower())
    lowered = answer.lower()
    return any(imp[:60] in lowered for imp in imperatives)
