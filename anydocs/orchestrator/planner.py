"""Question -> plan -> specialist tasks.

The planner's output is recorded on the plan span, including the tools it *expected* to be
called, so a tool-selection eval can compare intent against what actually ran.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.models.roles import call_role_json, resolve_role
from anydocs.prompts import load_prompt
from anydocs.specialists.runner import load_spec
from anydocs.tools import registry

SPECIALISTS_README = Path(__file__).resolve().parent.parent / "specialists" / "README.md"
# The order the turn runs them in. Membership is owned by `telemetry.SPECIALISTS`, which cannot
# import from here without inverting the layering — `test_specialists.py` asserts the two agree.
ROUTABLE = ("librarian", "scout", "archivist", "auditor", "analyst")


class Plan(BaseModel):
    specialists: list[str] = Field(
        description="Specialists to involve, in order. The Librarian comes first on any "
        "question about documentation content."
    )
    corpora: list[str] = Field(
        default_factory=list, description="Corpus names to search. Empty means all active."
    )
    intent_tools: list[str] = Field(
        default_factory=list, description="Tools you expect will be called."
    )
    needs_learning: bool = Field(
        default=False, description="True when the store plainly does not cover this product."
    )
    is_numeric: bool = Field(
        default=False, description="True for questions about documentation usage metrics."
    )
    sub_questions: list[str] = Field(default_factory=list)
    rationale: str = Field(default="", description="One or two sentences.")


def enabled_specialists() -> list[str]:
    return [name for name in ROUTABLE if load_spec(name).enabled]


def specialist_descriptions() -> str:
    """The paragraphs from specialists/README.md, which is the planner's only knowledge of them.

    Adding a specialist means adding a paragraph there, not editing this prompt.
    """
    if not SPECIALISTS_README.exists():
        return ""
    text = SPECIALISTS_README.read_text()
    enabled = set(enabled_specialists())
    blocks: list[str] = []
    for match in re.finditer(r"^## (\w+)\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL):
        name, body = match.group(1), match.group(2).strip()
        if name in enabled:
            blocks.append(f"### {name}\n{body}")
    return "\n\n".join(blocks)


def make_plan(
    message: str,
    *,
    corpora: list[str],
    known_corpora: list[str],
    session_summary: str = "",
    profile: str = "",
) -> Plan:
    prompt = load_prompt("orchestrator")
    system = "\n\n".join(
        [
            prompt.text,
            "## Specialists\n\n" + specialist_descriptions(),
            f"## Corpora currently ingested\n\n{', '.join(known_corpora) or '(none yet)'}",
            # `intent_tools` is compared against the tools actually called, so the names have to
            # be real ones. Listing them is cheaper than correcting the model afterwards.
            "## Tool names you may use in `intent_tools`\n\n"
            + ", ".join(sorted(registry.all_specs())),
        ]
        + ([f"## User profile\n\n{profile}"] if profile else [])
    )
    context = [f"Question: {message}"]
    if corpora:
        context.append(f"The caller restricted this turn to corpora: {', '.join(corpora)}")
    if session_summary:
        context.append(f"Conversation so far: {session_summary}")

    role_cfg = resolve_role("router")
    with telemetry.plan_span(
        role="router", model=role_cfg.model, provider=role_cfg.provider
    ) as span:
        telemetry.set_attributes(span, {telemetry.Attr.INPUT_VALUE: message})
        plan = call_role_json(
            telemetry.PLAN,
            "router",
            [SystemMessage(system), HumanMessage("\n\n".join(context))],
            Plan,
            prompt_version=prompt.version,
            span=span,
        )
        plan = _sanitize(plan, corpora)
        telemetry.set_attributes(
            span,
            {
                telemetry.Attr.PLAN_JSON: plan.model_dump(mode="json"),
                telemetry.Attr.PLAN_SPECIALISTS: plan.specialists,
                telemetry.Attr.PLAN_INTENT_TOOLS: plan.intent_tools,
                telemetry.Attr.PROMPT_VERSION: prompt.version,
            },
        )
    return plan


def _sanitize(plan: Plan, requested_corpora: list[str]) -> Plan:
    """Drop specialists that are not routable and keep the invariants the turn relies on.

    The turn runs the Librarian first and the Auditor last regardless of what the plan says;
    normalizing here keeps the recorded plan honest about what will happen.
    """
    enabled = enabled_specialists()
    chosen = [s for s in plan.specialists if s in enabled]
    known_tools = set(registry.all_specs())
    intent = [t for t in plan.intent_tools if t in known_tools]
    invented = [t for t in plan.intent_tools if t not in known_tools]
    if "librarian" not in chosen and not plan.is_numeric:
        chosen.insert(0, "librarian")
    if "auditor" not in chosen:
        chosen.append("auditor")
    ordered = [s for s in ROUTABLE if s in chosen]
    rationale = plan.rationale
    if invented:
        # Recorded rather than silently dropped: a tool-selection eval reading `intent_tools`
        # should not have to wonder whether the planner named something that does not exist.
        rationale = f"{rationale} (dropped unknown tool names: {', '.join(invented)})".strip()
    return plan.model_copy(
        update={
            "specialists": ordered,
            "corpora": requested_corpora or plan.corpora,
            "intent_tools": intent,
            "rationale": rationale,
        }
    )
