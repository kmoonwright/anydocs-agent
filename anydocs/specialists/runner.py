"""The agent loop every specialist runs.

Two things are enforced here rather than asked for in a prompt: the tool allowlist and the
budget. A specialist that calls a tool outside its list gets a hard error on the span — the
Librarian cannot crawl and the Scout cannot save, by construction rather than by instruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputError
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from anydocs import telemetry
from anydocs.budget import BudgetExceeded, current_ledger
from anydocs.config import get_settings
from anydocs.models.roles import chat_model
from anydocs.policy import Policy, default_policy
from anydocs.prompts import load_prompt
from anydocs.specialists.results import RESULT_SCHEMAS
from anydocs.tools import registry
from anydocs.tools.langchain_tools import Outcomes, tools_for

SPECIALISTS_DIR = Path(__file__).resolve().parent


class SpecError(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    """What the orchestrator grants a specialist. Enforced by this runner."""

    max_tool_calls: int = 8
    max_tokens: int = 20_000
    deadline_ms: int = 30_000

    def as_dict(self) -> dict[str, int]:
        return {
            "max_tool_calls": self.max_tool_calls,
            "max_tokens": self.max_tokens,
            "deadline_ms": self.deadline_ms,
        }


@dataclass(frozen=True)
class Task:
    """The delegation protocol's request half."""

    task: str
    inputs: dict[str, Any] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    corpora: list[str] = field(default_factory=list)


@dataclass
class SpecialistResult:
    """The delegation protocol's response half."""

    specialist: str
    result: BaseModel | None
    confidence: float = 0.0
    cost_usd: float = 0.0
    tool_calls: int = 0
    truncated: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "specialist": self.specialist,
            "result": self.result.model_dump(mode="json") if self.result else None,
            "confidence": self.confidence,
            "cost_usd": round(self.cost_usd, 6),
            "tool_calls": self.tool_calls,
            "truncated": self.truncated,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SpecialistSpec:
    name: str
    model_role: str
    tools: list[str]
    budget: Budget
    result_schema: type[BaseModel]
    description: str = ""
    enabled: bool = True

    @property
    def prompt_version(self) -> str:
        return load_prompt(self.name).version


def load_spec(name: str) -> SpecialistSpec:
    path = SPECIALISTS_DIR / name / "spec.yaml"
    if not path.exists():
        raise SpecError(f"No spec.yaml for specialist {name!r} at {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    schema_name = raw.get("result_schema", "")
    if schema_name not in RESULT_SCHEMAS:
        raise SpecError(
            f"{name}/spec.yaml names result_schema {schema_name!r}, which is not in "
            f"anydocs/specialists/results.py"
        )

    unknown = [t for t in raw.get("tools", []) if t not in registry.all_specs()]
    if unknown:
        raise SpecError(f"{name}/spec.yaml allows unregistered tools: {unknown}")

    budget_raw = raw.get("budget") or {}
    return SpecialistSpec(
        name=name,
        model_role=raw.get("model_role", "reader"),
        tools=list(raw.get("tools", [])),
        budget=Budget(
            max_tool_calls=int(budget_raw.get("max_tool_calls", 8)),
            max_tokens=int(budget_raw.get("max_tokens", 20_000)),
            deadline_ms=int(budget_raw.get("deadline_ms", 30_000)),
        ),
        result_schema=RESULT_SCHEMAS[schema_name],
        description=str(raw.get("description", "")),
        enabled=bool(raw.get("enabled", True)),
    )


def run_specialist(
    name: str,
    task: Task,
    *,
    policy: Policy | None = None,
    model_role: str | None = None,
    requested_by: str = "agent",
) -> SpecialistResult:
    """Run one specialist to a structured result.

    The loop itself is LangChain's `create_agent`; what stays ours is the span, the
    allowlist, the budget and the result contract. `model_role` overrides the spec's role, which
    is how the orchestrator downgrades the Auditor when the turn budget is nearly spent.
    """
    spec = load_spec(name)
    role = model_role or spec.model_role
    budget = task.budget or spec.budget
    pol = policy or default_policy()
    prompt = load_prompt(name)
    ledger = current_ledger()

    # A specialist resends its whole history on every tool call, so its prompt is cached.
    model, cfg = chat_model(role, cache_prompt=True)
    # The allowlist is enforced by construction: the agent is only handed the tools `spec.yaml`
    # names, so there is nothing outside the list for it to reach.
    outcomes = Outcomes()
    tools = tools_for(
        spec.tools,
        policy=pol,
        requested_by=requested_by,
        pre_approved=bool(task.inputs.get("pre_approved")),
        outcomes=outcomes,
        max_calls=budget.max_tool_calls,
    )

    notes: list[str] = []
    truncated = False
    result: BaseModel | None = None
    cost_before = ledger.cost_usd

    with telemetry.specialist_span(
        name, budget=budget.as_dict(), tools_allowed=spec.tools
    ) as span:
        telemetry.set_attributes(
            span,
            {
                telemetry.Attr.INPUT_VALUE: task.task,
                telemetry.Attr.MODEL_ROLE: role,
                telemetry.Attr.MODEL_NAME: cfg.model,
                telemetry.Attr.PROMPT_VERSION: prompt.version,
            },
        )

        # `create_agent` binds the result schema as a tool the model calls to finish: the
        # specialist works, then answers.
        agent = create_agent(
            model,
            tools,
            system_prompt=_assemble_system(prompt.text, task),
            response_format=spec.result_schema,
        )
        try:
            # `recursion_limit` is how LangGraph spells a step budget. Each tool call costs two
            # steps — the model turn that asks for it and the tool turn that answers — so the
            # allowance is twice the tool-call budget plus room to respond.
            final = agent.invoke(
                {"messages": [HumanMessage(_task_message(task))]},
                config={"recursion_limit": budget.max_tool_calls * 2 + 4},
            )
            result = final.get("structured_response")
            if result is None:
                notes.append("the specialist finished without returning a structured result")
        except BudgetExceeded as exc:
            # A budget breach is a partial result tagged `truncated`, which the Auditor reads as
            # a confidence penalty — not an exception the turn has to handle.
            truncated = True
            notes.append(str(exc))
        except GraphRecursionError:
            truncated = True
            notes.append(
                f"stopped after about {budget.max_tool_calls} tool calls "
                f"(budget {budget.max_tool_calls})"
            )
        except StructuredOutputError as exc:
            # The specialist worked but could not deliver its result — most often its answer was
            # cut off at the model's output limit, which arrives as unparseable JSON rather than
            # as a short result. A specialist that cannot report is a partial result like any
            # other: the turn says what is missing instead of failing with a stack trace.
            truncated = True
            notes.append(f"the {name} could not return a usable result: {exc}")

        requested = _tool_calls_made(locals().get("final"), spec.result_schema.__name__)
        tools_called = [t for t in requested if t in spec.tools]
        # A call to a tool outside the list never ran — it was not bound — so it is reported
        # rather than retried, and it costs no budget slot.
        off_list = sorted({t for t in requested if t not in spec.tools})
        for denied in off_list:
            notes.append(f"the {name} is not allowed to call {denied!r}")
        notes.extend(outcomes.notes)

        confidence = _confidence_of(result)
        if truncated:
            confidence = round(confidence * 0.6, 3)

        telemetry.set_attributes(
            span,
            {
                telemetry.Attr.RESULT_CONFIDENCE: confidence,
                telemetry.Attr.RESULT_TRUNCATED: truncated,
                telemetry.Attr.RESULT_NOTES: notes,
                telemetry.Attr.TOOLS_CALLED: tools_called,
                telemetry.Attr.TOOLS_DENIED: off_list + outcomes.denied,
                telemetry.Attr.TOOLS_FAILED: outcomes.failed,
                telemetry.Attr.OUTPUT_VALUE: result.model_dump(mode="json") if result else {},
            },
        )
        if result is not None:
            # The result says what an evaluator needs; publishing it here keeps evals a cheap
            # code check instead of a parse of `output.value`.
            telemetry.set_attributes(span, result.eval_attributes())

    return SpecialistResult(
        specialist=name,
        result=result,
        confidence=confidence,
        cost_usd=ledger.cost_usd - cost_before,
        tool_calls=len(tools_called),
        truncated=truncated,
        notes=notes,
    )


def _tool_calls_made(final: dict[str, Any] | None, result_schema: str) -> list[str]:
    """Every tool the agent asked for, read back off the message history.

    Recorded so a tool-selection eval can compare it against `plan.intent_tools`. The result
    schema is bound as a tool so the model can finish with it; that is answering, not working,
    so it is not counted here.
    """
    if not final:
        return []
    called: list[str] = []
    for message in final.get("messages", []):
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] != result_schema:
                called.append(call["name"])
    return called


def _assemble_system(prompt_text: str, task: Task) -> str:
    settings = get_settings()
    blocks = [prompt_text]
    if settings.operator_notes:
        blocks.append(
            "## Operator notes (standing instructions for this deployment)\n\n"
            + settings.operator_notes
        )
    if task.corpora:
        blocks.append(f"Active corpora for this task: {', '.join(task.corpora)}")
    blocks.append(
        f"Budget: at most {task.budget.max_tool_calls} tool calls. Stop and answer as soon as "
        f"you have what you need. If you run out of budget, answer with whatever you have and "
        f"say what is missing — a partial result that admits the gap is worth more than none."
    )
    return "\n\n".join(blocks)


def _task_message(task: Task) -> str:
    parts = [f"Task: {task.task}"]
    if task.inputs:
        parts.append(f"Inputs:\n{json.dumps(task.inputs, indent=2, default=str)}")
    return "\n\n".join(parts)


def _confidence_of(result: BaseModel | None) -> float:
    """A specialist's own confidence when it reports one; otherwise presence of a result."""
    if result is None:
        return 0.0
    for attribute in ("retrieval_confidence", "confidence"):
        value = getattr(result, attribute, None)
        if isinstance(value, (int, float)):
            return round(float(value), 3)
    return 0.7
