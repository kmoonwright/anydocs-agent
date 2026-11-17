"""Our `ToolSpec` declarations, as LangChain tools.

The registry stays the single declaration: it already derives the JSON schema, the policy check
and the span wrapper from one `ToolSpec`, and this adds the LangChain shape to that list rather
than restating any of it. Every call still goes through `registry.invoke`, so the policy gate,
the ledger and `tool.side_effect` work exactly as they did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool

from anydocs.budget import BudgetExceeded
from anydocs.policy import NeedsApproval, Policy, PolicyDenied, default_policy
from anydocs.tools import registry


@dataclass
class Outcomes:
    """What went wrong during one specialist run, for the span and the result notes.

    A tool failure comes back to the model as text so it can route around it, which means the
    run can succeed with a failure inside it. Recording them here is how the failure still
    reaches the span and the answer instead of being swallowed by the recovery.
    """

    denied: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    calls: int = 0


def as_langchain_tool(
    name: str,
    *,
    policy: Policy | None = None,
    requested_by: str = "agent",
    pre_approved: bool = False,
    outcomes: Outcomes | None = None,
    max_calls: int | None = None,
) -> StructuredTool:
    """One registered tool, wrapped for LangGraph.

    Failures the model can act on come back as text — a tool result it can read and route around.
    The two that belong to the turn, `NeedsApproval` and `BudgetExceeded`, are re-raised so they
    reach the orchestrator: an approval suspends the turn and a budget breach ends it.
    """
    spec = registry.get_spec(name)
    pol = policy or default_policy()
    seen = outcomes or Outcomes()

    def run(**kwargs: Any) -> str:
        # The tool-call budget is enforced here rather than asked for in a prompt, and it counts
        # tools that ran: the call that would exceed it is refused before it does any work.
        if max_calls is not None and seen.calls >= max_calls:
            raise BudgetExceeded(
                "specialist tool calls", f"stopped after {max_calls} tool calls"
            )
        seen.calls += 1
        try:
            result = registry.invoke(
                spec.name, kwargs, policy=pol, requested_by=requested_by, auto_approve=pre_approved
            )
        except (NeedsApproval, BudgetExceeded):
            raise
        except PolicyDenied as exc:
            seen.denied.append(spec.name)
            seen.notes.append(str(exc))
            return f"Denied: {exc}"
        except registry.ToolError as exc:
            # Fail visibly: the model sees the failure and may route around it.
            seen.failed.append(str(exc))
            seen.notes.append(str(exc))
            return f"Failed: {exc}"
        except Exception as exc:
            # The backstop. A tool failing in a way nobody anticipated must not take the turn
            # with it.
            seen.failed.append(f"{type(exc).__name__}: {exc}")
            seen.notes.append(f"{spec.name} failed: {type(exc).__name__}: {exc}")
            return f"Failed: {type(exc).__name__}: {exc}"
        return result.model_dump_json()

    return StructuredTool.from_function(
        func=run,
        name=spec.name,
        description=spec.description.strip(),
        args_schema=spec.input_model,
    )


def tools_for(
    names: list[str],
    *,
    policy: Policy | None = None,
    requested_by: str = "agent",
    pre_approved: bool = False,
    outcomes: Outcomes | None = None,
    max_calls: int | None = None,
) -> list[StructuredTool]:
    """A specialist's allowlist, as LangChain tools.

    The allowlist is still enforced by construction: a specialist is only ever handed the tools
    its `spec.yaml` names, so there is nothing outside the list for it to call.
    """
    return [
        as_langchain_tool(
            n,
            policy=policy,
            requested_by=requested_by,
            pre_approved=pre_approved,
            outcomes=outcomes,
            max_calls=max_calls,
        )
        for n in names
    ]
