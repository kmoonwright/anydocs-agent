"""Static checks on a code sample, against the API surface of the docs it was cited from.

v1 is Python only: parse, lint, then check that every imported and referenced symbol appears in
the cited digests' parameters and examples. Other languages return `skipped` rather than a false
pass — a green check the Auditor cannot back up is worse than no check.
"""

from __future__ import annotations

import ast
import io

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.knowledge.store import open_active
from anydocs.tools.registry import ToolContext, ToolSpec, register

SUPPORTED = ("python", "py")


class Input(BaseModel):
    code: str = Field(description="The code sample to check.")
    language: str = Field(default="python", description="Language of the sample.")
    context_doc_ids: list[str] = Field(
        default_factory=list,
        description="Doc ids the sample was drawn from; their API surface is what symbols are "
        "checked against.",
    )


class Finding(BaseModel):
    severity: str  # error | warning | info
    kind: str  # syntax | lint | unknown_symbol | unverifiable
    message: str
    line: int = 0


class Output(BaseModel):
    status: str  # pass | fail | skipped
    language: str
    findings: list[Finding] = Field(default_factory=list)
    symbols_used: list[str] = Field(default_factory=list)
    symbols_unknown: list[str] = Field(default_factory=list)
    api_surface_size: int = 0


def run(inp: Input, ctx: ToolContext) -> Output:
    language = inp.language.lower().strip()
    if language not in SUPPORTED:
        return Output(
            status="skipped",
            language=language,
            findings=[
                Finding(
                    severity="info",
                    kind="unverifiable",
                    message=f"No checker for {language!r}; the sample was not verified.",
                )
            ],
        )

    findings: list[Finding] = []

    try:
        tree = ast.parse(inp.code)
    except SyntaxError as exc:
        findings.append(
            Finding(
                severity="error",
                kind="syntax",
                message=f"{exc.msg}",
                line=exc.lineno or 0,
            )
        )
        telemetry.set_attributes(
            ctx.span,
            {
                "lint.findings": [f.model_dump() for f in findings],
                "lint.status": "fail",
                "lint.unknown_symbols": [],
            },
        )
        return Output(status="fail", language="python", findings=findings)

    findings.extend(_pyflakes(inp.code))
    symbols = sorted(_referenced_symbols(tree))
    surface = _api_surface(inp.context_doc_ids)

    unknown: list[str] = []
    if surface:
        unknown = [s for s in symbols if s not in surface and not _is_stdlib_ish(s)]
        for symbol in unknown:
            findings.append(
                Finding(
                    severity="warning",
                    kind="unknown_symbol",
                    message=(
                        f"`{symbol}` does not appear in the API surface of the cited docs. "
                        f"Either the sample is wrong or the citation is."
                    ),
                )
            )
    else:
        findings.append(
            Finding(
                severity="info",
                kind="unverifiable",
                message=(
                    "No cited docs supplied, so symbols were not checked against an API "
                    "surface."
                ),
            )
        )

    status = "fail" if any(f.severity == "error" for f in findings) else "pass"
    telemetry.set_attributes(
        ctx.span,
        {
            "lint.findings": [f.model_dump() for f in findings],
            "lint.status": status,
            "lint.unknown_symbols": unknown,
        },
    )
    return Output(
        status=status,
        language="python",
        findings=findings,
        symbols_used=symbols,
        symbols_unknown=unknown,
        api_surface_size=len(surface),
    )


def _pyflakes(code: str) -> list[Finding]:
    from pyflakes.api import check
    from pyflakes.reporter import Reporter

    out, err = io.StringIO(), io.StringIO()
    check(code, "sample.py", Reporter(out, err))
    findings: list[Finding] = []
    for line in out.getvalue().splitlines():
        parts = line.split(":", 3)
        message = parts[-1].strip()
        try:
            lineno = int(parts[1])
        except (IndexError, ValueError):
            lineno = 0
        severity = "error" if "undefined name" in message else "warning"
        findings.append(Finding(severity=severity, kind="lint", message=message, line=lineno))
    return findings


def _referenced_symbols(tree: ast.AST) -> set[str]:
    """Imported modules and names, plus attribute chains — what a docs page would name."""
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                symbols.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                symbols.add(node.module.split(".")[0])
            symbols.update(a.name for a in node.names)
        elif isinstance(node, ast.Attribute):
            symbols.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            symbols.add(node.arg)
    return {s for s in symbols if s and not s.startswith("_")}


def _api_surface(doc_ids: list[str]) -> set[str]:
    if not doc_ids:
        return set()
    surface: set[str] = set()
    for snapshot in open_active():
        for doc_id in doc_ids:
            doc = snapshot.get_doc(doc_id)
            if doc:
                surface.update(doc.digest.api_surface())
    return surface


# Names that are Python's own or near-universal; flagging them as "not in the docs" is noise.
_COMMON = frozenset(
    """
    os sys json time re math typing datetime pathlib asyncio logging dataclasses collections
    itertools functools print len range str int float dict list set tuple bool bytes open
    append extend items keys values get format join split strip lower upper encode decode
    read write close dumps loads main self cls args kwargs
    """.split()
)


def _is_stdlib_ish(symbol: str) -> bool:
    return symbol in _COMMON


SPEC = register(
    ToolSpec(
        name="lint_code",
        description=(
            "Check a code sample without running it: parse it, lint it, and verify that the "
            "symbols it uses appear in the API surface of the docs it was cited from. Returns "
            "`skipped` for languages with no checker rather than a false pass."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="read",
        span_kind="TOOL",
        cost_hint="low",
    )
)
