# Adding a tool

One file plus one registry line. Registration generates the LLM tool schema, the MCP tool
definition, the permission check and the span wrapper — you write none of those.

## 1. Write the file

`anydocs/tools/<name>.py` contains exactly four things.

```python
"""One sentence on what this tool is for, and anything surprising about it."""

from __future__ import annotations

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register


class Input(BaseModel):
    target: str = Field(description="What the model should pass. This text reaches the model.")
    limit: int = Field(default=10, ge=1, le=50)


class Output(BaseModel):
    found: list[str] = Field(default_factory=list)
    notes: str = ""


def run(inp: Input, ctx: ToolContext) -> Output:
    # ctx.span is for your domain attributes. ctx.policy and ctx.ledger are already enforced
    # around you — do not re-check them.
    result = do_the_thing(inp.target, inp.limit)
    telemetry.set_attributes(ctx.span, {telemetry.Attr.URL: inp.target})
    if not result:
        raise ToolError(f"nothing at {inp.target}")  # the model sees this and may route around
    return Output(found=result)


SPEC = register(
    ToolSpec(
        name="my_tool",
        description=(
            "What this does and when to call it. Write it for the model: it is the only "
            "documentation the model gets."
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="read",      # read | write | delete
        span_kind="TOOL",        # or RETRIEVER, LLM, ...
        cost_hint="low",         # low | medium | high
        policy_key="none",       # crawl | write | delete | web_search | none
    )
)
```

### Field descriptions are prompt text

Every `Field(description=...)` is sent to the model as part of the tool schema. Write them as
instructions, not as type documentation: "One focused query; rewrite the user's question first"
beats "the query string".

### Choosing `side_effect` and `policy_key`

`side_effect` is what shows on the span as `tool.side_effect` and what the MCP annotations are
derived from. `policy_key` is what gates the call.

A tool with `side_effect` other than `read` **must** declare a `policy_key` — `ToolSpec` raises
at import otherwise. The single exception is `digest_page`, which is a pure function over
content that was already fetched.

### Untrusted inputs

Anything in `Input` came from a model. A URL goes through `policy.check_domain` (the HTTP client
does this for you). A filesystem path goes through `anydocs/tools/_raw_pages.py`, which confines
it to the data directory. Never open a path a model supplied without that check.

### Heavy or optional dependencies

Every tool module is imported at startup by `registry.load_tools()`, including on the Vercel
function, which does not install the `crawl` extra. Import an optional package inside the function
that needs it, behind a helper that raises `ToolError` naming the missing extra, the way
`page_text.trafilatura()` does. A top-level import of an extra fails the whole process, not the
one tool. `tests/unit/test_registry.py::test_tools_load_without_the_crawl_extra` checks this.

## 2. Register it

Add the module name to `TOOL_MODULES` in `anydocs/tools/registry.py`. That is the whole
registration step; the module registers itself on import.

## 3. Add it to an allowlist

A registered tool is callable by nobody until a specialist is allowed to call it. Add the name
to `tools:` in `anydocs/specialists/<name>/spec.yaml`. The runner enforces the list, so this is
a permission grant, not a hint.

## 4. Test it

Two tests, both in the existing style:

- **Unit**, in `tests/unit/`: the input contract, the failure modes, and what it returns. Use a
  recorded fixture rather than the network — `respx` for HTTP, `tests/fixtures/` for data.
- **Integration**, in `tests/integration/`: assert the span it emits. The `spans` fixture gives
  you names, kinds, parent/child shape and attributes.

```python
def test_my_tool_records_what_it_found(spans, relayr_corpus):
    result = registry.invoke("my_tool", {"target": "x"})
    assert spans.attr("my_tool", Attr.TOOL_SIDE_EFFECT) == "read"
    assert spans.attr("my_tool", Attr.TOOL_OUTCOME) == "ran"
```
