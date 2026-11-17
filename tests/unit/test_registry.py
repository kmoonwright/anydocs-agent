"""The registry generates the wrapper; the tool author writes the declaration."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from anydocs import telemetry
from anydocs.budget import BudgetExceeded, Ledger, set_ledger
from anydocs.policy import NeedsApproval, PolicyDenied, policy_with_overrides
from anydocs.telemetry import Attr
from anydocs.tools import registry
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec


def test_the_catalog_is_fourteen_tools_split_read_and_write():
    specs = registry.all_specs()
    assert len(specs) == 14
    reads = {n for n, s in specs.items() if s.side_effect == "read"}
    writes = {n for n, s in specs.items() if s.side_effect == "write"}
    deletes = {n for n, s in specs.items() if s.side_effect == "delete"}
    assert reads == {
        "search_docs",
        "get_doc",
        "list_corpora",
        "web_search",
        "fetch_page",
        "diff_page",
        "query_metrics",
        "lint_code",
    }
    assert writes == {"crawl_site", "digest_page", "save_note", "link_sources", "schedule_refresh"}
    assert deletes == {"forget"}


def test_every_non_read_tool_names_a_policy_key_except_digest_page():
    for name, spec in registry.all_specs().items():
        if spec.side_effect == "read":
            continue
        if name == "digest_page":
            # The deliberate exception: a pure function over content already fetched.
            assert spec.policy_key == "none"
            continue
        assert spec.policy_key != "none", f"{name} writes but names no policy key"


def test_a_write_tool_without_a_policy_key_is_rejected_at_declaration():
    class Inp(BaseModel):
        pass

    with pytest.raises(ValueError, match="must declare a policy_key"):
        ToolSpec(
            name="sneaky_write",
            description="x",
            input_model=Inp,
            output_model=Inp,
            run=lambda i, c: Inp(),
            side_effect="write",
        )


def test_a_tool_exposes_its_input_model_to_langchain():
    """LangChain takes the Pydantic model directly as `args_schema`, so the schema the model
    sees is the one the tool declared — there is no second copy to drift."""
    from anydocs.tools.langchain_tools import as_langchain_tool

    tool = as_langchain_tool("save_note")
    assert tool.name == "save_note"
    assert tool.args_schema is registry.get_spec("save_note").input_model
    assert "corpus" in tool.args


def test_invalid_arguments_become_a_tool_error_not_a_crash(spans):
    with pytest.raises(ToolError, match="invalid arguments"):
        registry.invoke("search_docs", {"query": "x", "k": 999})


def test_ask_mode_suspends_with_a_resume_token(spans, data_dir):
    policy = policy_with_overrides({"crawl": {"mode": "ask"}})
    with pytest.raises(NeedsApproval) as excinfo:
        registry.invoke(
            "crawl_site",
            {"root_url": "https://docs.example.com/", "corpus": "x", "max_pages": 10},
            policy=policy,
        )
    assert excinfo.value.resume_token
    assert excinfo.value.payload["estimated_pages"] == 10
    # The pause is a span, so the approval moment is visible in the trace.
    assert spans.has(telemetry.APPROVAL_PENDING)
    assert spans.attr(telemetry.APPROVAL_PENDING, Attr.POLICY_ESTIMATE_PAGES) == 10


def test_off_mode_denies_and_records_the_decision(spans, data_dir):
    policy = policy_with_overrides({"crawl": {"mode": "off"}})
    with pytest.raises(PolicyDenied):
        registry.invoke(
            "crawl_site",
            {"root_url": "https://docs.example.com/", "corpus": "x"},
            policy=policy,
        )
    assert spans.attr(telemetry.APPROVAL_DECISION, Attr.POLICY_DECISION) == "denied"


def test_a_denied_call_still_costs_a_tool_call(spans, data_dir):
    """The cap bounds the loop; being refused is not a free retry."""
    ledger = Ledger(max_tool_calls=10)
    set_ledger(ledger)
    policy = policy_with_overrides({"crawl": {"mode": "off"}})
    with pytest.raises(PolicyDenied):
        registry.invoke(
            "crawl_site", {"root_url": "https://x.example.com/", "corpus": "x"}, policy=policy
        )
    assert ledger.tool_calls == 1


def test_tool_call_budget_is_enforced_by_the_ledger(spans, relayr_corpus):
    ledger = Ledger(max_tool_calls=2)
    set_ledger(ledger)
    registry.invoke("list_corpora", {})
    registry.invoke("list_corpora", {})
    with pytest.raises(BudgetExceeded, match="max_tool_calls_per_turn"):
        registry.invoke("list_corpora", {})


def test_span_carries_side_effect_and_kind(spans, relayr_corpus):
    registry.invoke("search_docs", {"query": "retries", "k": 3})
    assert spans.attr("search_docs", Attr.TOOL_SIDE_EFFECT) == "read"
    assert spans.attr("search_docs", Attr.SPAN_KIND) == "RETRIEVER"


def test_tool_context_is_what_tools_get(spans, relayr_corpus):
    captured = {}

    class Inp(BaseModel):
        pass

    class Out(BaseModel):
        ok: bool = True

    def run(inp: Inp, ctx: ToolContext) -> Out:
        captured["policy"] = ctx.policy.name
        captured["requested_by"] = ctx.requested_by
        return Out()

    registry.register(
        ToolSpec(
            name="_probe",
            description="test probe",
            input_model=Inp,
            output_model=Out,
            run=run,
        )
    )
    try:
        registry.invoke("_probe", {}, requested_by="operator")
        assert captured["requested_by"] == "operator"
        assert captured["policy"] == "default"
    finally:
        registry._REGISTRY.pop("_probe", None)


def test_tools_load_without_the_crawl_extra():
    """The serverless deploy omits trafilatura, and `load_tools` runs at its startup. Every tool
    module must import without it; only a fetch should fail, and say why.

    A subprocess, because blocking an import in this one would leave every tool module reloaded
    against a registry the rest of the suite shares.
    """
    import subprocess
    import sys

    script = (
        "import sys; sys.modules['trafilatura'] = None\n"
        "from anydocs.tools import page_text, registry\n"
        "registry.load_tools()\n"
        "try:\n"
        "    page_text.to_markdown('<p>hi</p>', 'https://example.com')\n"
        "except registry.ToolError as exc:\n"
        "    print(exc)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "crawl" in result.stdout
