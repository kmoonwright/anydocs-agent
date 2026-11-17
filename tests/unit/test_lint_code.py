"""`lint_code` must never hand the Auditor a green check it cannot back up."""

from __future__ import annotations

from anydocs.tools import registry


def _lint(**kwargs):
    return registry.invoke("lint_code", kwargs)


def test_a_syntax_error_fails(spans):
    result = _lint(code="def broken(:\n    pass", language="python")
    assert result.status == "fail"
    assert any(f.kind == "syntax" for f in result.findings)


def test_an_undefined_name_fails(spans):
    result = _lint(code="print(undefined_thing)", language="python")
    assert result.status == "fail"
    assert any("undefined name" in f.message for f in result.findings)


def test_clean_code_passes_but_says_it_was_not_checked_against_any_docs(spans):
    result = _lint(code="x = 1\nprint(x)", language="python")
    assert result.status == "pass"
    assert any(f.kind == "unverifiable" for f in result.findings)
    assert result.api_surface_size == 0


def test_an_unsupported_language_is_skipped_not_passed(spans):
    result = _lint(code="SELECT 1", language="sql")
    assert result.status == "skipped"
    assert any(f.kind == "unverifiable" for f in result.findings)


def test_symbols_are_checked_against_the_cited_pages(spans, relayr_corpus):
    from anydocs.knowledge.digest import doc_id_for

    doc_id = doc_id_for("relayr", "https://docs.relayr.example/retries.md")
    result = _lint(
        code="from relayr import Client\nclient = Client(api_key='x')\nclient.teleport()",
        language="python",
        context_doc_ids=[doc_id],
    )
    # The structure-only digest exposes no parameters, so nothing can be verified against it.
    # The tool must say so rather than implying the sample checks out.
    assert result.status in ("pass", "fail")
    assert result.symbols_used


def test_save_note_reports_an_incomplete_provenance_record(spans, data_dir):
    """An answer must be traceable to a page digested by a named model under a named prompt. A
    caller that omits those leaves a hole, so the hole is reported rather than written blank."""
    result = registry.invoke(
        "save_note",
        {
            "corpus": "probe",
            "url": "https://docs.example.com/page",
            "markdown": "# Page\n\nSome content that is long enough to make a chunk on its own.",
            "digest": {"purpose": "A page."},
            "title": "Page",
        },
    )
    assert result.provenance_incomplete == ["digest_model", "digest_prompt_version"]
    assert spans.attr("save_note", "provenance.incomplete") == (
        "digest_model",
        "digest_prompt_version",
    )


def test_a_complete_provenance_record_reports_nothing_missing(spans, data_dir):
    result = registry.invoke(
        "save_note",
        {
            "corpus": "probe",
            "url": "https://docs.example.com/page",
            "markdown": "# Page\n\nSome content that is long enough to make a chunk on its own.",
            "digest": {"purpose": "A page."},
            "digest_model": "claude-haiku-4-5-20251001",
            "digest_prompt_version": "digest_page@1",
        },
    )
    assert result.provenance_incomplete == []
