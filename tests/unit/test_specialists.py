"""Specialist contracts: specs, prompts, and the README that routes them."""

from __future__ import annotations

import pytest

from anydocs import telemetry
from anydocs.orchestrator.planner import enabled_specialists, specialist_descriptions
from anydocs.prompts import load_prompt
from anydocs.specialists.runner import SPECIALISTS_DIR, load_spec
from anydocs.tools.registry import all_specs

ALL = ("librarian", "scout", "archivist", "analyst", "auditor")


@pytest.mark.parametrize("name", ALL)
def test_every_specialist_has_a_spec_and_a_prompt(name):
    spec = load_spec(name)
    assert spec.tools, f"{name} has an empty tool allowlist"
    assert spec.model_role
    prompt = load_prompt(name)
    assert prompt.text.strip()
    # The version is the span attribute a behavior change is tied to; it only has to be present
    # and namespaced, not frozen at 1.
    assert prompt.version.startswith(f"{name}@")
    assert prompt.version.split("@", 1)[1].strip()


@pytest.mark.parametrize("name", ALL)
def test_a_specialists_allowlist_names_only_registered_tools(name):
    known = set(all_specs())
    assert set(load_spec(name).tools) <= known


def test_telemetry_knows_every_specialist_folder():
    """A specialist whose name telemetry does not know cannot open a span, so this would fail
    at runtime rather than at import."""
    folders = {p.name for p in SPECIALISTS_DIR.iterdir() if (p / "spec.yaml").exists()}
    assert folders == set(telemetry.SPECIALISTS)


def test_the_archivist_is_the_only_specialist_that_can_write():
    """The whole permission model in one assertion.

    Stronger than it looks: it is also what stops the Librarian crawling and the Scout saving,
    because any non-read tool in their allowlists would put them in this set.
    """
    writers = {
        name
        for name in ALL
        if any(all_specs()[t].side_effect != "read" for t in load_spec(name).tools)
    }
    assert writers == {"archivist"}


def test_the_planner_sees_every_enabled_specialist():
    from anydocs.orchestrator.planner import ROUTABLE

    assert enabled_specialists() == list(ROUTABLE)
    descriptions = specialist_descriptions()
    for name in ALL:
        assert f"### {name}" in descriptions


def test_a_disabled_specialist_is_hidden_from_the_planner(tmp_path, monkeypatch):
    """`enabled: false` is how a specialist whose tool has no body stays unroutable, rather than
    being routed to and failing."""
    from dataclasses import replace

    from anydocs.orchestrator import planner
    from anydocs.specialists.runner import load_spec as real_load_spec

    def hide_analyst(name: str):
        spec = real_load_spec(name)
        return replace(spec, enabled=False) if name == "analyst" else spec

    monkeypatch.setattr(planner, "load_spec", hide_analyst)
    assert "analyst" not in planner.enabled_specialists()
    assert "### analyst" not in planner.specialist_descriptions()


def test_adding_a_specialist_means_adding_a_readme_paragraph():
    """The planner's prompt is assembled from these paragraphs, so a specialist with no
    paragraph is invisible to routing."""
    readme = (SPECIALISTS_DIR / "README.md").read_text()
    for name in ALL:
        assert f"## {name}" in readme


def test_an_unknown_result_schema_in_a_spec_fails_loudly(tmp_path, monkeypatch):
    folder = tmp_path / "ghost"
    folder.mkdir()
    (folder / "spec.yaml").write_text(
        "name: ghost\nmodel_role: reader\ntools: []\nresult_schema: NoSuchResult\n"
    )
    (folder / "prompt.md").write_text("version: 1\n\nx")
    monkeypatch.setattr("anydocs.specialists.runner.SPECIALISTS_DIR", tmp_path)
    from anydocs.specialists.runner import SpecError

    with pytest.raises(SpecError, match="not in"):
        load_spec("ghost")


def test_a_spec_allowing_an_unregistered_tool_fails_loudly(tmp_path, monkeypatch):
    folder = tmp_path / "ghost"
    folder.mkdir()
    (folder / "spec.yaml").write_text(
        "name: ghost\nmodel_role: reader\ntools: [teleport]\nresult_schema: ScoutResult\n"
    )
    monkeypatch.setattr("anydocs.specialists.runner.SPECIALISTS_DIR", tmp_path)
    from anydocs.specialists.runner import SpecError

    with pytest.raises(SpecError, match="unregistered tools"):
        load_spec("ghost")


def test_the_planner_drops_tool_names_that_do_not_exist():
    """`intent_tools` is compared against the tools actually called, so an invented name there
    makes a tool-selection eval meaningless."""
    from anydocs.orchestrator.planner import Plan, _sanitize

    plan = _sanitize(
        Plan(
            specialists=["librarian", "auditor"],
            intent_tools=["search_docs", "retrieval", "magic_wand"],
            rationale="because",
        ),
        [],
    )
    assert plan.intent_tools == ["search_docs"]
    # Dropped, not silently: a reader of the plan span should see what was discarded.
    assert "retrieval" in plan.rationale and "magic_wand" in plan.rationale


def test_the_planner_always_runs_the_librarian_first_and_the_auditor_last():
    from anydocs.orchestrator.planner import Plan, _sanitize

    plan = _sanitize(Plan(specialists=["archivist"]), [])
    assert plan.specialists[0] == "librarian"
    assert plan.specialists[-1] == "auditor"


def test_the_two_specialist_lists_agree():
    """`planner.ROUTABLE` owns the order, `telemetry.SPECIALISTS` owns membership. They cannot
    reference each other without inverting the layering, so this is what keeps them in step."""
    from anydocs.orchestrator.planner import ROUTABLE

    assert set(ROUTABLE) == set(telemetry.SPECIALISTS)
