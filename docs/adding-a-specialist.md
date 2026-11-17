# Adding a specialist

A folder, a paragraph, and a scenario. Nothing in the orchestrator changes.

## 1. Create the folder

`anydocs/specialists/<name>/` with two files.

`spec.yaml`:

```yaml
name: teacher
description: Turns a corpus into a tutorial path for a stated goal.
model_role: reader          # router | summarizer | reader | judge | synthesizer
enabled: true               # false keeps it out of the planner's choices
tools:
  - search_docs
  - get_doc
budget:
  max_tool_calls: 8
  max_tokens: 24000
  deadline_ms: 30000
result_schema: TeacherResult
```

`prompt.md`, starting with a `version: N` line:

```markdown
version: 1

You are the Teacher. You ...
```

Write the prompt as a job description: what this specialist is for, how it should work step by
step, and — most importantly — how to set any score it returns. A confidence field with no
guidance on how to set it produces a number nobody can use.

## 2. Add the result schema

Specialists return Pydantic models, never prose. Add yours to
`anydocs/specialists/results.py` and to `RESULT_SCHEMAS`; `load_spec` fails loudly if
`result_schema` names something that is not there.

## 3. Add the name to telemetry

`telemetry.SPECIALISTS` is the list of names allowed to open a specialist span. A folder whose
name is missing from it raises at the first call rather than at import, so add it in the same
change.

## 4. Write the README paragraph

**This is the step that makes the specialist routable.** The orchestrator's planning prompt is
assembled from the paragraphs in `anydocs/specialists/README.md` — the planner knows nothing
else about any specialist. Write one paragraph covering what it is for, what it is good at, and
**when the orchestrator should choose it**, with that last part in bold so it survives being
read quickly.

Then it is routable. Do not edit `anydocs/orchestrator/prompt.md` to add a specialist; if you
find yourself wanting to, the paragraph is not doing its job.

## 5. Wire it into the flow, if it needs to be

`run_specialist("teacher", Task(...))` runs it. Whether a turn *calls* it is a decision in
`anydocs/orchestrator/graph.py` — the planner names specialists, but the graph owns the order and
the invariants (the Auditor sees every draft, one learning loop at most). A specialist that
answers a new class of question needs a node and a conditional edge there; one that refines an
existing step may not.

Add it the way `_analyst` was added: a node function taking `TurnState` and returning the keys it
changed, `builder.add_node("teacher", _teacher)`, and an entry in the routing function that
decides when to reach it. Keep the graph acyclic — "one learning loop per turn" holds because
the retry is its own node rather than an edge back, and a cycle would quietly undo that.

## 6. Test it

`tests/unit/test_specialists.py` already covers every specialist generically — the spec loads,
the allowlist names registered tools, the prompt exists, the README has a paragraph, and the
write permissions are what they claim to be. Adding a folder extends that coverage for free.

Add an integration test of your own for the behavior: script the model with the `scripted`
fixture, run the specialist, and assert both the structured result and the span tree.
