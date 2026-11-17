# Conventions

Read this before changing anything. The rules here are what keep the traces legible, which is
most of the point of this agent.

## Commands

```bash
uv sync --all-extras          # install, with the `crawl` and `phoenix` extras
uv run pytest                 # whole suite; needs no API key
uv run pytest tests/unit      # fast
uv run ruff check .           # lint; CI runs this
uv run phoenix serve          # local trace collector on :6006
uv run anydocs serve          # the HTTP API on :8000
uv run anydocs policy tools   # what is registered and what gates it
uv run anydocs bundle --corpus arize --corpus openinference  # corpus for the Vercel deploy
```

The test suite runs without any API key: embeddings are `FakeEmbeddings`, web search is
`FixtureProvider`, pages are replayed with respx, and models are a `ScriptedChatModel`. If a change
makes the suite need a key, the change is wrong.

## The one hard rule

**`anydocs/telemetry.py` is the only file that names a span.** Every span name, kind and
attribute key is a constant or a factory there. Tool and specialist code opens a context manager
from that module and never passes a string literal to a tracer.

`tests/unit/test_telemetry.py::test_no_other_module_names_a_span` enforces this mechanically.
If you need a new span, add a factory to `telemetry.py` — do not reach for `get_tracer()`.

A corollary: one node in the design doc's span tree is **one** span. If you wrap a named span
around an LLM call that would emit the same name, pass the span down instead —
`call_role(..., span=span)` exists for exactly that, and the planner, the reranker and
`digest_page` all use it.

LangChain's own spans sit *beside* that tree, not inside it: its callbacks do not put their spans
in the OpenTelemetry context, so nothing of ours can nest inside them. Measured on a real turn,
that makes them a second complete copy rather than extra depth — every specialist, tool and model
call twice, 60 spans where 25 carry the same information, named `model` and `tools` instead of
`librarian` and `search_docs`. They are off by default; `ANYDOCS_FRAMEWORK_SPANS=on` adds them.

## Writing a tool

A tool file contains exactly four things: a Pydantic `Input`, a Pydantic `Output`, a
`run(inp, ctx) -> Output`, and a `SPEC`. Registration derives the LLM tool schema, the MCP tool
definition, the permission check and the span wrapper from that one declaration — so the `SPEC`
is the part of a tool that must exist even before its body does, and the policy-key rule applies
to a stub like any other.

- Never write the policy check yourself. Declare `policy_key` and let `registry.invoke` enforce
  it. A non-read tool that declares no policy key fails at import.
- Never open a span. `ctx.span` is handed to you for domain attributes; the span itself was
  opened by the registry.
- Raise `ToolError` for something the model should see and may route around. Let `PolicyDenied`
  and `NeedsApproval` escape — they belong to the turn.
- A path or URL in an `Input` arrived from a model. Treat it as untrusted: `_raw_pages.py`
  confines paths to the data directory, and `policy.check_domain` confines URLs.

See `docs/adding-a-tool.md`.

## Writing a specialist

A specialist is LangChain's `create_agent` with our spec around it: `runner.py` opens the span,
hands it only the tools `spec.yaml` allows, caps its tool calls, and binds the result schema as
the tool it answers with. A specialist is a folder with `spec.yaml` and `prompt.md`, plus a
paragraph in `anydocs/specialists/README.md`. **The orchestrator's planning prompt is assembled from those
paragraphs**, so adding a specialist must not mean editing the orchestrator prompt.

- The tool allowlist lives in `spec.yaml`, not in the prompt. The runner enforces it; a call to
  a tool outside the list is a hard error on the span, not a retry.
- Budgets are enforced, not requested. The runner passes the cap down and
  `tools/langchain_tools.py` counts the calls, refusing the one that would exceed it. Exceeding a
  budget returns a partial result tagged `truncated`, which the Auditor reads as a confidence
  penalty — as does a result the model could not finish writing.
- Add the name to `telemetry.SPECIALISTS` or the specialist cannot open a span.
- Results are Pydantic models in `anydocs/specialists/results.py`. Free text appears only in the
  final synthesis.

See `docs/adding-a-specialist.md`.

## Prompts

Repo files are the source of truth. Each starts with a `version: N` line, which becomes
`prompt.version` on the span — so a behavior change can be tied to a prompt change. Bump the
version when you change a prompt's meaning, not when you fix a typo.

The harness pushes these files to the prompt hub under a tag; runtime may later pull by tag with
the file as fallback. Nothing in this process invents prompt text.

## Models

`config/models.yaml` maps a **role** to a provider and model; `models/chat.py` turns a role into
a LangChain chat model. The role name is what appears on spans, so experiments compare roles
rather than model strings. Never hard-code a model id in Python. Reach for a role:

- `router` — cheap and fast: planning, source judging, reranking.
- `summarizer` — cheap and high volume: digests, rolling session summaries.
- `reader` — mid-tier: query rewriting, drafting answers.
- `judge` — the Auditor's per-claim verification.
- `synthesizer` — mid-tier: the final answer.

Two provider quirks worth knowing before you edit `models.yaml`:

- The Anthropic Messages API prefers `effort` over `temperature`, and **only Claude 5 models
  accept it** — Claude 4.x answers a request carrying it with a 400, on *every* call rather than
  just the first. `resolve_role` therefore strips `effort` pre-emptively from any model whose id
  is not Claude 5 and records `model.degraded` on the span. This matters because `--judge-model`
  can point a Claude 5 role at a 4.x model, and without the strip every turn would fail.
- The Anthropic SDK is on `httpx2`; `anydocs/tools/http_client.py` is on `httpx`. Patching httpx
  does not reach either provider SDK, so a test that patches httpx globally silently makes real
  network calls instead of failing. Mock the SDK's own transport, or script the model.

## Streaming

`/invoke/stream` and `/invoke` run the same turn. The difference is an `on_event` callback on
`TurnRequest` — not a second code path, which would drift. The turn runs on a worker thread
because it is synchronous throughout (the tool layer, the store and the provider SDKs all
block), and a broken event consumer cannot fail it: a client that disconnected mid-stream must
not take the answer down with it.

## Budgets

`config/policy.default.yaml` bounds **a turn**, because a user is waiting: 24 tool calls, 250k
tokens, 240 seconds. Two things deliberately do not share that:

- **Batch ingest** (`ingest_folder`, `crawl_and_ingest`) gives each page its own ledger. A
  200-page crawl legitimately exceeds every per-turn cap, and the daily USD cap is the
  job-level guard.
- **Headless learning** (`/learn`, `anydocs learn`) raises the wall clock and token caps,
  because nobody is waiting on an interactive deadline.

Both numbers are measured, not inherited from the design doc's illustrative 45s — a crawling turn
took 105s and a learning turn more. The Auditor is the single largest share of an answering turn
(10–35s depending on the judge model), so lower the wall clock, or move `judge` cheaper, if
latency matters more than the verification.

## Style

- Say what failed. A blocked crawl, a timed-out tool or two sources disagreeing belong in the
  answer and on the span, not in a swallowed exception.
- Every factual sentence in an answer carries a citation. One that cannot be attributed is
  marked as inference.
- Confidence is a number with reasons attached. A score with no reasons is not usable by
  anything downstream.
- Comment the *why*, not the *what*. The code already says what it does.

## The flow

`orchestrator/graph.py` is workflow 1 as a LangGraph state graph: librarian, then scout,
archivist and a second librarian pass when confidence is low, then verify and synthesize.
`turn.py` is the envelope around it — ledger, session, turn span, guardrail downgrade, suspend
and resume.

The retry is its own `relibrarian` node rather than a cycle, so **one learning loop per turn** is
a property of the graph instead of a counter someone has to maintain. Add a node; never add an
edge back.

A turn paused for approval re-enters the same graph at the `archivist` node with the approval
granted, which is why `/invoke/approve` produces an answer built by exactly the nodes that would
have built it had nobody been asked.

The Librarian is not always first: `_entry` routes a numeric plan to the Analyst, and a numeric
answer ends the turn there without an audit.

## Retrieval

`knowledge/retriever.py` is a LangChain `BaseRetriever`, so AnyDocs retrieval drops into any
chain, but `search_docs` calls `.search()` rather than `.invoke()` because it needs the per-stage
scores. Do not replace the hybrid leg with `EnsembleRetriever` + `BM25Retriever`: it fuses
internally and returns documents with no stage scores, which costs the `dense`/`sparse_bm25`/
`rrf_fusion` logs, the digest down-weight, the session-pin boost and link expansion — the eval
hooks. The reasoning is written out in that module's docstring.

## Contradictions

Conflicts between pages are found **at ingest**, by a bounded pass over the docs a run just
wrote (`links.detect_across`), not inside `save_note` and not at answer time. Two reasons: a
500-page crawl would otherwise make thousands of nested comparison calls, and those calls would
come out of the Archivist's own per-turn budget and truncate it.

The detector is a cheap model and it is wrong in predictable ways, so `links._implausible`
rejects a claimed conflict its own quotes do not support. Every check there corresponds to a
false positive observed on real documentation — different subjects, a version difference, two
quotes stating the same number. **A false conflict is worse than a missed one**: it makes the
agent hedge an answer that was right. When you touch the prompt or the schema, re-run the live
check in `docs/design.md` order and confirm the corpus with no real conflicts still records
none.

## Three doors, one implementation

HTTP (`anydocs/server.py`), MCP (`anydocs/mcp_server.py`) and the CLI all call the same tool
layer, so a trace from any of them lines up span for span. Two things keep it that way:

- The MCP tools are **generated** from the same `ToolSpec` declarations — `_tool_function`
  synthesizes a function signature from the Pydantic `Input` model, because the SDK derives its
  schema from the signature. Hand-writing those would put the same declaration in two places and
  the second one would drift.
- `create_app()` is a factory, not a module-level app, because an MCP session manager can be run
  exactly once per instance. Tests build their own app; `app` at the bottom of the module is the
  one uvicorn and the Dockerfile serve.

Write tools are deliberately not exposed over MCP. A client gets `learn`, which carries the same
approval semantics — `needs_approval` comes back as a tool *result* with the resume token,
because MCP has no standard approval flow and an error is not something a client can act on. For
the same reason, `ToolError` is returned rather than raised: the SDK renders a raised exception
as "Error executing tool <name>" with the detail stripped, and a model cannot route around a
failure it cannot read.

## Serverless

The Vercel deploy (`docs/deploy.md`) is the same app with `ANYDOCS_FLUSH_SPANS=true`, not a
second code path. That one switch makes `server._prepare` run at `create_app()` (a host may skip
the lifespan), copy `bundled/` onto the writable data directory, and refuse to start on a default
credential. Three things keep it read-only and traceable:

- `ANYDOCS_CRAWL_MODE`/`WRITE_MODE`/`DELETE_MODE` are a **ceiling**, applied by `policy._clamp`
  after every merge. A request's `overrides.policy` can tighten a mode, never raise it.
- `telemetry.remote_parent` makes an incoming `traceparent` the turn span's parent, so a turn
  started by an AX experiment lands under its run span.
- Spans are always batched; the per-request `flush_spans()` is what makes a frozen process safe.
  Phoenix and AX share the stock OTLP/HTTP exporter. Do not add a vendor wrapper or the
  `opentelemetry-exporter-otlp` meta-package: either brings grpcio back.
- `GET /cron/turn` is Vercel Cron's door: `CRON_SECRET`, not the API token, and the same
  `run_turn` with `user_id=scheduler`. The questions are `config/demo-questions.yaml`; change
  what the demo asks there, not in code.
- The function ships without `numpy` or `trafilatura` (about 260 MB of dependencies down to 129
  MB). BM25 is `retrieval._bm25`, a copy of `rank_bm25.BM25Okapi` that
  `tests/unit/test_bm25_parity.py` pins to the original, so the logged stage scores do not move.
  Page cleaning is the `crawl` extra, imported only through `page_text.trafilatura()`; the
  container installs it, and a tool module must never import it at the top level.

## Memory

Three kinds, deliberately separate:

- **Session** (`memory/session.py`) — in-process, 24-hour TTL. Cited doc ids are boosted in
  retrieval through a contextvar, not a tool argument: the model should not have to pass its own
  history into every search. Past a token budget, `roll_up` replaces the raw turns with a
  summary, which lands on the span so a session eval can see what the agent thought the
  conversation was about.
- **Profile** (`memory/profile.py`) — opt-in, on disk, keyed by `user_id`. The line that matters
  is between **declared** and **inferred**: a declared preference changes behaviour, an inferred
  one is a suggestion carrying its evidence and only becomes a preference when the user confirms
  it. `as_prompt_block()` returns declared preferences only — injecting a suggestion would be
  acting on it, which is the thing this split exists to prevent.
- **Operator notes** (`config/operator-notes.md`) — deployment-level, versioned like a prompt.

## Provider output limits are a correctness problem here, not a cost one

Three separate failures traced back to `max_tokens: 4096`, and none of them looked like
truncation. Anthropic streams tool input as partial JSON, so a response cut off mid-value arrives
as a **well-formed** tool call whose unclosed field is a raw string — what surfaces is a Pydantic
"Input should be a valid list", or `StructuredOutputValidationError: Unterminated string`, both
of which read like a schema or prompt bug and are neither.

Every role returning a structured result with a list in it — facts, citations, per-claim verdicts
— is therefore at 16384. `roles._truncation_note` names the cause when it happens anyway, and
`runner.py` catches `StructuredOutputError` so a specialist that cannot report degrades to a
`truncated` partial result instead of ending the turn.

A related one: a model asked for a long nested array sometimes emits it as a JSON *string*, with
the markdown inside escaped as though it were JSON markup (`` \` ``). `roles._repair` decodes
that rather than spending another model call on a retry. Its unescaper matches escape sequences
left to right as whole units — a pattern that only looks for invalid escapes can start matching
*inside* a valid `\\` pair and corrupt it, which is a bug this code had for one round.

## Spans have an attribute limit

OpenTelemetry caps attributes per span at 128 by default, and a turn span passes it: an answer's
citations and an audit's per-claim verdicts are each several attributes. Past the cap the SDK
drops the **oldest** attribute per new one, with only a log line — so the first thing lost is the
identity block written first (`session.id`, `turn.id`, `corpus.active`), which is exactly what
every downstream filter and eval is keyed on. `telemetry._span_limits` raises it to 512. A trace
that looks complete and is missing those keys is this bug.

## What is not built yet

Everything in the design doc's build order is built. What remains is explicitly deferred there:
`compare_versions`, `run_example`, `ask_user` and `open_issue` from "tools worth adding later",
the `Teacher` and `Migrator` specialists, `watch` mode, and image understanding. A Redis session
backend is one class behind the existing `SessionStore` interface; compose already declares the
service.
