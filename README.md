# AnyDocs Agent

A documentation agent that answers questions about any software product's docs — and when it
does not know, goes and learns: it searches the web, crawls the pages it finds, digests them
into a durable knowledge store with provenance, and answers from that store next time.

Built to be instrumented end to end. Every answer cites a corpus, a snapshot, a URL and a
heading; every answer carries a confidence score with the reasons behind it; and every trace
reads as a short story — asked, looked, found, checked, answered.

## How it works

A single orchestrator owns the turn and delegates to specialists, each with its own prompt,
model role and tool allowlist. Specialists never call each other, which keeps the trace a
shallow tree rather than a thicket.

| Specialist | Job | Model role |
| --- | --- | --- |
| **Librarian** | Answers from what is already known: rewrites the question into retrieval queries, searches the active corpora, drafts an answer with inline citations, and reports how well the store covered the question. | `reader` |
| **Scout** | Finds material the store lacks: searches the web, judges which results are a vendor's own docs rather than blogs and forums, and probes candidates for a doc root. | `router` |
| **Archivist** | Turns pages into durable knowledge: crawls an approved source, digests each page, writes it with provenance, and reports what was added, skipped and blocked. | `summarizer` |
| **Auditor** | Checks before shipping: verifies each cited claim against the cited chunk, lints code samples against the cited API surface, and scores confidence. Runs on every turn that produces a draft. | `judge` |
| **Analyst** | Answers numeric questions about the docs themselves — page views, searches that find nothing, time to answer — by choosing parameters for a query the code then writes. Never free-form SQL from a model. | `reader` |

Fourteen tools, split read and write, registered once and shared by every path into the agent.
Write tools are gated by a policy enforced in code — `off`, `ask` or `auto` per side-effect
class — so a crawl can require a human decision, and that approval moment shows up in the trace.

## Quickstart

```bash
uv sync --all-extras         # includes the `crawl` and `phoenix` extras
cp .env.example .env         # add your keys
```

Two extras are optional at runtime. `crawl` is the page cleaner (`trafilatura`) that fetching,
crawling and diffing need; `phoenix` is the local trace collector. The Vercel function installs
neither, and the container installs `crawl`.

Ingest the fixture corpus and ask it something — no API key needed for this path, since
`--no-digest` fills digests from page structure and the index uses local embeddings:

```bash
ANYDOCS_EMBEDDINGS=fake uv run anydocs corpus ingest tests/fixtures/corpus \
  --name relayr --no-digest --base-url https://docs.relayr.example
uv run anydocs corpus list
```

With `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` set, ask a real question:

```bash
uv run phoenix serve          # terminal 1: the local trace collector, on :6006
uv run anydocs ask "what is the maximum max_attempts?" --corpus relayr
```

The trace shows up at <http://localhost:6006>.

### Watching it work

`/invoke/stream` emits the plan, each specialist result as it lands, and the final answer as
Server-Sent Events:

```bash
curl -N -XPOST localhost:8000/invoke/stream \
  -H "Authorization: Bearer $ANYDOCS_API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"message": "what is the maximum max_attempts?", "corpora": ["relayr"]}'
```

### Over MCP

The MCP server is mounted at `/mcp` on the same process, so its tools are the same tools:

```python
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

async with streamable_http_client("http://localhost:8000/mcp/") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("ask_anydocs", {"question": "how do retries work?"})
```

It exposes the read tools plus `ask_anydocs`, `learn` and `approve`, the `anydocs://corpora` and
`anydocs://docs/{id}` resources, and the `answer-from-docs` and `compare-versions` prompts. Write
tools are deliberately absent: a client gets `learn`, which returns `needs_approval` with a
resume token when the policy requires a human decision.

### Numbers about the docs themselves

```bash
uv run python analytics/seed.py --out data/analytics.db
uv run anydocs ask "which section has the most search misses?" --corpus relayr
```

The seeded data is synthetic and describes the `relayr` fixture corpus only.

### Over HTTP

```bash
uv run anydocs serve

curl -XPOST localhost:8000/invoke \
  -H "Authorization: Bearer $ANYDOCS_API_TOKEN" -H 'Content-Type: application/json' \
  -d '{"session_id": "s1", "message": "how long are receipts kept?", "corpora": ["relayr"]}'
```

When the policy requires approval, `/invoke` returns `pending_approval` with a resume token and
a cost estimate instead of an answer. Continue the turn with `/invoke/approve` — the turn picks
up where it paused rather than starting over.

### Learning something new

```bash
uv run anydocs learn https://docs.pydantic.dev/latest/concepts/validators/ --max-pages 10
```

### Keeping it fresh

Each page carries a cadence — 2 days for changelogs, 14 for reference, 30 for guides. A refresh
pass re-checks what is due, re-digests what moved, and records the diff as a changelog entry:

```bash
uv run anydocs refresh --corpus pydantic
uv run anydocs corpus schedule pydantic --show
uv run anydocs corpus links pydantic        # including which sources disagree
```

This runs headless under `anydocs.maintenance` with `user.id=system`, so it shows on dashboards
and can be filtered out of user-facing eval scores.

## Configuration

| File | What it controls |
| --- | --- |
| `config/models.yaml` | Role → provider and model. The **role** is what appears on spans, so experiments compare roles rather than model strings. |
| `config/policy.default.yaml` | Permission modes per side-effect class, crawl caps, rate limits, and the per-turn and per-day budgets. |
| `config/operator-notes.md` | Standing instructions for the deployment: house style, preferred corpora, domains the Scout must never crawl. Versioned like a prompt. |
| `.env` | Keys, the bearer token, the resume-token signing key, and where traces go. |

Set `TRACE_TARGET=phoenix` for a local collector or `ax` for Arize AX. That environment variable
is the only difference between the two; both use the same OTLP/HTTP exporter and no code path
changes.

A few environment variables matter only to a deployment:

| Variable | Effect |
| --- | --- |
| `ANYDOCS_FLUSH_SPANS=true` | Serverless mode: flush spans after every response, run MCP stateless, copy `bundled/` onto the data directory at startup, and refuse to start on a default token or signing key. |
| `ANYDOCS_CRAWL_MODE`, `ANYDOCS_WRITE_MODE`, `ANYDOCS_DELETE_MODE` | A ceiling below the policy file. A request's `overrides.policy` can tighten these but never raise them. |
| `CRON_SECRET` | Turns on `GET /cron/turn`: one question from `config/demo-questions.yaml`, run as `user.id=scheduler`. Vercel Cron calls it hourly so a demo project keeps receiving traces. |
| `ANYDOCS_DATA_DIR`, `ANYDOCS_BUNDLE_DIR` | Where the writable store lives, and where a deploy's shipped corpus is read from. |

An incoming `traceparent` header becomes the parent of the turn span, so a turn started by an AX
experiment lands under that experiment's run.

## Deploying

Two hosts, one implementation:

- **The container** (`Dockerfile`, `docker-compose.yml`) is the full agent: learning, crawling,
  approval and the stateful MCP session. Its knowledge store is a volume at `/data`.
- **Vercel** hosts a read-only `/invoke` for the AX agent playground. It answers from the corpus
  in `bundled/` (built with `uv run anydocs bundle --corpus arize --corpus openinference`), with
  crawl, write and delete locked off, and asks itself one demo question an hour. See
  [docs/deploy.md](docs/deploy.md).

## Tests

```bash
uv run pytest           # whole suite
uv run ruff check .
```

The suite needs no API key: embeddings are deterministic hashes, web search replays recorded
fixtures, pages are served by `respx`, and models are a scripted client. The integration tests
assert on real span trees — names, kinds, parent/child shape and attributes — so renaming a span
or dropping an attribute fails a test.

## Where the build is

All nine build-order steps of `docs/design.md` are implemented: telemetry and the HTTP API, the
knowledge store and the Librarian, the Auditor, the Archivist, the policy layer with the Scout
and the learn-then-answer loop, the freshness workflow with links and "sources disagree" answers,
the MCP server and CLI parity, the Analyst with session memory and an opt-in profile, and SSE
streaming with guardrails and cost control.

What is not built is what the doc defers — see "What is not built yet" in
[CLAUDE.md](CLAUDE.md), which is the one list kept current.

## Reading further

- `docs/design.md` — the design doc this was built from.
- `docs/deploy.md` — the Vercel deploy: bundle, env, and first-deploy checks.
- `CLAUDE.md` — conventions, including the rule that `telemetry.py` is the only file that names
  a span.
- `docs/adding-a-tool.md`, `docs/adding-a-specialist.md` — how to extend it.
