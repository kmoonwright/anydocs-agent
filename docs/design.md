# AnyDocs Agent — Design Doc

Nov 2, 2026

## Purpose and scope

AnyDocs is a documentation agent that answers questions about any software product's docs, and when it does not know, goes and learns: it searches the web, crawls the pages it finds, digests them into a durable knowledge store with provenance, and answers from that store next time. It is built to be instrumented end to end, so every capability below exists partly because it produces a trace shape worth observing.

The agent lives in its own repository and is consumed over HTTP and MCP. Hosting, traffic simulation, AX lifecycle seeding and CI belong to the companion harness repo and are out of scope here except where the agent must expose something for them.

**In scope**

- Multi-step question answering over one or many documentation corpora, with citations that resolve to real URLs and versions.
- Discovering and ingesting new documentation on request or when retrieval confidence is low.
- Keeping ingested docs fresh: detecting that a page changed, re-digesting it, and recording what changed.
- Structured answers: code samples validated against the docs they came from, version-aware comparisons, numeric answers from an analytics tool.
- Self-checking its own answers before returning them, and flagging low-confidence ones for review.

**Out of scope for v1**

- Editing or publishing the user's documentation.
- Authenticated crawling behind logins.
- Running untrusted code from the docs; code is parsed and linted, not executed.

## Design principles

1. **Legible first.** A prospect should follow any trace without a walkthrough. Sub-agents have job titles, tools have verbs, and a trace reads as a short story: asked, looked, found, checked, answered.
2. **Learn, don't just retrieve.** When retrieval confidence is low the agent is allowed to go get new material, but what it learns is written down once, with provenance, and reused. No answer depends on a live fetch that cannot be reproduced.
3. **Every claim has a source.** Answers cite a corpus, a snapshot id, a URL and a heading. A sentence the agent cannot attribute is marked as inference, not fact.
4. **Side effects are explicit.** Tools that write (crawl, digest, save, forget) are separate from tools that read, are tagged as such in telemetry, and are gated by a policy the deployment controls.
5. **One implementation, two doors.** The HTTP agent and the MCP server share the same tool code, so traces from both paths line up span for span.
6. **Small models for small jobs.** Routing, query rewriting, chunk scoring and summarization run on cheap models; only synthesis and self-check use the frontier model. Model choice is a per-role config, never hard-coded.
7. **Confidence is a first-class output.** Every answer carries a confidence score and the reasons behind it, so the harness can route low-confidence answers to a labeling queue without re-judging them.
8. **Fail visibly.** When a crawl is blocked, a tool times out or sources disagree, the agent says so in the answer and records it on the span rather than papering over it.

## Architecture

A single orchestrator owns the turn and delegates to five specialists, each a small agent with its own prompt, model and tool allowlist. Specialists never call each other; they report back to the orchestrator, which keeps the trace a shallow tree rather than a thicket. All specialists share one tool layer, and all tools read from or write to one knowledge store.

&#91;embedded content: agent architecture: orchestrator, 5 specialists, tools, store\]

The Auditor is the only specialist that runs on every turn: it receives the draft answer and its citations, checks them against the retrieved material, and attaches a confidence score and reasons. The orchestrator can accept, revise once, or answer with an explicit caveat.

## Specialist sub-agents

Each specialist is a small ReAct-style agent with a fixed tool allowlist, its own prompt in the prompt hub, and a model role the deployment maps to a concrete model. The orchestrator hands each one a structured task and gets a structured result back; free text only appears in the final synthesis.

| Specialist | Job | Tools allowed | Model role | Returns |
| --- | --- | --- | --- | --- |
| Librarian | Answer from what is already known. Rewrites the question into 1–3 retrieval queries, runs hybrid search across the active corpora, reranks, and drafts an answer with inline citations. | `search_docs`, `get_doc`, `list_corpora` | `reader` (mid-tier) | draft, citations, retrieval confidence, unanswered sub-questions |
| Scout | Find material the store lacks. Searches the web with site and version hints, judges which results are authoritative docs vs. blogs and forums, and probes candidate sites for sitemaps and doc roots. | `web_search`, `fetch_page` | `router` (cheap) | ranked source candidates with a reason and an estimated page count |
| Archivist | Turn pages into durable knowledge. Crawls an approved source, digests each page into a structured summary (purpose, key facts, parameters, examples, versions), writes it with provenance, links related digests, and schedules re-checks. | `crawl_site`, `digest_page`, `save_note`, `link_sources`, `diff_page`, `schedule_refresh`, `forget` | `summarizer` (cheap) | snapshot id, pages added/changed/unchanged, digest quality score |
| Analyst | Answer numeric questions. Translates a question into a parameterized query against the analytics store, formats results in human units, and states assumptions. | `query_metrics` | `reader` | table, formatted answer, units, query used |
| Auditor | Check before shipping. Verifies each cited claim against the cited chunk, checks code samples with `lint_code`, flags version mismatches and contradictions between sources, and scores confidence. | `get_doc`, `lint_code` | `judge` (frontier) | pass/revise, per-claim verdicts, confidence 0–1, reasons |

**Why five and not one**

A single tool-using loop can do all of this, but it produces one long span with twenty tool calls that nobody can read on a call. Specialists give the trace natural chapters, let each role run on the cheapest model that works, and make tool permissions enforceable: the Librarian cannot crawl, and the Scout cannot save.

**Delegation protocol**

The orchestrator sends `{task, inputs, budget: {max_tool_calls, max_tokens, deadline_ms}, corpora}` and receives `{result, confidence, cost, notes}`. Budgets are enforced by the specialist runner, not trusted to the prompt; exceeding one returns a partial result tagged `truncated`, which the Auditor treats as a confidence penalty.

## Tool catalog

Fourteen tools, split into read and write. Every tool is a plain Python function with a Pydantic schema, registered once and exposed identically to specialists and to the MCP server. Write tools are gated by the permission policy in a later section.

**Read tools**

| Tool | Inputs | What it does | Span kind |
| --- | --- | --- | --- |
| `search_docs` | query, corpora\[\], k, filters (version, section) | Hybrid dense + BM25 search over the active snapshots, reranked; returns chunks with doc id, heading path, URL, score | RETRIEVER |
| `get_doc` | doc id or URL, snapshot id? | Full digested page plus raw chunks, for the Auditor to verify against | TOOL |
| `list_corpora` | — | Corpora, their default snapshot, page counts, last refresh | TOOL |
| `web_search` | query, site?, recency? | Calls a search API (Brave, Tavily or Exa behind one interface); returns URL, title, snippet, domain authority hint | TOOL |
| `fetch_page` | URL | Fetches one page, strips chrome to markdown, extracts canonical URL, title, last-modified, version markers; honors robots.txt | TOOL |
| `diff_page` | URL or doc id | Re-fetches a page, compares with the stored digest's content hash and returns a structured diff (added, removed, changed sections) | TOOL |
| `query_metrics` | metric, dimensions\[\], window, filters | Parameterized SQL over the analytics SQLite; never free-form SQL from the model | TOOL |
| `lint_code` | code, language, context docs | Parses, type-checks or lints a code sample and checks imports and symbols against the cited docs' known API surface | TOOL |

**Write tools**

| Tool | Inputs | What it does | Guard |
| --- | --- | --- | --- |
| `crawl_site` | root URL, include/exclude globs, max pages, depth | Polite BFS crawl with per-domain rate limit; writes raw pages to a new snapshot folder | policy `crawl` + page cap |
| `digest_page` | raw page | Produces a structured digest: purpose, key facts, parameters, code examples, version notes, 3–5 Q&A pairs the page answers | none (pure function over fetched content) |
| `save_note` | digest, provenance, tags | Embeds and indexes the digest into the corpus snapshot and appends a provenance record | policy `write` |
| `link_sources` | doc ids\[\] | Records related-doc edges (supersedes, see-also, contradicts) used to expand retrieval and to flag conflicts | policy `write` |
| `schedule_refresh` | doc id or corpus, cadence | Adds or updates an entry in the freshness schedule the Archivist works through | policy `write` |
| `forget` | doc id, reason | Tombstones a digest, keeps the provenance record, removes it from retrieval | policy `delete`, always logged |

**Tools worth adding later**

- `compare_versions`: answer "what changed between v2 and v3" from stored snapshots of the same doc.
- `run_example`: execute a code sample in a sandbox against a pinned SDK version; promotes `lint_code` from static to real validation.
- `ask_user`: a clarifying-question tool so ambiguous questions become a span rather than a guess.
- `open_issue`: file a docs bug upstream when two authoritative pages contradict each other.

## Core workflows

Four workflows cover the agent's behavior. The first is the hot path; the other three are what make it more than a retrieval bot.

**1. Answer, or learn then answer**

&#91;embedded content: answer-or-learn flow: 1 decision, 1 loop\]

The Librarian returns a retrieval confidence (reranker score spread, coverage of the question's sub-parts, citation density). Below a threshold the orchestrator asks the Scout for sources, the policy approves or declines the crawl, the Archivist ingests, and the Librarian retries against the new snapshot. One learning loop per turn at most; a second miss produces an answer that says what was not found.

**2. Learn on request**

A user or operator says "learn the Stripe Connect docs." The Scout resolves that to a doc root and sitemap, estimates page count, and the orchestrator asks for confirmation when the estimate exceeds the policy's auto-approve cap. The Archivist crawls, digests, saves and reports: pages added, duplicates skipped, pages that failed to parse, and the three questions the new corpus is best at answering, generated from the digests' Q&A pairs. The Librarian is warmed up by running those questions once.

**3. Keep docs fresh**

The Archivist works a freshness schedule: each digest has a cadence (default 14 days, shorter for pages tagged release notes or changelog). It calls `diff_page`; unchanged pages get a longer cadence, changed pages are re-digested and the diff is stored as a changelog entry on the doc. Contradictions between a new digest and a linked one are recorded with `link_sources(kind=contradicts)` and surface in answers as "sources disagree." This workflow runs headless on a timer and produces traces without any user.

**4. Self-check**

The Auditor receives the draft, the citations and the retrieved chunks. For each claim it finds the supporting span of text or marks it unsupported; for each code sample it runs `lint_code` against the cited API surface; it checks that version markers in the answer match the versions in the sources. It returns pass, or revise with specific edits, and a confidence score with reasons. The orchestrator applies one round of revisions at most, then answers with the Auditor's caveats inline when confidence stays low.

## Knowledge store

The store holds five kinds of things, all keyed so that any answer can be traced back to a specific page at a specific time. It is a single directory per corpus snapshot (a LanceDB or SQLite + sqlite-vec index plus JSONL sidecars), so a snapshot can be copied to object storage or handed to a colleague as one folder.

| Object | Shape | Written by | Read by |
| --- | --- | --- | --- |
| Corpus | name, source config, list of snapshots, default snapshot | operator via CLI, Archivist on first crawl | everyone |
| Snapshot | id (date + content hash), index files, manifest (page count, embedding model, crawl stats) | Archivist | Librarian, Auditor |
| Digest | doc id, canonical URL, title, heading path, version markers, structured summary, Q&A pairs, chunks with embeddings, content hash | Archivist via `digest_page` + `save_note` | Librarian, Auditor |
| Provenance record | doc id, fetched at, HTTP metadata, robots decision, digest model and prompt version, who or what requested it, tombstone if forgotten | `save_note`, `forget` | Auditor, operators, the answer's citation footer |
| Link | from doc, to doc, kind (supersedes, see-also, contradicts, example-of), evidence | `link_sources` | Librarian (query expansion), Auditor (conflict check) |

**Digest format**

A digest is the unit of knowledge, not the raw page. It is a fixed schema the summarizer fills: `purpose` (one sentence), `facts` (bullet claims each tied to a source span), `parameters` (name, type, default, constraint), `examples` (code with language and the API symbols it uses), `versions` (applies-to, deprecated-in), `answers` (3–5 natural questions this page answers, used for retrieval and for warm-up). Chunks are made from the raw page for citation fidelity; digests are indexed alongside them with a `kind` field so retrieval can weight them differently.

**Retrieval**

Hybrid search runs dense and BM25 in parallel over chunks and digests, fuses with reciprocal rank fusion, filters by corpus and version, then reranks the top 30 with a cross-encoder or a cheap LLM pass. Links expand the result set by one hop for `see-also` and `supersedes`. The retriever span records every stage's candidates and scores so retrieval evals can see where a good chunk was lost.

**Freshness schedule**

A small table: doc id, next check, cadence, last result. Cadence doubles on unchanged, resets on changed, and is capped by corpus policy. Release-note pages default to 2 days, API reference 14, conceptual guides 30.

**Analytics store**

A seeded SQLite database of synthetic usage metrics about the docs themselves (page views, search misses, time to answer by section). It exists so the Analyst has deterministic numbers to answer with and so numeric evals have exact expected values.

## Sessions, memory and personalization

The agent remembers within a session by default and across sessions only when told to, because persistent memory is a feature prospects ask about and a trace shape worth showing, but it must never make a demo answer unexplainable.

**Session memory**

- Keyed by `session_id`; stores turns, the corpora in play, the versions the user has mentioned, and the doc ids already cited. Lives in memory or Redis; expires after 24 hours of inactivity.
- A rolling summary replaces raw turns past a token budget. The summary is written by the summarizer role and stored as a span attribute so session evals can see what the agent thought the conversation was about.
- Cited doc ids are pinned into retrieval for the rest of the session, which makes follow-ups like "and how do I configure that?" land on the right page.

**User profile (opt-in)**

- Keyed by `user_id`; holds declared preferences: primary language or SDK, product versions in use, verbosity, whether learning new sources without asking is allowed.
- Inferred preferences ("this user always asks about Python") are written as suggestions with evidence and only promoted to the profile when the user confirms, so the agent never silently changes behavior.
- The profile is injected into the orchestrator prompt as a short block and recorded on the trace as `user.profile.version`, so a behavior change can be tied to a profile change.

**Operator memory**

A deployment-level notes file (editable in the repo) for standing instructions: house style for answers, corpora to prefer, domains the Scout must never crawl. It is versioned like a prompt and loaded at startup.

**What is not remembered**

Raw pages the Scout fetched but the policy declined to ingest, the content of declined crawls, and anything a `forget` call tombstoned. Provenance keeps the record that something was fetched and discarded, but not the content.

## Permissions, guardrails and cost control

An agent that can search, crawl and save needs a policy layer that is enforced in code, not in the prompt. The policy is a YAML file loaded at startup and overridable per request by a signed header, so the harness can flip modes for a demo.

**Permission policy**

```yaml
policy: default
crawl:
  mode: ask            # off | ask | auto
  auto_approve_pages: 50
  max_pages_per_crawl: 500
  allow_domains: []    # empty = any public domain
  deny_domains: ["*.internal", "localhost", "127.0.0.1"]
  respect_robots: true
  per_domain_rps: 1
write:
  mode: auto           # save_note, link_sources, schedule_refresh
delete:
  mode: ask            # forget
web_search:
  mode: auto
  max_calls_per_turn: 3
budget:
  max_tool_calls_per_turn: 24
  max_tokens_per_turn: 60000
  max_wall_ms: 45000
  daily_usd_cap: 25
```

`ask` mode returns a structured `needs_approval` result from `/invoke` with a resume token; the client (or the harness) calls `/invoke/approve` to continue. The pending action and the decision are both spans, so the approval moment is visible in the trace.

**Content guardrails**

- Fetched pages are untrusted: instructions found in a page are never followed, and the digest prompt is told so. The Auditor flags any answer whose text mirrors imperative content from a source.
- Prompt-injection check on the user turn runs on the cheap router model before planning; a hit downgrades the turn to read-only tools and tags the span `guardrail.injection=true`.
- Secrets scanner on fetched content before `save_note`; pages containing API keys or credentials are digested with those spans redacted and the provenance record notes the redaction.
- Private-network and file-scheme URLs are refused at the HTTP client layer, not just by the deny list.

**Cost control**

- Each role maps to a model tier; the orchestrator sees running cost and can downgrade the Auditor to a cheaper model when the turn budget is 80% spent.
- Crawl cost is estimated before approval: pages × average digest tokens × summarizer price, shown in the approval payload.
- The daily cap stops write tools first, then web search, then answers from the store only, with a visible notice in the answer.

## Observability

The trace is a product surface. Span names, kinds and attributes are defined once in `telemetry.py` using OpenInference semantic conventions and exported over OTLP; nothing else in the codebase names a span directly.

**Span tree for one turn**

```text
AGENT  anydocs.turn                      session.id, user.id, corpus.active, policy.name
├─ LLM    orchestrator.plan              model, prompt.version, plan (json)
├─ AGENT  librarian                      budget, result.confidence
│  ├─ LLM       librarian.rewrite        queries[]
│  ├─ RETRIEVER search_docs              candidates per stage, scores, snapshot.id
│  └─ LLM       librarian.draft          citations[]
├─ AGENT  scout                          (only on low confidence or learn request)
│  ├─ TOOL      web_search               query, results[]
│  └─ TOOL      fetch_page               url, status, robots.decision
├─ AGENT  archivist
│  ├─ TOOL      crawl_site               pages.fetched, pages.skipped, snapshot.id
│  ├─ LLM       digest_page (×N)         digest.quality
│  └─ TOOL      save_note (×N)           doc.id, provenance.id
├─ AGENT  auditor
│  ├─ TOOL      get_doc (×k)
│  ├─ TOOL      lint_code                findings[]
│  └─ LLM       auditor.verdict          verdict, confidence, reasons[]
└─ LLM    orchestrator.synthesize        answer, citations[], confidence
```

**Attributes every span carries**

`session.id`, `user.id`, `turn.id`, `corpus.active[]`, `snapshot.id`, `prompt.version`, `model.role`, `model.name`, `policy.name`, `fault.injected` (set by the harness's fault layer, empty in a clean run), `tool.side_effect` (read | write | delete).

**Eval hooks**

The agent writes what evaluators need instead of making them infer it:

- The Auditor's per-claim verdicts and confidence are span attributes, so a groundedness eval can be a cheap code check against them, with an LLM judge as the second opinion.
- The retriever records the gold chunk's rank when a dataset example provides one, so retrieval evals measure recall at k directly.
- Tool-selection intent is recorded on the plan span, so a tool-selection eval compares intent to the tools actually called.
- `confidence` on the turn span is the routing signal for the harness's low-confidence labeling queue.

**Headless traffic**

The freshness workflow and warm-up runs are traced under `anydocs.maintenance` with `user.id=system`, so they show up on dashboards but can be filtered out of user-facing eval scores.

**Local development**

`docker compose up` runs the agent with traces to a local Phoenix instance; the same exporter config points at AX in deployed environments. No code path differs between the two.

## Interfaces

Three doors into one implementation: HTTP for the harness and remote-agent experiments, MCP for bring-your-own-agent clients, and a CLI for operators. Tool code is shared; only the transport differs.

**HTTP API (FastAPI)**

| Route | Purpose | Notes |
| --- | --- | --- |
| `POST /invoke` | One turn: `{session_id, user_id?, message, corpora?, overrides?: {model_roles, prompt_versions, policy}}` → `{answer, citations[], confidence, reasons[], trace_id, pending_approval?}` | Streaming variant via SSE at `/invoke/stream` emits plan, specialist results and the final answer as events |
| `POST /invoke/approve` | `{resume_token, approve: bool, note?}` continues a turn paused on a write action | The decision becomes a span |
| `POST /learn` | \`{source: url | "query", corpus?, max\_pages?}\` runs workflow 2 without a chat turn |
| `GET /corpora`, `GET /corpora/{name}/snapshots` | What the agent knows and how fresh it is | Used by the harness runbook and the MCP `list_corpora` tool |
| `GET /docs/{doc_id}` | Digest, provenance, changelog and links for one doc | Powers citation links in answers |
| `POST /maintenance/refresh` | Runs the freshness workflow once | Called by a scheduler; traced under `anydocs.maintenance` |
| `GET /healthz`, `GET /readyz` | Liveness; readiness waits for the default snapshot to load | — |

All routes except health take a bearer token. `overrides` is how an experiment changes a model or prompt without a redeploy; the override set is recorded on the turn span.

**MCP server (Streamable HTTP at `/mcp`)**

- Tools: the full read set plus `learn` (wraps `/learn` with the same approval semantics) and `ask_anydocs` (a whole turn as one tool, for clients that want the agent rather than its parts).
- Resources: `anydocs://corpora`, `anydocs://docs/{id}` so a client can pull a digest into its own context.
- Prompts: `answer-from-docs`, `compare-versions` as reusable MCP prompts.
- Every MCP call is traced under a separate project name so native and MCP paths compare cleanly.

**CLI (`anydocs`)**

`anydocs ask`, `anydocs learn <url|query>`, `anydocs corpus list|crawl|index|publish|forget`, `anydocs refresh`, `anydocs policy show|set`, `anydocs serve`. The CLI calls the same Python functions the server does, so a crawl from a laptop and a crawl triggered by a user produce identical traces.

## Extending the agent

Adding a tool is one file plus one registry line; adding a specialist is a folder. Both should be doable by Claude Code from this doc without touching the orchestrator.

**Adding a tool**

1. Create `agent/tools/<name>.py` with a Pydantic `Input` and `Output`, a `run(input) -> Output` function, and a `ToolSpec` declaring `side_effect` (read | write | delete), `span_kind`, `cost_hint`, and the policy key it is gated by.
2. Register it in `agent/tools/registry.py`. Registration generates the LLM tool schema, the MCP tool, the permission check, and the span wrapper; the tool author writes none of those.
3. Add the tool's name to the allowlist of whichever specialists may call it in `agent/specialists/<name>/spec.yaml`.
4. Add a unit test with a recorded fixture and an integration test that asserts the span attributes it emits.

**Adding a specialist**

1. Create `agent/specialists/<name>/` with `spec.yaml` (model role, tool allowlist, default budget, result schema) and `prompt.md` (pushed to the prompt hub by the harness seed under a tag).
2. Describe the specialist in `agent/specialists/README.md` in one paragraph: the orchestrator's planning prompt is assembled from these paragraphs, so a new specialist becomes routable without editing the orchestrator prompt by hand.
3. Add a scenario to the harness that exercises it, so there is traffic that reaches it from day one.

**Model roles**

`config/models.yaml` maps roles to providers and models: `router`, `summarizer`, `reader`, `judge`, `synthesizer`. Swapping a provider for a role is a config change; the role name is what appears on spans, so experiments compare roles, not model strings.

**Capability ideas that fit this frame**

- A `Teacher` specialist that turns a corpus into a short tutorial path for a stated goal, using digests' Q&A pairs as checkpoints.
- A `Migrator` specialist that, given two versions of a doc set, produces an upgrade checklist from `compare_versions` diffs and lint findings against the user's pasted code.
- `watch` mode: subscribe a session to a doc; when the freshness workflow finds a change, the next turn opens with what changed.
- Image understanding for screenshots in docs: `fetch_page` extracts images, a vision model writes alt-text into the digest so UI-heavy docs become retrievable.

## Repository layout

One Python package managed with `uv`, one image, one CLI. Proposed repo name: `anydocs-agent`.

```text
anydocs-agent/
├── README.md
├── CLAUDE.md                    # conventions, test commands, span naming rules, "never name a span outside telemetry.py"
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml           # agent + local Phoenix + redis
├── config/
│   ├── models.yaml              # role → provider/model
│   ├── policy.default.yaml      # permissions and budgets
│   └── operator-notes.md        # standing instructions, versioned like a prompt
├── agent/
│   ├── server.py                # FastAPI routes, SSE, auth
│   ├── orchestrator/
│   │   ├── planner.py           # plan → specialist tasks
│   │   ├── synthesizer.py       # final answer + citations
│   │   └── prompt.md
│   ├── specialists/
│   │   ├── runner.py            # ReAct loop with budgets and allowlists
│   │   ├── README.md            # one paragraph per specialist; feeds the planner prompt
│   │   ├── librarian/  {spec.yaml, prompt.md}
│   │   ├── scout/
│   │   ├── archivist/
│   │   ├── analyst/
│   │   └── auditor/
│   ├── tools/
│   │   ├── registry.py          # ToolSpec, schema gen, policy + span wrapping
│   │   ├── search_docs.py  get_doc.py  list_corpora.py
│   │   ├── web_search.py   fetch_page.py  diff_page.py
│   │   ├── crawl_site.py   digest_page.py  save_note.py  link_sources.py
│   │   ├── schedule_refresh.py  forget.py
│   │   ├── query_metrics.py  lint_code.py
│   │   └── http_client.py       # robots.txt, rate limits, private-network refusal
│   ├── knowledge/
│   │   ├── store.py             # snapshot open/close, index I/O
│   │   ├── digest.py            # digest schema and chunking
│   │   ├── retrieval.py         # hybrid search, fusion, rerank, link expansion
│   │   ├── provenance.py
│   │   ├── links.py
│   │   └── freshness.py
│   ├── memory/
│   │   ├── session.py
│   │   └── profile.py
│   ├── policy.py                # loads policy, enforces modes, approval tokens
│   ├── guardrails.py            # injection check, secrets scan
│   ├── telemetry.py             # the only file that names spans
│   └── mcp_server.py
├── analytics/                   # schema + deterministic seed for query_metrics
├── cli/
├── tests/
│   ├── unit/
│   ├── integration/             # tiny fixture corpus, asserted span trees
│   └── fixtures/                # recorded web pages, search results
└── docs/
    ├── design.md                # export of this doc
    ├── adding-a-tool.md
    └── adding-a-specialist.md
```

**Dependencies to pin early**

FastAPI, pydantic v2, httpx, trafilatura or readability for page cleaning, LanceDB (or sqlite-vec), rank-bm25 or tantivy, a reranker client, the OpenInference instrumentors for the LLM SDKs in use, the MCP Python SDK, and `uv` for everything.

## Build order

Build the thin vertical slice first so a trace exists on day two, then widen one specialist at a time. Each step ends with something a trace can show.

1. **Skeleton and telemetry.** Repo, `CLAUDE.md`, FastAPI `/invoke`, `telemetry.py`, Phoenix in compose, an orchestrator that answers with no tools. Done when a single-turn trace shows up locally.
2. **Knowledge store and Librarian.** Snapshot format, digest schema, hybrid retrieval, `search_docs`, `get_doc`, `list_corpora`; ingest one corpus from a local folder with the CLI. Done when an answer carries real citations and a RETRIEVER span with scores.
3. **Auditor.** Per-claim verification, `lint_code`, confidence on the turn span. Done when a deliberately wrong draft is caught and the trace shows why.
4. **Archivist offline.** `fetch_page`, `digest_page`, `save_note`, provenance, `crawl_site` from the CLI only. Done when `anydocs learn <url>` produces a queryable snapshot.
5. **Policy and Scout.** Permission modes, approval tokens, `web_search`, the learn-then-answer loop in workflow 1. Done when a question about an unknown product triggers an approved crawl and a second-pass answer in one trace.
6. **Freshness and links.** `diff_page`, `schedule_refresh`, `link_sources`, contradiction surfacing, the maintenance route. Done when a changed fixture page produces a changelog entry and a "sources disagree" answer.
7. **MCP server and CLI parity.** `/mcp` with tools, resources and prompts; CLI covers every route. Done when the same question over HTTP and MCP yields matching span trees under two project names.
8. **Analyst and memory.** `query_metrics` over seeded SQLite, session memory with rolling summary, opt-in profile. Done when a numeric follow-up question lands on the Analyst with the right filters from context.
9. **Streaming, guardrails, cost.** SSE events, injection check, secrets scan, budgets and daily cap. Done when a budget breach is visible on the span and in the answer.

Steps 1–3 are the minimum the companion harness needs to start generating traffic; 4–5 are what makes the agent new; 6–9 can land in any order.

## Open questions

| Question | Default | Why it matters |
| --- | --- | --- |
| Search provider: Brave, Tavily, Exa, or a self-hosted SearXNG | Tavily behind a `SearchProvider` interface | Result quality for docs sites varies; keep it swappable and record the provider on the span |
| Reranker: cross-encoder (local) vs. LLM rerank vs. a hosted API | Local cross-encoder, LLM fallback | Latency and cost on every turn; local keeps the hot path cheap |
| Digest model: can a cheap model fill the digest schema reliably at scale? | Cheap model with a quality score; re-digest below threshold on the reader model | Crawl cost scales with pages × digest tokens |
| Should the Auditor run on every turn or only below a confidence threshold? | Every turn in v1 | It is the demo's best span; cost can be trimmed later with a cheaper judge on high-confidence turns |
| Approval UX for `ask` mode over MCP | Return `needs_approval` as a tool result with the resume token | MCP clients have no standard approval flow yet |
| Snapshot store: LanceDB vs. SQLite + sqlite-vec | LanceDB | Single-directory snapshots and good hybrid support; sqlite-vec is the fallback if dependency weight matters |
| Where do prompts live at runtime: repo files or prompt hub pull? | Repo files are the source; the harness seed pushes them to the hub; runtime pulls by tag with a file fallback | Lets Signal propose a prompt change as a repo PR and lets experiments override by tag |
| Code sandbox for `run_example` | Deferred; `lint_code` only in v1 | Sandboxing is a project of its own |
| One repo or two: should the harness live here? | Two repos, as discussed | Signal and CI history should read as this agent's story alone |

**Decisions fixed by this doc**

- Orchestrator plus five named specialists; specialists never call each other.
- Fourteen tools in v1, split read/write, registered once, shared by agent and MCP.
- Knowledge is stored as digests with provenance in versioned snapshots; no live-fetch answers.
- Policy is enforced in code with `off | ask | auto` modes per side-effect class.
- `telemetry.py` is the only place that names spans.
