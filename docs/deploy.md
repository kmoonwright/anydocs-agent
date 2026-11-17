# Deploying the playground endpoint to Vercel

Vercel hosts one thing: a read-only `/invoke` the AX agent playground can call. Learning,
crawling, approval and the stateful MCP session stay on the container from the `Dockerfile`.

With crawl `off`, a question the corpus covers is answered as usual, and one it does not cover
goes straight to verify and synthesize instead of the Scout — so the playground always gets an
answer, never `pending_approval`.

## What makes it work

| Piece | Where | Why |
|---|---|---|
| `bundled/` | `anydocs bundle` | The corpus ships with the code; `data/` stays local. |
| Copy onto `/tmp` | `server._prepare` | Deployed files are read-only and SQLite writes a journal even to read. |
| Side-effect ceiling | `ANYDOCS_*_MODE`, `policy._clamp` | A request's `overrides.policy` can tighten a mode, never raise it. |
| Trace continuation | `telemetry.remote_parent` | The turn span lands under the experiment's `agent.experiment.run`. |
| Per-request flush | `ANYDOCS_FLUSH_SPANS=true` | The host may freeze the process the moment a response is sent. |

In serverless mode startup fails, rather than the first request, if the API token or signing key
is still a default, or if there is no store and no bundle to copy.

## 1. Build the bundle

```bash
uv run anydocs bundle --corpus arize --corpus openinference
```

This copies each corpus's default snapshot and a registry naming only those corpora. Spend,
profiles and pending turns stay behind. A corpus ingested from a local folder cites `file://`
URLs that name the machine it was built on, so `bundle` refuses it — re-ingest it with
`--base-url`, or leave it out. Re-run this after a re-crawl.

## 2. Check the build before deploying

```bash
vercel build
du -sh .vercel/output/functions/*
```

In the function's output, confirm `config/`, `bundled/`, `anydocs/**/prompt.md` and
`anydocs/specialists/*/spec.yaml` are present and that `.env`, `data/` and `.arize-tmp-traces/`
are not. `numpy`, `trafilatura`, `lxml`, `babel` and `uvloop` should be absent too: crawling is
the `crawl` extra, which only the container installs.

Vercel installs the project non-editable, so the `anydocs` that Python imports may be the copy in
`site-packages`, whose wheel has no `config/` or `bundled/`. `config.find_root` handles this by
taking the first of the package's parent and the working directory that holds
`config/models.yaml`. Vercel runs from the project root, and `includeFiles` ships `config/` and
`bundled/` there.

## 3. Project env

Set these in the Vercel project. They never go in an uploaded file.

```
ANYDOCS_API_TOKEN=<random>
ANYDOCS_SIGNING_KEY=<random>
ANYDOCS_DATA_DIR=/tmp/anydocs
ANYDOCS_FLUSH_SPANS=true
ANYDOCS_CRAWL_MODE=off
ANYDOCS_WRITE_MODE=off
ANYDOCS_DELETE_MODE=off
TRACE_TARGET=ax
ANYDOCS_PROJECT_NAME=Agent Experiment Traces
ARIZE_SPACE_ID=
ARIZE_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

Generate the two secrets with `python -c "import secrets; print(secrets.token_hex(32))"`.

Deployment Protection can put a Vercel login in front of preview URLs, which the playground
cannot pass. Register the production URL, or turn protection off for this project.

## 4. Register the remote agent in AX

- URL: `https://<project>.vercel.app/invoke`
- Header: `Authorization: Bearer <ANYDOCS_API_TOKEN>`
- Body: `{"message": "...", "corpora": ["arize"], "overrides": {...}}` — only `message` is required.

Turns take 20–105 s; make sure the playground's request timeout allows that. `maxDuration` is 300.

## 5. First-deploy checks

1. `GET /readyz` lists `arize` and `openinference`. This also proves `sqlite-vec` loaded through
   `enable_load_extension`, since opening a snapshot loads it.
2. One authenticated `/invoke` against `arize` returns an answer.
3. In AX, project *Agent Experiment Traces*, a playground row shows the turn span under
   `agent.experiment.run`.
4. `/invoke` with `"overrides": {"policy": {"crawl": {"mode": "auto"}}}` on an uncovered question
   still returns no `pending_approval`.

## Known limits

- The daily USD cap is per instance: `spend.json` lives on that instance's `/tmp`.
- `/invoke/stream` runs its turn on a plain thread, which does not inherit the caller's trace
  context. Register `/invoke`.
