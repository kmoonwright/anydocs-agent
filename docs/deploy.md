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

Production deploys come from the Git repository: a push to `main` deploys production, any other
branch deploys a preview. The CLI is only for linking, a local build check, and pulling env.

## 1. Keep the bundle in the repo

```bash
uv run anydocs bundle --corpus arize --corpus openinference
git add bundled && git commit -m "Refresh bundled corpus" && git push
```

`bundled/` is committed because the deploy reads it from the repository; `data/` stays
gitignored. `bundle` copies each corpus's default snapshot and a registry naming only those
corpora, and leaves spend, profiles and pending turns behind. A corpus ingested from a local
folder cites `file://` URLs that name the machine it was built on, so `bundle` refuses it —
re-ingest it with `--base-url`, or leave it out. Re-run it, commit and push after a re-crawl.

## 2. Import the repository

In the Vercel dashboard: **Add New → Project**, then import the GitHub repository.

- **Framework Preset:** FastAPI. Vercel should detect it; pick it if it does not. The app itself
  is found through `[tool.vercel] entrypoint` in `pyproject.toml`.
- **Root Directory:** the repository root (`./`).
- **Build and Output Settings, Install Command:** leave every override off. Vercel installs from
  `uv.lock` and picks Python 3.12 from `.python-version`.
- **Environment Variables:** add the block in step 3 *before* the first **Deploy**. Without real
  credentials the function refuses to start, and the first deploy is wasted.

`vercel.json` sets the function's `maxDuration` (300 s) and the files it ships (`bundled/`,
`config/`, `anydocs/`); nothing about it is configured in the dashboard.

## 3. Project env

Set these for the **Production** environment. In the import screen or in **Settings →
Environment Variables**, you can paste the whole block into the first key field and Vercel splits
it into variables.

```
ANYDOCS_API_TOKEN=<random>
ANYDOCS_SIGNING_KEY=<random>
ANYDOCS_DATA_DIR=/tmp/anydocs
ANYDOCS_FLUSH_SPANS=true
ANYDOCS_CRAWL_MODE=off
ANYDOCS_WRITE_MODE=off
ANYDOCS_DELETE_MODE=off
TRACE_TARGET=ax
ANYDOCS_PROJECT_NAME=anydocs-agent
ARIZE_SPACE_ID=
ARIZE_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
CRON_SECRET=<random>
```

Generate the three secrets with `python3 -c "import secrets; print(secrets.token_hex(32))"`. Mark
the keys **Sensitive**. Env changes apply only to new deployments: after changing one, use
**Deployments → ⋯ → Redeploy** on the latest production deployment.

**Settings → Deployment Protection:** Vercel Authentication puts a Vercel login in front of the
URL, which the playground cannot pass. Standard Protection covers previews only, so the
production domain is open; a team may instead enforce it on every deployment, production
included (`curl` then gets a 302 or a `"protection"` JSON body). Either set it to Standard
Protection, or keep it and create a **Protection Bypass for Automation** secret on the same page:
every caller, the playground included, then sends `x-vercel-protection-bypass: <secret>` beside
the bearer token. `vercel curl <deployment-url>/readyz` reaches a protected deployment from a
logged-in CLI, which tells an app failure apart from a protection one.

## 4. Check the build

Open the deployment and read **Build Logs**:

- The install lists neither `arize-phoenix`, `trafilatura`, `numpy` nor `pytest`. If
  `arize-phoenix` appears, the extras were installed and the function is several times larger
  than measured (about 145 MB with the bundle).
- No error about a missing `config/`. Vercel installs the project non-editable, so the imported
  `anydocs` may be the copy in `site-packages`, whose wheel holds neither `config/` nor
  `bundled/`. `config.find_root` takes the first of the package's parent and the working
  directory that holds `config/models.yaml`, and Vercel runs from the repository root.

To reproduce the build locally, link the folder to the imported project and build it:

```bash
vercel link                                    # choose the existing project
vercel pull --yes --environment=production     # writes .vercel/, which is gitignored
vercel build --prod
du -sh .vercel/output/functions/*
```

## 5. First-deploy checks

```bash
URL=https://<production-domain>
T=<ANYDOCS_API_TOKEN>
curl -s $URL/readyz
curl -s -XPOST $URL/invoke -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"message":"How do I send traces to Arize AX?","corpora":["arize"]}'
curl -s -XPOST $URL/invoke -H "Authorization: Bearer $T" -H 'Content-Type: application/json' \
  -d '{"message":"How do I configure Kubernetes ingress?","overrides":{"policy":{"crawl":{"mode":"auto"}}}}'
```

1. `/readyz` lists `arize` and `openinference`. It reads only the registry, so it says nothing
   about the vector index.
2. The first `/invoke` returns an answer with citations. This is the check that sqlite-vec
   loaded: an answer with no citations and "ran out of budget" means every `search_docs` failed,
   and the turn's trace in AX names why.
3. The second returns an answer with caveats and `"pending_approval": null`: the crawl stays off.
4. **Logs** (in the project) show `Tracing to Arize AX (project 'anydocs-agent')`, and
   the turn appears in that AX project.

## 6. Register the remote agent in AX

- URL: `https://<production-domain>/invoke`
- Header: `Authorization: Bearer <ANYDOCS_API_TOKEN>`, plus `x-vercel-protection-bypass` if the
  production domain is protected
- Body: `{"message": "...", "corpora": ["arize"], "overrides": {...}}` — only `message` is required.

Turns take 20–110 s; set the playground's request timeout to 300 s, the function's
`maxDuration`. Run one playground row and confirm the turn span sits under
`agent.experiment.run`. That span is written to the experiment's project and ours to
`ANYDOCS_PROJECT_NAME`, so if the two differ the turn may appear in `anydocs-agent` under a parent
that project does not hold.

## 7. Scheduled demo traces

`vercel.json` has a cron that calls `GET /cron/turn` at the top of every hour. Each call asks one
question from `config/demo-questions.yaml` and traces it like any other turn, with
`user.id=scheduler`, so a demo project always has recent traffic that evals and dashboards can
filter out. At about $0.02–0.05 a turn, hourly is roughly $1 a day.

- `CRON_SECRET` turns it on. Vercel sends it as `Authorization: Bearer $CRON_SECRET`; without it
  the route answers 404. It is not the API token, so either can be rotated alone.
- Cron jobs run against the production deployment only, and are listed under **Settings → Cron
  Jobs**, where **Run** fires one immediately. Its result lands in **Logs** under `/cron/turn`.
- Change the interval in `vercel.json` (`schedule` is cron syntax, UTC). Hobby allows once a day;
  Pro allows any interval.
- Change what it asks in `config/demo-questions.yaml`; a test fails if a question names a corpus
  the bundle does not ship.
- To run one by hand: `curl -H "Authorization: Bearer $CRON_SECRET" -H "x-vercel-protection-bypass:
  $B" "$URL/cron/turn?index=0"`.

## If something fails

| Symptom | Fix |
|---|---|
| Logs: "must be set to real values" | `ANYDOCS_API_TOKEN` or `ANYDOCS_SIGNING_KEY` is unset or a default. Set it, then Redeploy. |
| Logs: "No knowledge store … no bundle" | `bundled/` is not in the pushed commit. Commit and push it. |
| Logs: "Missing config file" | The root resolved to neither the package's parent nor the working directory. Send the log. |
| `search_docs` spans fail with `enable_load_extension` | The runtime's `sqlite3` cannot load extensions (Vercel's cannot). `store._sqlite_module` falls back to `pysqlite3-binary`, a Linux dependency; check the build installed it. |
| Logs: Anthropic or OpenAI `401` | That provider key in Vercel is wrong. Replace it, then Redeploy. |
| No spans in AX | Wrong `ARIZE_SPACE_ID`, `ARIZE_API_KEY` or `ANYDOCS_PROJECT_NAME`. |
| Turn is its own root, not under the run | The playground is not sending `traceparent`, or it calls `/invoke/stream`. |
| Vercel login page, 302, or a `"protection"` JSON body | Deployment Protection. Use the production domain, or if that is protected too, the bypass header from step 3. |
| `/cron/turn` 404 | `CRON_SECRET` is unset in Production, or set after the last deploy. |
| Slow responses even on `/readyz` | The trace collector is unreachable and the flush is retrying; check the Arize endpoint and keys. |

## Known limits

- The daily USD cap is per instance: `spend.json` lives on that instance's `/tmp`.
- `/invoke/stream` runs its turn on a plain thread, which does not inherit the caller's trace
  context. Register `/invoke`.
