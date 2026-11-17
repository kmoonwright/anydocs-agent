"""`anydocs` — the operator CLI.

Calls the same Python functions the server does, so a crawl from a laptop and a crawl triggered
by a user produce identical traces.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from anydocs import telemetry
from anydocs.config import get_settings

app = typer.Typer(
    help="AnyDocs: answer questions about any product's documentation, and learn what is missing.",
    no_args_is_help=True,
)
corpus_app = typer.Typer(
    help="Manage documentation corpora and their snapshots.", no_args_is_help=True
)
policy_app = typer.Typer(help="Inspect the permission policy.", no_args_is_help=True)
profile_app = typer.Typer(help="Manage the opt-in user profile.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(policy_app, name="policy")
app.add_typer(profile_app, name="profile")


def _echo_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _remote(method: str, path: str, **kwargs) -> dict:
    """Call a running server instead of doing the work in this process.

    `--remote` exists for one reason: a turn that pauses for approval is suspended in the
    process that paused it, and operating on a deployed agent means talking to that process.
    Everything else the CLI does is identical either way, because both call the same functions.
    """
    import httpx

    settings = get_settings()
    base = os.getenv("ANYDOCS_REMOTE_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        response = httpx.request(
            method,
            f"{base}{path}",
            headers={"Authorization": f"Bearer {settings.api_token}"},
            timeout=180.0,
            **kwargs,
        )
    except httpx.HTTPError as exc:
        typer.secho(f"Could not reach {base}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from None
    if response.status_code >= 400:
        typer.secho(f"{response.status_code}: {response.text}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    return response.json()


# ----------------------------------------------------------------------------------------
# ask
# ----------------------------------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    corpus: list[str] = typer.Option([], "--corpus", "-c", help="Restrict to these corpora."),
    session: str = typer.Option("", "--session", help="Session id, for follow-up questions."),
    judge_model: str = typer.Option("", "--judge-model", help="Override the judge role's model."),
    as_json: bool = typer.Option(False, "--json", help="Print the full result as JSON."),
    remote: bool = typer.Option(
        False, "--remote", help="Ask a running server instead of answering in this process."
    ),
) -> None:
    """Ask one question and print the answer with its citations."""
    from anydocs.orchestrator.synthesizer import FinalAnswer
    from anydocs.orchestrator.turn import TurnOutcome, TurnRequest, run_turn
    from anydocs.specialists.results import Citation

    if remote:
        body = _remote(
            "POST",
            "/invoke",
            json={
                "session_id": session,
                "user_id": "operator",
                "message": question,
                "corpora": list(corpus),
                **({"overrides": {"model_roles": {"judge": judge_model}}} if judge_model else {}),
            },
        )
        if as_json:
            _echo_json(body)
            return
        outcome = TurnOutcome(
            turn_id=body.get("turn_id", ""),
            trace_id=body.get("trace_id", ""),
            cost_usd=body.get("cost_usd", 0.0),
            pending_approval=body.get("pending_approval"),
            answer=None
            if body.get("pending_approval")
            else FinalAnswer(
                answer=body.get("answer", ""),
                citations=[Citation.model_validate(c) for c in body.get("citations", [])],
                confidence=body.get("confidence", 0.0),
                reasons=body.get("reasons", []),
                caveats=body.get("caveats", []),
                notice=body.get("notice", ""),
            ),
        )
        _print_answer(outcome)
        return

    telemetry.ensure_tracing()
    try:
        overrides = {"model_roles": {"judge": judge_model}} if judge_model else {}
        outcome = run_turn(
            TurnRequest(
                message=question,
                session_id=session,
                user_id="operator",
                corpora=list(corpus),
                overrides=overrides,
                requested_by="operator",
            )
        )
        if as_json:
            _echo_json(
                {
                    "turn_id": outcome.turn_id,
                    "trace_id": outcome.trace_id,
                    "answer": outcome.answer.model_dump(mode="json") if outcome.answer else None,
                    "pending_approval": outcome.pending_approval,
                    "plan": outcome.plan,
                    "specialists": outcome.specialists,
                    "cost_usd": outcome.cost_usd,
                }
            )
            return

        _print_answer(outcome)
    finally:
        telemetry.flush_tracing()


def _print_answer(outcome) -> None:
    if outcome.pending_approval:
        pending = outcome.pending_approval
        typer.secho("Approval required", fg=typer.colors.YELLOW, bold=True)
        typer.echo(
            f"  {pending.get('action')} — about {pending.get('estimated_pages', 0)} pages, "
            f"roughly ${pending.get('estimated_usd', 0):.2f}"
        )
        typer.echo(f"  resume token: {pending.get('resume_token')}")
        typer.echo("\nApprove with:  anydocs approve <token>")
        return

    answer = outcome.answer
    typer.echo(f"\n{answer.answer}\n")
    if answer.citations:
        typer.secho("Sources", bold=True)
        for i, citation in enumerate(answer.citations, 1):
            heading = f" — {citation.heading}" if citation.heading else ""
            typer.echo(f"  [{i}] {citation.title or citation.doc_id}{heading}")
            typer.echo(f"      {citation.url}")
    typer.echo(
        f"\nconfidence {answer.confidence:.2f} · ${outcome.cost_usd:.4f} · "
        f"trace {outcome.trace_id}"
    )
    for caveat in answer.caveats:
        typer.secho(f"  ! {caveat}", fg=typer.colors.YELLOW)


@app.command()
def approve(
    resume_token: str = typer.Argument(..., help="Token from a pending approval."),
    decline: bool = typer.Option(False, "--decline", help="Decline instead of approving."),
    note: str = typer.Option("", "--note", help="Why, recorded on the decision span."),
    remote: bool = typer.Option(
        False, "--remote", help="Approve against a running server, where the turn is suspended."
    ),
) -> None:
    """Continue a turn that paused for approval."""
    from anydocs.orchestrator.turn import resume_turn

    if remote:
        body = _remote(
            "POST",
            "/invoke/approve",
            json={"resume_token": resume_token, "approve": not decline, "note": note},
        )
        typer.echo(f"\n{body.get('answer') or '(no answer)'}\n")
        typer.echo(f"trace {body.get('trace_id', '')}")
        return

    telemetry.ensure_tracing()
    try:
        outcome = resume_turn(resume_token, approve=not decline, note=note)
        typer.echo(f"\n{outcome.answer.answer if outcome.answer else '(no answer)'}\n")
        typer.echo(f"trace {outcome.trace_id}")
    finally:
        telemetry.flush_tracing()


@app.command()
def learn(
    source: str = typer.Argument(..., help="A documentation URL, or a query to resolve first."),
    corpus: str = typer.Option("", "--corpus", help="Corpus name to write into."),
    max_pages: int = typer.Option(50, "--max-pages", help="Cap on pages crawled."),
) -> None:
    """Learn a new documentation source without a chat turn (workflow 2)."""
    from anydocs.ingest import learn_source

    telemetry.ensure_tracing()
    try:
        _echo_json(learn_source(source, corpus=corpus, max_pages=max_pages))
    finally:
        telemetry.flush_tracing()


@app.command()
def refresh(
    corpus: list[str] = typer.Option([], "--corpus", "-c", help="Corpora to check."),
    limit: int = typer.Option(25, "--limit", help="Docs to check per corpus."),
    no_digest: bool = typer.Option(
        False, "--no-digest", help="Re-digest changed pages from structure, without a model."
    ),
) -> None:
    """Run the freshness workflow once: re-check due pages and re-digest what changed."""
    from anydocs.orchestrator.maintenance import run_refresh

    telemetry.ensure_tracing()
    try:
        report = run_refresh(corpora=list(corpus) or None, limit=limit, digest=not no_digest)
        _echo_json(report.as_dict())
    finally:
        telemetry.flush_tracing()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
    reload: bool = typer.Option(False, "--reload", help="Reload on code changes."),
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("anydocs.server:app", host=host, port=port, reload=reload)


@app.command()
def bundle(
    corpus: list[str] = typer.Option(..., "--corpus", help="Corpus to ship. Repeat for more."),
    out: Path = typer.Option(Path("bundled"), "--out", help="Where to write the bundle."),
) -> None:
    """Build the read-only corpus a serverless deploy copies onto its data directory."""
    from anydocs.bundle import BundleError, export

    settings = get_settings()
    try:
        shipped = export(settings.data_dir, out, corpus)
    except BundleError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from None
    typer.echo(f"Bundled {', '.join(shipped)} into {out}")


# ----------------------------------------------------------------------------------------
# corpus
# ----------------------------------------------------------------------------------------


@corpus_app.command("list")
def corpus_list() -> None:
    """List corpora, their default snapshot and page counts."""
    from anydocs.knowledge.store import CorpusRegistry, open_snapshot

    registry = CorpusRegistry()
    entries = registry.list()
    if not entries:
        typer.echo("No corpora yet. Try: anydocs corpus ingest <folder> --name <corpus>")
        return
    for entry in entries:
        pages = ""
        if entry.default_snapshot:
            try:
                stats = open_snapshot(entry.name, entry.default_snapshot).stats()
                pages = f"{stats['page_count']} pages"
            except Exception as exc:
                pages = f"(unreadable: {exc})"
        typer.echo(f"{entry.name:24s} {entry.default_snapshot:24s} {pages}")


@corpus_app.command("ingest")
def corpus_ingest(
    folder: Path = typer.Argument(..., help="Folder of .md files to ingest."),
    name: str = typer.Option(..., "--name", help="Corpus name."),
    base_url: str = typer.Option("", "--base-url", help="Prefix for citation URLs."),
    no_digest: bool = typer.Option(
        False, "--no-digest", help="Fill digests from page structure instead of calling a model."
    ),
    no_publish: bool = typer.Option(False, "--no-publish", help="Do not make this the default."),
) -> None:
    """Ingest a local folder of markdown as a corpus."""
    from anydocs.ingest import ingest_folder

    telemetry.ensure_tracing()
    try:
        report = ingest_folder(
            folder,
            name,
            base_url=base_url,
            digest=not no_digest,
            publish=not no_publish,
        )
        _echo_json(report.as_dict())
    finally:
        telemetry.flush_tracing()


@corpus_app.command("crawl")
def corpus_crawl(
    root_url: str = typer.Argument(..., help="Documentation root to crawl."),
    name: str = typer.Option(..., "--name", help="Corpus name."),
    max_pages: int = typer.Option(50, "--max-pages"),
    max_depth: int = typer.Option(3, "--max-depth"),
    no_digest: bool = typer.Option(
        False, "--no-digest", help="Crawl and save without calling a model for digests."
    ),
    exclude: list[str] = typer.Option(
        [], "--exclude", help="Glob of URL paths to skip, repeatable."
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Crawl a documentation site into a new snapshot, digest it and index it."""
    from anydocs.ingest import crawl_and_ingest
    from anydocs.policy import policy_with_overrides

    # `crawl` is gated by the policy like any other write, and the policy default is `ask`. An
    # operator at a terminal who typed this command and confirmed the page cap has given exactly
    # the approval that mode exists to collect, so there is nobody left to ask.
    if not yes:
        typer.confirm(
            f"Crawl up to {max_pages} pages from {root_url} into corpus {name!r}?", abort=True
        )

    telemetry.ensure_tracing()
    try:
        _echo_json(
            crawl_and_ingest(
                root_url,
                name,
                max_pages=max_pages,
                max_depth=max_depth,
                exclude=list(exclude),
                digest=not no_digest,
                policy=policy_with_overrides({"crawl": {"mode": "auto"}}),
            ).as_dict()
        )
    finally:
        telemetry.flush_tracing()


@corpus_app.command("index")
def corpus_index(
    name: str = typer.Argument(..., help="Corpus to re-index."),
    snapshot: str = typer.Option("", "--snapshot", help="Snapshot id; default if omitted."),
) -> None:
    """Rebuild a snapshot's index from its docs.jsonl."""
    from anydocs.knowledge.store import open_snapshot

    snap = open_snapshot(name, snapshot or None)
    typer.echo(f"Indexed {snap.reindex()} nodes into {name}/{snap.snapshot_id}")


@corpus_app.command("publish")
def corpus_publish(
    name: str = typer.Argument(...),
    snapshot: str = typer.Argument(..., help="Snapshot id to make the default."),
) -> None:
    """Make a snapshot the corpus default, which is what retrieval reads."""
    from anydocs.knowledge.store import publish_snapshot

    entry = publish_snapshot(name, snapshot)
    typer.echo(f"{name} default snapshot is now {entry.default_snapshot}")


@corpus_app.command("forget")
def corpus_forget(
    name: str = typer.Argument(...),
    doc_id: str = typer.Option(..., "--doc-id"),
    reason: str = typer.Option(..., "--reason", help="Recorded in provenance. Required."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Tombstone one doc: drop it from retrieval, keep the record that it existed."""
    from anydocs.policy import policy_with_overrides
    from anydocs.tools.registry import invoke

    if not yes:
        typer.confirm(
            f"Remove {doc_id} from {name} and drop its content? Provenance is kept.", abort=True
        )

    telemetry.ensure_tracing()
    try:
        result = invoke(
            "forget",
            {"doc_id": doc_id, "reason": reason, "corpus": name},
            # An operator at a terminal who passed --yes has given the approval that `ask` mode
            # exists to collect.
            policy=policy_with_overrides({"delete": {"mode": "auto"}}),
            requested_by="operator",
        )
        _echo_json(result.model_dump(mode="json"))
    finally:
        telemetry.flush_tracing()


@corpus_app.command("links")
def corpus_links(
    name: str = typer.Argument(...),
    doc_id: str = typer.Option("", "--doc-id", help="Only edges touching this doc."),
) -> None:
    """Show related-doc edges, including which sources disagree."""
    from anydocs.knowledge.links import links_for, load_links
    from anydocs.knowledge.store import open_snapshot

    snapshot = open_snapshot(name)
    edges = links_for(snapshot, [doc_id]) if doc_id else load_links(snapshot)
    if not edges:
        typer.echo("No links recorded.")
        return
    for link in edges:
        typer.echo(f"{link.kind:12s} {link.from_doc_id} -> {link.to_doc_id}")
        if link.evidence:
            typer.echo(f"             {link.evidence[:160]}")


@corpus_app.command("schedule")
def corpus_schedule(
    name: str = typer.Argument(...),
    doc_id: str = typer.Option("", "--doc-id", help="Schedule one doc instead of the corpus."),
    cadence_days: int = typer.Option(
        0, "--cadence-days", help="Days between checks. 0 picks a default per page character."
    ),
    show: bool = typer.Option(False, "--show", help="Print the schedule instead of changing it."),
) -> None:
    """Set or show how often a corpus is re-checked for changes."""
    from anydocs.knowledge.freshness import load_schedule
    from anydocs.knowledge.store import open_snapshot
    from anydocs.policy import policy_with_overrides
    from anydocs.tools.registry import invoke

    if show:
        for entry in sorted(load_schedule(open_snapshot(name)), key=lambda e: e.next_check):
            typer.echo(
                f"{entry.doc_id}  every {entry.cadence_days}d  next {entry.next_check or 'now'}"
                f"  last {entry.last_result or '-'}"
            )
        return

    telemetry.ensure_tracing()
    try:
        result = invoke(
            "schedule_refresh",
            {"doc_id": doc_id, "corpus": name, "cadence_days": cadence_days},
            policy=policy_with_overrides({"write": {"mode": "auto"}}),
            requested_by="operator",
        )
        _echo_json(result.model_dump(mode="json"))
    finally:
        telemetry.flush_tracing()


# ----------------------------------------------------------------------------------------
# docs, snapshots and profile
# ----------------------------------------------------------------------------------------


@app.command("docs")
def docs_command(
    doc_id: str = typer.Argument(..., help="Doc id from a citation."),
    chunks: bool = typer.Option(False, "--chunks", help="Print the raw chunks too."),
) -> None:
    """Show one doc: its digest, provenance, links and changelog. What a citation resolves to."""
    from anydocs.knowledge.doc_view import doc_detail

    detail = doc_detail(doc_id, include_chunks=chunks)
    if detail is None:
        typer.secho(f"No doc {doc_id!r} in the active snapshots", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    _echo_json(detail)


@corpus_app.command("snapshots")
def corpus_snapshots(name: str = typer.Argument(...)) -> None:
    """List every snapshot of a corpus, and which one retrieval reads."""
    from anydocs.knowledge.store import CorpusRegistry, StoreError, open_snapshot

    entry = CorpusRegistry().get(name)
    if entry is None:
        typer.secho(f"No corpus named {name!r}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    for snapshot_id in entry.snapshots:
        marker = "*" if snapshot_id == entry.default_snapshot else " "
        try:
            stats = open_snapshot(name, snapshot_id).stats()
            typer.echo(
                f"{marker} {snapshot_id}  {stats['page_count']:4d} pages  "
                f"{stats['node_count']:5d} nodes  {stats['embedding_model']}"
            )
        except StoreError as exc:
            typer.echo(f"{marker} {snapshot_id}  (unreadable: {exc})")


# ----------------------------------------------------------------------------------------
# profile
# ----------------------------------------------------------------------------------------


@profile_app.command("show")
def profile_show(user_id: str = typer.Argument(..., help="The user to inspect.")) -> None:
    """Show what the agent has been told, and what it has noticed but not acted on."""
    from anydocs.memory.profile import get_profile_store

    profile = get_profile_store().get(user_id)
    _echo_json(profile.to_json())
    if profile.suggestions:
        typer.secho(
            "\nSuggestions are not in effect. Promote one with: "
            f"anydocs profile confirm {user_id} --key <key>",
            fg=typer.colors.YELLOW,
        )


@profile_app.command("declare")
def profile_declare(
    user_id: str = typer.Argument(...),
    set_: list[str] = typer.Option(
        ..., "--set", help="key=value, repeatable. An empty value clears the key."
    ),
) -> None:
    """Record what a user says about themselves. This is what changes behaviour."""
    from anydocs.memory.profile import DECLARABLE, get_profile_store

    preferences: dict[str, str] = {}
    for item in set_:
        if "=" not in item:
            typer.secho(f"Expected key=value, got {item!r}", fg=typer.colors.RED)
            typer.echo(f"Declarable keys: {', '.join(sorted(DECLARABLE))}")
            raise typer.Exit(code=1)
        key, value = item.split("=", 1)
        preferences[key.strip()] = value.strip()

    try:
        _echo_json(get_profile_store().declare(user_id, preferences).to_json())
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from None


@profile_app.command("confirm")
def profile_confirm(
    user_id: str = typer.Argument(...),
    key: str = typer.Option(..., "--key", help="The suggestion to act on."),
    reject: bool = typer.Option(False, "--reject", help="Drop it instead of accepting."),
) -> None:
    """Promote a suggestion the agent made, or drop it. Only a human does this."""
    from anydocs.memory.profile import get_profile_store

    _echo_json(get_profile_store().confirm(user_id, key, accept=not reject).to_json())


@profile_app.command("forget")
def profile_forget(
    user_id: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Delete a user's profile entirely."""
    from anydocs.memory.profile import get_profile_store

    if not yes:
        typer.confirm(f"Delete the profile for {user_id!r}?", abort=True)
    get_profile_store().forget(user_id)
    typer.echo(f"Forgot the profile for {user_id}")


# ----------------------------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------------------------


@policy_app.command("show")
def policy_show() -> None:
    """Print the effective policy and the per-turn budget."""
    _echo_json(get_settings().policy)


@policy_app.command("tools")
def policy_tools() -> None:
    """List registered tools with their side effect and the policy key gating each."""
    from anydocs.tools.registry import all_specs

    for name, spec in sorted(all_specs().items()):
        typer.echo(
            f"{name:18s} {spec.side_effect:7s} {spec.span_kind:10s} "
            f"policy={spec.policy_key:11s} cost={spec.cost_hint}"
        )


if __name__ == "__main__":
    app()
