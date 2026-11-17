"""HTTP API. One door into the shared implementation; the CLI and MCP are the others.

Every route except health takes a bearer token. `overrides` is how an experiment changes a model
or a prompt without a redeploy, and the override set is recorded on the turn span.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.config import get_settings
from anydocs.ingest import learn_source
from anydocs.knowledge.doc_view import doc_detail
from anydocs.knowledge.store import CorpusRegistry, StoreError, open_snapshot
from anydocs.mcp_server import build_server
from anydocs.memory.profile import get_profile_store
from anydocs.orchestrator.maintenance import run_refresh
from anydocs.orchestrator.turn import TurnRequest, resume_turn, run_turn
from anydocs.policy import NeedsApproval, PolicyDenied, default_policy
from anydocs.specialists.results import Citation


def create_app() -> FastAPI:
    """Build the application, including the MCP server mounted at `/mcp`.

    A factory rather than a module-level app because an MCP session manager can be run exactly
    once: each app instance needs its own, or a second instance finds the manager already
    started. `app` below is the one uvicorn and the Dockerfile use; tests build their own.
    """
    settings = get_settings()
    stateless = settings.flush_spans_per_request
    mcp = build_server()
    if stateless:
        # A serverless host may not run the lifespan, so do not depend on it. Everything in
        # `_prepare` is idempotent, which makes the second call inside the lifespan harmless.
        _prepare(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        _prepare(settings)
        if stateless:
            # Nothing to run: a stateless MCP app holds no session between requests, which is
            # the only mode that works where consecutive requests may hit different processes.
            yield
        else:
            # Mounted rather than run as its own process, so the session manager has to be
            # started here or `/mcp` accepts connections and then hangs.
            async with mcp.session_manager.run():
                yield
        telemetry.flush_tracing()

    application = FastAPI(
        title="AnyDocs Agent",
        version="0.1.0",
        description=(
            "Answers questions about any product's documentation, and learns what it lacks."
        ),
        lifespan=lifespan,
    )
    application.include_router(_router)

    @application.middleware("http")
    async def trace_context(request, call_next):
        # The caller's `traceparent` becomes the turn span's parent, so an experiment's trace
        # continues into ours. Sync routes run in the threadpool, which copies this context.
        with telemetry.remote_parent(request.headers):
            response = await call_next(request)
            if stateless:
                # The process may be frozen the moment a response is returned, so anything the
                # batch exporter still holds would never be sent.
                telemetry.flush_spans()
        return response

    # One process, one tool layer, three doors: HTTP here, MCP at /mcp, and the CLI calling the
    # same functions directly.
    application.mount(
        "/mcp",
        mcp.streamable_http_app(
            streamable_http_path="/", stateless_http=stateless, json_response=stateless
        ),
    )
    return application


_router = APIRouter()

# Credentials a public deployment must never start with: the code defaults and `.env.example`.
_DEFAULT_SECRETS = {"dev-token", "insecure-dev-key", "change-me-in-deployment"}


def _prepare(settings) -> None:
    """Everything a process needs before its first request. Idempotent.

    In serverless mode this is also where a misconfigured deploy fails: on a default credential,
    or with no corpus to answer from, rather than on the first request.
    """
    from anydocs import bundle
    from anydocs.tools import registry

    if settings.flush_spans_per_request:
        if {settings.api_token, settings.signing_key} & _DEFAULT_SECRETS:
            raise RuntimeError(
                "ANYDOCS_API_TOKEN and ANYDOCS_SIGNING_KEY must be set to real values in "
                "serverless mode; the defaults are public."
            )
        # The deployed files are read-only, so the bundle is copied onto the data directory.
        # Not done elsewhere: a local or container store is built by ingesting, not shipped.
        bundle.materialize(settings.bundle_dir, settings.data_dir)
        if not settings.corpora_registry.exists():
            raise RuntimeError(
                f"No knowledge store at {settings.data_dir} and no bundle at "
                f"{settings.bundle_dir}. Build one with `anydocs bundle`."
            )
    telemetry.ensure_tracing()
    # Load the registry at startup so a bad ToolSpec fails the process, not a request.
    registry.load_tools()



# ----------------------------------------------------------------------------------------
# Auth
# ----------------------------------------------------------------------------------------


def require_token(authorization: str = Header(default="")) -> str:
    expected = get_settings().api_token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Expected an `Authorization: Bearer` header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="Invalid bearer token")
    return token


# ----------------------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------------------


class Overrides(BaseModel):
    model_roles: dict[str, str] = Field(
        default_factory=dict, description="Role -> model id, e.g. {'judge': 'claude-haiku-4-5'}."
    )
    policy: dict[str, Any] = Field(
        default_factory=dict, description="Policy overrides merged one level deep."
    )


class InvokeRequest(BaseModel):
    session_id: str = ""
    user_id: str = ""
    message: str
    corpora: list[str] = Field(default_factory=list)
    overrides: Overrides | None = None


class PendingApproval(BaseModel):
    action: str
    resume_token: str
    reason: str = ""
    tool: str = ""
    estimated_pages: int = 0
    estimated_usd: float = 0.0
    arguments: dict[str, Any] = Field(default_factory=dict)


class InvokeResponse(BaseModel):
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    notice: str = ""
    trace_id: str = ""
    turn_id: str = ""
    cost_usd: float = 0.0
    pending_approval: PendingApproval | None = None


class ApproveRequest(BaseModel):
    resume_token: str
    approve: bool
    note: str = ""


class LearnRequest(BaseModel):
    source: str = Field(description="A documentation URL to crawl, or a query to resolve first.")
    corpus: str = ""
    max_pages: int = Field(default=50, ge=1, le=500)


# ----------------------------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------------------------


@_router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@_router.get("/readyz")
def readyz() -> dict[str, Any]:
    """Ready means the default snapshots load — not merely that the process is up."""
    registry_ = CorpusRegistry()
    corpora = registry_.active_corpora()
    problems: list[str] = []
    for name in corpora:
        try:
            open_snapshot(name).stats()
        except Exception as exc:
            problems.append(f"{name}: {exc}")
    if problems:
        raise HTTPException(status_code=503, detail={"corpora": corpora, "problems": problems})
    return {"status": "ready", "corpora": corpora}


# ----------------------------------------------------------------------------------------
# Turns
# ----------------------------------------------------------------------------------------


@_router.post("/invoke", response_model=InvokeResponse)
def invoke(request: InvokeRequest, _: str = Depends(require_token)) -> InvokeResponse:
    overrides = request.overrides.model_dump(exclude_defaults=True) if request.overrides else {}
    outcome = run_turn(
        TurnRequest(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
            corpora=request.corpora,
            overrides=overrides,
        )
    )
    return _to_response(outcome)


@_router.post("/invoke/stream")
def invoke_stream(request: InvokeRequest, _: str = Depends(require_token)) -> StreamingResponse:
    """The same turn as `/invoke`, emitting progress as Server-Sent Events.

    Events, in order: `plan`, then one `specialist` per specialist as it finishes, `learned` if
    the turn ingested anything, `approval` if it paused, and `answer` or `error` last. The turn
    runs on a worker thread because it is synchronous throughout — the tool layer, the store and
    the provider SDKs are all blocking, and pretending otherwise would mean two implementations.

    A plain thread does not inherit the caller's trace context, so a streamed turn starts its own
    trace even when the request carried a `traceparent`. `/invoke` is the route that continues one.
    """
    import queue
    import threading

    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    DONE = object()

    def on_event(event: str, payload: dict[str, Any]) -> None:
        events.put((event, payload))

    overrides = request.overrides.model_dump(exclude_defaults=True) if request.overrides else {}

    def run() -> None:
        try:
            outcome = run_turn(
                TurnRequest(
                    message=request.message,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    corpora=request.corpora,
                    overrides=overrides,
                    on_event=on_event,
                )
            )
            response = _to_response(outcome)
            if outcome.pending_approval:
                events.put(("approval", response.model_dump(mode="json")["pending_approval"]))
            else:
                events.put(("answer", response.model_dump(mode="json")))
        except Exception as exc:
            # The stream has to carry the failure: a client watching events would otherwise see
            # the connection close with no explanation.
            events.put(("error", {"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            events.put((DONE, None))  # type: ignore[arg-type]

    worker = threading.Thread(target=run, daemon=True, name="anydocs-stream")
    worker.start()

    def stream():
        while True:
            event, payload = events.get()
            if event is DONE:
                break
            yield f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@_router.post("/invoke/approve", response_model=InvokeResponse)
def approve(request: ApproveRequest, _: str = Depends(require_token)) -> InvokeResponse:
    try:
        outcome = resume_turn(request.resume_token, approve=request.approve, note=request.note)
    except PolicyDenied as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return _to_response(outcome)


@_router.post("/learn")
def learn(request: LearnRequest, _: str = Depends(require_token)) -> dict[str, Any]:
    try:
        return learn_source(
            request.source, corpus=request.corpus, max_pages=request.max_pages
        )
    except NeedsApproval as exc:
        return {
            "pending_approval": {
                "action": exc.action,
                "resume_token": exc.resume_token,
                **exc.payload,
            }
        }
    except PolicyDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None


# ----------------------------------------------------------------------------------------
# What the agent knows
# ----------------------------------------------------------------------------------------


@_router.get("/corpora")
def corpora(_: str = Depends(require_token)) -> dict[str, Any]:
    from anydocs.tools import registry

    result = registry.invoke("list_corpora", {}, policy=default_policy(), requested_by="operator")
    return result.model_dump(mode="json")


@_router.get("/corpora/{name}/snapshots")
def snapshots(name: str, _: str = Depends(require_token)) -> dict[str, Any]:
    entry = CorpusRegistry().get(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No corpus named {name!r}")
    out = []
    for snapshot_id in entry.snapshots:
        try:
            out.append(open_snapshot(name, snapshot_id).stats())
        except StoreError as exc:
            out.append({"snapshot_id": snapshot_id, "error": str(exc)})
    return {"corpus": name, "default_snapshot": entry.default_snapshot, "snapshots": out}


@_router.get("/docs/{doc_id}")
def doc(doc_id: str, _: str = Depends(require_token)) -> dict[str, Any]:
    """Digest, provenance, links and changelog for one doc. What a citation resolves to."""
    detail = doc_detail(doc_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No doc {doc_id!r} in the active snapshots")
    return detail


class DeclareRequest(BaseModel):
    user_id: str
    preferences: dict[str, str] = Field(
        default_factory=dict,
        description="What the user declares about themselves. An empty value clears a key.",
    )


class ConfirmRequest(BaseModel):
    user_id: str
    key: str
    accept: bool = True


@_router.get("/profile/{user_id}")
def get_profile(user_id: str, _: str = Depends(require_token)) -> dict[str, Any]:
    """What the agent has been told, and what it has noticed but not acted on."""
    return get_profile_store().get(user_id).to_json()


@_router.post("/profile/declare")
def declare_profile(request: DeclareRequest, _: str = Depends(require_token)) -> dict[str, Any]:
    """Record a declared preference. This is the only thing that changes behaviour."""
    try:
        return get_profile_store().declare(request.user_id, request.preferences).to_json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@_router.post("/profile/confirm")
def confirm_profile(request: ConfirmRequest, _: str = Depends(require_token)) -> dict[str, Any]:
    """Promote a suggestion the agent made into a preference, or drop it.

    The agent never does this for itself — that is the whole point of the suggestion/preference
    split.
    """
    store = get_profile_store()
    return store.confirm(request.user_id, request.key, accept=request.accept).to_json()


@_router.delete("/profile/{user_id}")
def forget_profile(user_id: str, _: str = Depends(require_token)) -> dict[str, str]:
    get_profile_store().forget(user_id)
    return {"status": "forgotten", "user_id": user_id}


class RefreshRequest(BaseModel):
    corpora: list[str] = Field(
        default_factory=list, description="Corpora to check. Empty means every active corpus."
    )
    limit: int = Field(default=25, ge=1, le=500, description="Docs to check per corpus.")


@_router.post("/maintenance/refresh")
def refresh(
    request: RefreshRequest | None = None, _: str = Depends(require_token)
) -> dict[str, Any]:
    """Run the freshness workflow once. Called by a scheduler.

    Traced under `anydocs.maintenance` with `user.id=system`, so it shows up on dashboards and
    can be filtered out of user-facing eval scores.
    """
    body = request or RefreshRequest()
    report = run_refresh(corpora=body.corpora or None, limit=body.limit)
    return report.as_dict()


def require_cron_secret(authorization: str = Header(default="")) -> None:
    """Vercel Cron's credential: `Authorization: Bearer $CRON_SECRET`, on a GET with no body.

    Separate from the API token so the scheduler can be revoked without breaking AX, and so a
    leaked cron secret buys nothing but more scheduled turns.
    """
    expected = get_settings().cron_secret
    if not expected:
        raise HTTPException(status_code=404, detail="Scheduled turns are off; set CRON_SECRET")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(status_code=403, detail="Invalid cron secret")


@_router.get("/cron/turn")
def cron_turn(
    force: bool = False,
    conversation: int | None = None,
    fault: str | None = None,
    _: None = Depends(require_cron_secret),
) -> dict[str, Any]:
    """One tick of synthetic demo traffic. Usually it starts nothing.

    Vercel Cron calls this every few minutes; `demo_traffic.start_probability` decides whether
    this tick starts a session, so a demo project fills at irregular times. `force` starts one
    regardless, and `conversation` and `fault` pin what it runs (`fault=` with no value pins no
    fault), for a manual run.
    """
    import random
    from datetime import UTC, datetime

    from anydocs import demo_traffic

    config = demo_traffic.load()
    rng = random.Random()
    p = demo_traffic.start_probability(config, datetime.now(UTC))
    if not force and rng.random() >= p:
        return {"started": False, "probability": round(p, 4)}
    try:
        session, turns, overrides = demo_traffic.plan(
            config, rng, conversation=conversation, fault=fault
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    corpora = list(config["conversations"][session.conversation].get("corpora") or [])
    demo_traffic.run_session(session, turns, corpora, overrides)
    return {"started": True, "probability": round(p, 4), **session.__dict__}


def _to_response(outcome) -> InvokeResponse:
    if outcome.pending_approval:
        payload = dict(outcome.pending_approval)
        return InvokeResponse(
            turn_id=outcome.turn_id,
            trace_id=outcome.trace_id,
            cost_usd=outcome.cost_usd,
            pending_approval=PendingApproval(
                action=payload.get("action", ""),
                resume_token=payload.get("resume_token", ""),
                reason=payload.get("reason", ""),
                tool=payload.get("tool", ""),
                estimated_pages=int(payload.get("estimated_pages", 0) or 0),
                estimated_usd=float(payload.get("estimated_usd", 0.0) or 0.0),
                arguments=payload.get("arguments", {}) or {},
            ),
        )
    answer = outcome.answer
    return InvokeResponse(
        answer=answer.answer if answer else "",
        citations=answer.citations if answer else [],
        confidence=answer.confidence if answer else 0.0,
        reasons=answer.reasons if answer else [],
        caveats=answer.caveats if answer else [],
        notice=answer.notice if answer else "",
        trace_id=outcome.trace_id,
        turn_id=outcome.turn_id,
        cost_usd=outcome.cost_usd,
    )


# The instance uvicorn and the Dockerfile serve. Tests build their own with `create_app()`.
app = create_app()
