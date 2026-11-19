"""Shared fixtures.

The span-capture fixture is the backbone of the integration suite: it installs an in-memory
exporter so tests assert on real span names, kinds, parent/child shape and attributes rather
than on a mock.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

os.environ.setdefault("TRACE_TARGET", "none")
os.environ.setdefault("ANYDOCS_SIGNING_KEY", "test-signing-key")
os.environ.setdefault("ANYDOCS_API_TOKEN", "test-token")
os.environ.setdefault("SEARCH_PROVIDER", "fixture")
os.environ.setdefault("ANYDOCS_EMBEDDINGS", "fake")

FIXTURES = Path(__file__).parent / "fixtures"


class SpanTree:
    """Captured spans with the few queries the assertions actually need.

    Reads from the exporter on every access, so a test can assert mid-flight rather than only
    after the fixture tears down.
    """

    def __init__(self, exporter, provider) -> None:
        self._exporter = exporter
        self._provider = provider

    @property
    def spans(self) -> list:
        self._provider.force_flush()
        return list(self._exporter.get_finished_spans())

    @property
    def _by_id(self) -> dict:
        return {s.context.span_id: s for s in self.spans}

    def names(self) -> list[str]:
        return [s.name for s in self.spans]

    def find(self, name: str):
        for span in self.spans:
            if span.name == name:
                return span
        raise AssertionError(f"No span named {name!r}. Captured: {sorted(set(self.names()))}")

    def find_all(self, name: str) -> list:
        return [s for s in self.spans if s.name == name]

    def has(self, name: str) -> bool:
        return any(s.name == name for s in self.spans)

    def parent_name(self, name: str) -> str | None:
        by_id = self._by_id
        span = self.find(name)
        if span.parent is None:
            return None
        parent = by_id.get(span.parent.span_id)
        return parent.name if parent else None

    def attr(self, name: str, key: str):
        return self.find(name).attributes.get(key)

    def path(self, name: str) -> list[str]:
        """Span names from the root down to `name`, for asserting tree shape."""
        by_id = self._by_id
        chain = [name]
        current = self.find(name)
        while current.parent is not None:
            parent = by_id.get(current.parent.span_id)
            if parent is None:
                break
            chain.append(parent.name)
            current = parent
        return list(reversed(chain))


@pytest.fixture
def spans(monkeypatch) -> Iterator[SpanTree]:
    """Capture every span emitted inside the test."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from anydocs import telemetry

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "tracer", lambda: provider.get_tracer("anydocs-test"))

    yield SpanTree(exporter, provider)
    provider.force_flush()


@pytest.fixture(autouse=True)
def _no_live_models(monkeypatch):
    """Fail loudly instead of calling a provider.

    The suite promises to run without an API key. Once a real key is in `.env`, nothing else
    stops a stray `digest=True` from quietly spending money and taking 90 seconds — this makes
    that a test failure with a clear message. The `scripted` fixture overrides this.
    """
    from anydocs.models import chat

    def refuse(cfg, **_):
        raise AssertionError(
            f"The test asked for a live {cfg.role!r} model. Install a script with the "
            f"`scripted` fixture, or use a code path that makes no model calls (digest=False)."
        )

    monkeypatch.setattr(chat, "_build", refuse)


@pytest.fixture(autouse=True)
def _clean_ledger():
    from anydocs.budget import reset_ledger

    reset_ledger()
    yield
    reset_ledger()


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    """An isolated knowledge store per test."""
    from anydocs import config

    monkeypatch.setenv("ANYDOCS_DATA_DIR", str(tmp_path))
    config.reset_settings_cache()
    yield tmp_path
    config.reset_settings_cache()


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays a fixed list of replies.

    LangChain's own fakes raise `NotImplementedError` for `bind_tools` and
    `with_structured_output`, which are the two things every call path here uses, so the suite
    needs its own. Integration tests keep the real agent, the real registry and the real spans —
    only the model is substituted.

    Each script entry is either a string (final text) or a list of `(tool_name, arguments)`
    pairs, or a callable taking the messages so far and returning either, for a step whose
    arguments depend on an earlier tool's result.

    A tool named `respond` means "finish with the result schema". `create_agent` binds the schema
    as a tool named after the class, which a test should not have to restate — the double sees
    the bound tools and resolves the sentinel to whichever of them is a result schema.
    """

    responses: list = Field(default_factory=list)
    calls: int = 0
    label: str = "scripted"
    bound: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, messages) -> AIMessage:
        self.calls += 1
        if not self.responses:
            raise AssertionError(
                f"ScriptedChatModel({self.label}) ran out of script after "
                f"{self.calls} calls."
            )
        step = self.responses.pop(0)
        if callable(step):
            step = step(list(messages))
        if isinstance(step, AIMessage):
            return step
        if isinstance(step, str):
            return AIMessage(content=step, usage_metadata=_USAGE)
        return AIMessage(
            content="",
            usage_metadata=_USAGE,
            tool_calls=[
                {"name": self._resolve(name), "args": args,
                 "id": f"call_{self.calls}_{i}", "type": "tool_call"}
                for i, (name, args) in enumerate(step)
            ],
        )

    def _resolve(self, name: str) -> str:
        """Map the `respond` sentinel to the result schema the agent bound."""
        if name != "respond":
            return name
        schemas = [t for t in self.bound if t.endswith("Result")]
        return schemas[0] if schemas else name

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def bind_tools(self, tools, **kwargs):
        """Record what was bound, so `respond` can be resolved, and replay as before."""
        for tool in tools:
            bound_name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if bound_name is None and isinstance(tool, type):
                bound_name = tool.__name__
            if bound_name and bound_name not in self.bound:
                self.bound.append(bound_name)
        return self

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        """Parse the next scripted reply's tool-call arguments into `schema`."""

        def parse(messages):
            reply = self._next(messages)
            args = reply.tool_calls[0]["args"] if reply.tool_calls else {}
            try:
                parsed, error = schema.model_validate(args), None
            except Exception as exc:  # surfaced the way LangChain surfaces it
                parsed, error = None, exc
            if not include_raw:
                if error is not None:
                    raise error
                return parsed
            return {"parsed": parsed, "raw": reply, "parsing_error": error}

        return RunnableLambda(parse)


_USAGE = {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}


@pytest.fixture
def scripted(monkeypatch):
    """Install per-role scripts. Usage: `scripted({"reader": [[("respond", {...})]]})`.

    The role a script is keyed by is the same role `config/models.yaml` names, so a test says
    which *job* it is scripting rather than which model.
    """
    from anydocs.models import chat

    installed: dict[str, ScriptedChatModel] = {}

    def install(scripts: dict[str, list]) -> None:
        for role, script in scripts.items():
            installed[role] = ScriptedChatModel(responses=list(script), label=role)
        chat.set_fake_models(installed)

    yield install
    chat.set_fake_models(None)


@pytest.fixture
def relayr_corpus(data_dir, monkeypatch):
    """The fixture doc corpus, ingested with structure-only digests. No API key needed."""
    monkeypatch.setenv("ANYDOCS_RERANKER", "noop")
    from anydocs.ingest import ingest_folder
    from anydocs.knowledge.embeddings import reset_embeddings

    reset_embeddings()
    report = ingest_folder(
        FIXTURES / "corpus",
        "relayr",
        base_url="https://docs.relayr.example",
        digest=False,
        publish=True,
    )
    yield report
    reset_embeddings()


@pytest.fixture
def lumen_site(monkeypatch):
    """Serve tests/fixtures/pages/ as https://lumen.example.dev, with a robots.txt.

    Used by the crawl and learn-then-answer tests so the whole Archivist path runs without the
    network.
    """
    import httpx
    import respx

    from anydocs import policy as policy_mod
    from anydocs.tools import http_client

    monkeypatch.setattr(policy_mod, "_refuse_private_network", lambda url, host: None)
    monkeypatch.setattr(http_client.time, "sleep", lambda _s: None)
    http_client.reset_robots_cache()

    pages = {
        "https://lumen.example.dev/docs/": "index.html",
        "https://lumen.example.dev/docs/flags": "flags.html",
        "https://lumen.example.dev/docs/rollouts": "rollouts.html",
    }
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://lumen.example.dev/robots.txt").mock(
            return_value=httpx.Response(200, text=(FIXTURES / "pages" / "robots.txt").read_text())
        )
        for url, filename in pages.items():
            mock.get(url).mock(
                return_value=httpx.Response(
                    200,
                    html=(FIXTURES / "pages" / filename).read_text(),
                    headers={"content-type": "text/html; charset=utf-8"},
                )
            )
        # Reachable from the nav but outside /docs/, so robots.txt must keep the crawler out.
        marketing = mock.get("https://lumen.example.dev/marketing/pricing").mock(
            return_value=httpx.Response(200, html="<html><body>Pricing</body></html>")
        )
        mock.marketing_route = marketing

        def serve(url: str, filename: str) -> None:
            """Re-point one route at a different fixture file, on this same router.

            How a test simulates a page being edited between two refresh passes. Starting a
            second respx router instead would not work: the two fight over the same routes and
            the first registered one keeps answering.
            """
            mock.get(url).mock(
                return_value=httpx.Response(
                    200,
                    html=(FIXTURES / "pages" / filename).read_text(),
                    headers={"content-type": "text/html; charset=utf-8"},
                )
            )

        mock.serve = serve
        yield mock


@asynccontextmanager
async def connected_mcp_session(server):
    """An MCP client wired to a server over in-memory streams.

    The SDK dropped its own helper for this in 2.x, so this is the equivalent: run the low-level
    server as a task against one end of a stream pair and initialize a client on the other. The
    point is to exercise the real protocol — listing, calls, resources, prompts — rather than the
    Python functions behind it.
    """
    import anyio
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    low_level = server._lowlevel_server
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as task_group:

            async def serve() -> None:
                await low_level.run(
                    server_read,
                    server_write,
                    low_level.create_initialization_options(),
                    raise_exceptions=True,
                )

            task_group.start_soon(serve)
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                yield session
            task_group.cancel_scope.cancel()


@pytest.fixture
def client(data_dir):
    """The HTTP app under test.

    A fresh app per test: an MCP session manager can only be run once per instance, which is why
    `create_app` is a factory rather than a module-level `app`.
    """
    from starlette.testclient import TestClient

    from anydocs.server import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def analytics(data_dir):
    """The deterministic analytics store, seeded to a fixed date so answers are exact."""
    from datetime import date

    from analytics.seed import seed

    seed(data_dir / "analytics.db", 90, date(2026, 11, 1))
    return data_dir / "analytics.db"
