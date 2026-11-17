"""The HTTP surface: auth, the turn routes, and what the agent says it knows."""

from __future__ import annotations

from fastapi.testclient import TestClient

from anydocs.server import create_app

TOKEN = {"Authorization": "Bearer test-token"}


def test_health_needs_no_token(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_every_other_route_requires_a_bearer_token(client):
    assert client.post("/invoke", json={"message": "hi"}).status_code == 401
    assert client.get("/corpora").status_code == 401
    assert (
        client.post("/invoke", json={"message": "hi"}, headers={"Authorization": "Bearer wrong"})
    ).status_code == 403


def test_readyz_reports_the_corpora_it_loaded(client, relayr_corpus):
    body = client.get("/readyz", headers=TOKEN).json()
    assert body["status"] == "ready"
    assert "relayr" in body["corpora"]


def test_corpora_lists_what_the_agent_knows(client, relayr_corpus):
    body = client.get("/corpora", headers=TOKEN).json()
    names = {c["name"]: c for c in body["corpora"]}
    assert names["relayr"]["page_count"] == 11
    assert names["relayr"]["embedding_model"] == "fake-embeddings"


def test_snapshots_lists_every_snapshot_of_a_corpus(client, relayr_corpus):
    body = client.get("/corpora/relayr/snapshots", headers=TOKEN).json()
    assert body["default_snapshot"] == relayr_corpus.snapshot_id
    assert body["snapshots"][0]["page_count"] == 11
    assert client.get("/corpora/nope/snapshots", headers=TOKEN).status_code == 404


def test_a_doc_route_resolves_a_citation(client, relayr_corpus):
    from anydocs.knowledge.digest import doc_id_for

    doc_id = doc_id_for("relayr", "https://docs.relayr.example/retries.md")
    body = client.get(f"/docs/{doc_id}", headers=TOKEN).json()
    assert body["doc"]["doc_id"] == doc_id
    assert body["doc"]["chunks"]
    assert body["provenance"][0]["doc_id"] == doc_id
    assert client.get("/docs/not-a-doc", headers=TOKEN).status_code == 404


def _receipts_turn(doc_id: str) -> dict:
    """A scripted turn that answers the receipts question from the relayr corpus."""
    return {
        "router": [
            [
                (
                    "plan",
                    {
                        "specialists": ["librarian", "auditor"],
                        "corpora": ["relayr"],
                        "intent_tools": ["search_docs"],
                        "needs_learning": False,
                        "is_numeric": False,
                        "sub_questions": [],
                        "rationale": "The relayr corpus covers this.",
                    },
                )
            ]
        ],
        "reader": [
            [("search_docs", {"query": "receipt retention", "k": 4})],
            [
                (
                    "respond",
                    {
                        "draft": "Receipts are retained for 90 days [1].",
                        "citations": [
                            {
                                "doc_id": doc_id,
                                # Deliberately wrong and blank: these are the model's
                                # claims, and the turn replaces them with the store's.
                                "url": "https://wrong.example/made-up",
                                "title": "",
                                "heading": "",
                                "corpus": "",
                                "quote": "Receipts are retained for 90 days.",
                            }
                        ],
                        "retrieval_confidence": 0.88,
                    },
                )
            ],
        ],
        "judge": [
            [
                (
                    "respond",
                    {
                        "verdict": "pass",
                        "claims": [
                            {
                                "claim": "Receipts are retained for 90 days.",
                                "verdict": "supported",
                                "evidence": "Receipts are retained for 90 days.",
                            }
                        ],
                        "confidence": 0.9,
                        "reasons": ["Quoted from the cited page."],
                    },
                )
            ]
        ],
        "synthesizer": ["Receipts are retained for 90 days [1]."],
    }


def test_invoke_returns_an_answer_with_citations(client, relayr_corpus, scripted):
    from anydocs.knowledge.digest import doc_id_for

    doc_id = doc_id_for("relayr", "https://docs.relayr.example/receipts.md")
    scripted(_receipts_turn(doc_id))
    response = client.post(
        "/invoke",
        json={"session_id": "s1", "message": "how long are receipts kept?", "corpora": ["relayr"]},
        headers=TOKEN,
    )
    body = response.json()
    assert response.status_code == 200
    assert "90 days" in body["answer"]
    assert body["confidence"] == 0.9
    assert body["trace_id"]
    assert body["pending_approval"] is None

    # The citation is what the store holds, not what the model said about it.
    citation = body["citations"][0]
    assert citation["corpus"] == "relayr"
    assert citation["url"] == "https://docs.relayr.example/receipts.md"
    assert citation["title"] == "Delivery receipts"
    assert citation["heading"].startswith("Delivery receipts")
    assert citation["snapshot_id"] == relayr_corpus.snapshot_id


def test_approve_rejects_a_token_it_never_issued(client, relayr_corpus):
    response = client.post(
        "/invoke/approve", json={"resume_token": "forged.token", "approve": True}, headers=TOKEN
    )
    assert response.status_code == 400


def test_the_maintenance_route_survives_pages_it_cannot_fetch(client, relayr_corpus):
    """The fixture corpus points at a host that does not exist. One dead page must not abandon
    the rest of the pass, and the route must not 500."""
    response = client.post("/maintenance/refresh", json={"limit": 3}, headers=TOKEN)
    assert response.status_code == 200
    body = response.json()
    assert body["checked"] == 3
    assert body["failed"] == 3
    assert all(d["result"] == "failed" for d in body["docs"])
    # Backed off rather than left due, so a dead link is not retried every cycle.
    assert all(d["next_check"] for d in body["docs"])


def test_a_doc_route_carries_its_links_and_changelog(client, relayr_corpus):
    from anydocs.knowledge.digest import doc_id_for

    doc_id = doc_id_for("relayr", "https://docs.relayr.example/retries.md")
    body = client.get(f"/docs/{doc_id}", headers=TOKEN).json()
    assert body["links"] == []
    assert body["changelog"] == []


def test_serverless_mode_flushes_spans_and_goes_stateless(data_dir, monkeypatch, relayr_corpus):
    """On a host that freezes the process after the response, a batched exporter never reaches
    its timer and the deployment traces nothing while looking healthy."""
    from anydocs import config, telemetry

    monkeypatch.setenv("ANYDOCS_FLUSH_SPANS", "true")
    config.reset_settings_cache()
    flushed: list[int] = []
    monkeypatch.setattr(telemetry, "flush_spans", lambda: flushed.append(1))

    try:
        with TestClient(create_app()) as client:
            assert client.get("/healthz").json() == {"status": "ok"}
            assert client.get("/corpora", headers=TOKEN).status_code == 200
    finally:
        config.reset_settings_cache()

    # One flush per request, and the MCP mount came up without a session manager.
    assert len(flushed) >= 2


def test_the_default_keeps_the_stateful_mcp_session(data_dir, monkeypatch):
    """A long-running container keeps one process, so the session manager is the right mode and
    flushing per request would be wasted work."""
    from anydocs import config

    monkeypatch.delenv("ANYDOCS_FLUSH_SPANS", raising=False)
    config.reset_settings_cache()
    try:
        assert config.get_settings().flush_spans_per_request is False
    finally:
        config.reset_settings_cache()


def test_a_turn_continues_the_callers_trace(client, relayr_corpus, scripted, spans):
    """An AX experiment sends `traceparent`; the turn must land under its run span, not as a
    new root. Goes through the real middleware and the threadpool a sync route runs in."""
    from anydocs import telemetry
    from anydocs.knowledge.digest import doc_id_for

    trace_id, parent_id = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    scripted(_receipts_turn(doc_id_for("relayr", "https://docs.relayr.example/receipts.md")))
    response = client.post(
        "/invoke",
        json={"message": "how long are receipts kept?", "corpora": ["relayr"]},
        headers={**TOKEN, "traceparent": f"00-{trace_id}-{parent_id}-01"},
    )
    assert response.status_code == 200
    assert response.json()["trace_id"] == trace_id

    turn = spans.find(telemetry.TURN)
    assert f"{turn.context.trace_id:032x}" == trace_id
    assert f"{turn.parent.span_id:016x}" == parent_id


def _serverless(monkeypatch, **env):
    from anydocs import config

    monkeypatch.setenv("ANYDOCS_FLUSH_SPANS", "true")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config.reset_settings_cache()


def test_serverless_startup_copies_the_bundle_without_a_lifespan(
    tmp_path, monkeypatch, relayr_corpus, data_dir
):
    """A host may skip the lifespan, so the app must be ready from `create_app()` alone."""
    from anydocs import bundle, config

    bundle.export(data_dir, tmp_path / "bundled", ["relayr"])
    _serverless(
        monkeypatch,
        ANYDOCS_DATA_DIR=str(tmp_path / "live"),
        ANYDOCS_BUNDLE_DIR=str(tmp_path / "bundled"),
    )
    try:
        client = TestClient(create_app())  # no `with`: the lifespan does not run
        assert client.get("/readyz").json() == {"status": "ready", "corpora": ["relayr"]}
    finally:
        config.reset_settings_cache()


def test_serverless_startup_refuses_a_default_credential(data_dir, monkeypatch, relayr_corpus):
    import pytest

    from anydocs import config

    _serverless(monkeypatch, ANYDOCS_API_TOKEN="dev-token")
    try:
        with pytest.raises(RuntimeError, match="real values"):
            create_app()
    finally:
        config.reset_settings_cache()


def test_serverless_startup_refuses_an_empty_store(tmp_path, monkeypatch):
    import pytest

    from anydocs import config

    _serverless(
        monkeypatch,
        ANYDOCS_DATA_DIR=str(tmp_path / "live"),
        ANYDOCS_BUNDLE_DIR=str(tmp_path / "missing"),
    )
    try:
        with pytest.raises(RuntimeError, match="anydocs bundle"):
            create_app()
    finally:
        config.reset_settings_cache()


def test_the_scheduled_turn_is_off_without_a_cron_secret(client, monkeypatch):
    from anydocs import config

    monkeypatch.delenv("CRON_SECRET", raising=False)
    config.reset_settings_cache()
    assert client.get("/cron/turn", headers={"Authorization": "Bearer x"}).status_code == 404


def test_the_scheduled_turn_runs_a_demo_question_as_the_scheduler(
    client, monkeypatch, relayr_corpus, scripted, spans
):
    from anydocs import config, server, telemetry
    from anydocs.knowledge.digest import doc_id_for

    monkeypatch.setenv("CRON_SECRET", "cron-secret")
    config.reset_settings_cache()
    monkeypatch.setattr(
        server,
        "_demo_questions",
        lambda: [
            {"message": "unused"},
            {"message": "how long are receipts kept?", "corpora": ["relayr"]},
        ],
    )
    scripted(_receipts_turn(doc_id_for("relayr", "https://docs.relayr.example/receipts.md")))

    # The API token is not the cron secret: revoking one must not open or break the other.
    assert client.get("/cron/turn", headers=TOKEN).status_code == 403

    response = client.get("/cron/turn?index=1", headers={"Authorization": "Bearer cron-secret"})
    assert response.status_code == 200
    assert "90 days" in response.json()["answer"]
    assert spans.find(telemetry.TURN).attributes[telemetry.Attr.USER_ID] == "scheduler"


def test_every_demo_question_names_only_a_bundled_corpus():
    """A question pinned to a corpus the deploy does not ship would fail on every tick."""
    import json

    from anydocs import server
    from anydocs.config import REPO_ROOT

    bundled = set(json.loads((REPO_ROOT / "bundled" / "corpora.json").read_text()))
    questions = server._demo_questions()
    assert len(questions) >= 10
    for question in questions:
        assert question["message"].strip()
        assert set(question.get("corpora") or []) <= bundled, question
