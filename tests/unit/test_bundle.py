"""The corpus a serverless deploy ships with, and the copy that makes it writable."""

from __future__ import annotations

import json

import pytest

from anydocs import bundle, config


def test_a_bundle_lands_in_an_empty_data_dir_and_opens(
    tmp_path, monkeypatch, relayr_corpus, data_dir
):
    from anydocs.knowledge.store import open_snapshot

    bundle.export(data_dir, tmp_path / "bundled", ["relayr"])
    live = tmp_path / "live"
    assert bundle.materialize(tmp_path / "bundled", live) is True

    monkeypatch.setenv("ANYDOCS_DATA_DIR", str(live))
    config.reset_settings_cache()
    assert open_snapshot("relayr").stats()["page_count"] > 0


def test_materialize_never_overwrites_a_working_store(tmp_path, relayr_corpus, data_dir):
    bundle.export(data_dir, tmp_path / "bundled", ["relayr"])
    before = (data_dir / "corpora.json").read_text()
    assert bundle.materialize(tmp_path / "bundled", data_dir) is False
    assert (data_dir / "corpora.json").read_text() == before


def test_export_ships_only_the_named_corpora(tmp_path, relayr_corpus, data_dir):
    registry = json.loads((data_dir / "corpora.json").read_text())
    registry["other"] = dict(registry["relayr"], name="other")
    (data_dir / "corpora.json").write_text(json.dumps(registry))

    bundle.export(data_dir, tmp_path / "bundled", ["relayr"])
    shipped = json.loads((tmp_path / "bundled" / "corpora.json").read_text())
    assert list(shipped) == ["relayr"]
    assert not (tmp_path / "bundled" / "spend.json").exists()


def test_export_refuses_a_corpus_that_cites_local_files(tmp_path, data_dir, monkeypatch):
    """A folder ingested without --base-url cites file:// URLs naming the build machine."""
    monkeypatch.setenv("ANYDOCS_RERANKER", "noop")
    from anydocs.ingest import ingest_folder
    from tests.conftest import FIXTURES

    ingest_folder(FIXTURES / "corpus", "local", digest=False, publish=True)
    with pytest.raises(bundle.BundleError, match="file://"):
        bundle.export(data_dir, tmp_path / "bundled", ["local"])
