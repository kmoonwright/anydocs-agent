"""Snapshots: open, write, close.

A snapshot is one directory. Copy it to object storage or hand it to a colleague as a folder and
it still answers questions, because everything an answer needs to be reproducible is inside it:
the index, the digests, the provenance and the manifest naming the embedding model.

Layout:

    data/
      corpora.json                       # corpus -> snapshots, default snapshot
      snapshots/<corpus>/<snapshot_id>/
        manifest.json                    # page count, embedding model + dim, crawl stats
        docs.jsonl                       # full Doc records, newest write wins
        provenance.jsonl                 # append-only; survives a tombstone
        links.jsonl                      # related-doc edges
        freshness.jsonl                  # re-check schedule
        changelog.jsonl                  # what changed, per doc, per refresh
        index.db                         # sqlite-vec index over chunks and digests
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anydocs.config import get_settings
from anydocs.knowledge.digest import Doc
from anydocs.knowledge.embeddings import get_embeddings

# Rows per embedding call when rebuilding an index. The embedding API is the slow part, and the
# provider batches internally at 96, so this only bounds memory.
INDEX_BATCH = 500


class StoreError(RuntimeError):
    """Something is wrong with a snapshot on disk."""


class EmbeddingMismatch(StoreError):
    """The snapshot was built with a different embedding model than the one configured.

    Loud on purpose: silently mixing vector spaces produces retrieval that looks like it works
    and is wrong.
    """


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_snapshot_id(content_hash: str = "") -> str:
    """Timestamp plus a distinguishing suffix, so ids sort chronologically and stay unique.

    The suffix is not decoration: without it two crawls started in the same second share an id
    and silently write into one directory, which looks exactly like duplicate detection working.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = content_hash[:8] if content_hash else uuid.uuid4().hex[:6]
    return f"{stamp}-{suffix}"


# ----------------------------------------------------------------------------------------
# Corpus registry
# ----------------------------------------------------------------------------------------


@dataclass
class CorpusEntry:
    name: str
    source: str = ""
    snapshots: list[str] = field(default_factory=list)
    default_snapshot: str = ""
    created_at: str = field(default_factory=_now)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "snapshots": self.snapshots,
            "default_snapshot": self.default_snapshot,
            "created_at": self.created_at,
        }


class CorpusRegistry:
    """data/corpora.json. Small enough to rewrite whole on every change."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().corpora_registry
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text() or "{}")

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def list(self) -> list[CorpusEntry]:
        return [CorpusEntry(**entry) for entry in self._read().values()]

    def get(self, name: str) -> CorpusEntry | None:
        raw = self._read().get(name)
        return CorpusEntry(**raw) if raw else None

    def upsert(self, entry: CorpusEntry) -> CorpusEntry:
        data = self._read()
        data[entry.name] = entry.to_json()
        self._write(data)
        return entry

    def add_snapshot(
        self, corpus: str, snapshot_id: str, *, make_default: bool = True
    ) -> CorpusEntry:
        entry = self.get(corpus) or CorpusEntry(name=corpus)
        if snapshot_id not in entry.snapshots:
            entry.snapshots.append(snapshot_id)
        if make_default or not entry.default_snapshot:
            entry.default_snapshot = snapshot_id
        return self.upsert(entry)

    def default_snapshot(self, corpus: str) -> str:
        entry = self.get(corpus)
        if not entry or not entry.default_snapshot:
            raise StoreError(f"Corpus {corpus!r} has no default snapshot")
        return entry.default_snapshot

    def active_corpora(self) -> list[str]:
        return [e.name for e in self.list() if e.default_snapshot]


# ----------------------------------------------------------------------------------------
# Snapshots
# ----------------------------------------------------------------------------------------


class Snapshot:
    """One snapshot directory, opened for reading or writing.

    Doc records live in `docs.jsonl` and are the source of truth; the Lance table is a
    derived index that `reindex()` can always rebuild from them.
    """

    def __init__(self, corpus: str, snapshot_id: str, root: Path) -> None:
        self.corpus = corpus
        self.snapshot_id = snapshot_id
        self.root = root
        self._docs: dict[str, Doc] | None = None

    # -- paths ------------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def docs_path(self) -> Path:
        return self.root / "docs.jsonl"

    @property
    def provenance_path(self) -> Path:
        return self.root / "provenance.jsonl"

    @property
    def links_path(self) -> Path:
        return self.root / "links.jsonl"

    @property
    def freshness_path(self) -> Path:
        return self.root / "freshness.jsonl"

    @property
    def changelog_path(self) -> Path:
        return self.root / "changelog.jsonl"

    @property
    def index_path(self) -> Path:
        return self.root / "index.db"

    # -- manifest ---------------------------------------------------------------------

    def manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text())

    def write_manifest(self, **updates: Any) -> dict[str, Any]:
        data = self.manifest()
        embeddings = get_embeddings()
        data.update(
            {
                "corpus": self.corpus,
                "snapshot_id": self.snapshot_id,
                "embedding_model": embeddings.model,
                "embedding_dim": embeddings.dim,
                "updated_at": _now(),
            }
        )
        data.setdefault("created_at", _now())
        data.update(updates)
        data["page_count"] = len(self.docs())
        self.manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        return data

    def check_embedding_compatibility(self) -> None:
        manifest = self.manifest()
        if not manifest:
            return
        configured = get_embeddings()
        stored_model = manifest.get("embedding_model")
        stored_dim = manifest.get("embedding_dim")
        if stored_model and stored_model != configured.model:
            raise EmbeddingMismatch(
                f"Snapshot {self.corpus}/{self.snapshot_id} was embedded with "
                f"{stored_model!r} (dim {stored_dim}), but the configured model is "
                f"{configured.model!r} (dim {configured.dim}). Re-index the snapshot or point "
                f"config/models.yaml back at {stored_model!r}."
            )

    # -- docs -------------------------------------------------------------------------

    def docs(self) -> dict[str, Doc]:
        """All docs, newest write per doc_id winning. Tombstoned docs are included here and
        filtered at retrieval, because provenance must survive a `forget`."""
        if self._docs is None:
            docs: dict[str, Doc] = {}
            if self.docs_path.exists():
                with self.docs_path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        doc = Doc.model_validate_json(line)
                        docs[doc.doc_id] = doc
            self._docs = docs
        return self._docs

    def refresh(self) -> None:
        """Drop the cached docs and index handle.

        `save_note` opens its own Snapshot for the same directory, so a long-lived instance can
        hold a stale view. Anything that reads after someone else wrote calls this first.
        """
        self._docs = None

    def get_doc(self, doc_id: str) -> Doc | None:
        return self.docs().get(doc_id)

    def find_by_url(self, url: str) -> Doc | None:
        for doc in self.docs().values():
            if url in (doc.url, doc.canonical_url):
                return doc
        return None

    def live_docs(self) -> list[Doc]:
        return [d for d in self.docs().values() if not d.tombstoned]

    def write_doc(self, doc: Doc) -> None:
        """Append-only: the last line for a doc_id wins, so history stays on disk."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self.docs_path.open("a") as fh:
            fh.write(doc.model_dump_json() + "\n")
        self.docs()[doc.doc_id] = doc

    # -- jsonl sidecars ----------------------------------------------------------------

    def append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out = []
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
        return out

    # -- index ------------------------------------------------------------------------

    def index_rows(self) -> list[dict[str, Any]]:
        """Chunks and digests as one row set, distinguished by `kind` so retrieval can weight
        them differently."""
        return [row for doc in self.live_docs() for row in self.doc_rows(doc)]

    def doc_rows(self, doc: Doc) -> list[dict[str, Any]]:
        """The index rows for one doc: its chunks plus its digest."""
        rows: list[dict[str, Any]] = []
        for chunk in doc.chunks:
            rows.append(
                {
                    "node_id": chunk.chunk_id,
                    "doc_id": doc.doc_id,
                    "corpus": doc.corpus,
                    "kind": "chunk",
                    "url": doc.canonical_url or doc.url,
                    "title": doc.title,
                    "heading": chunk.heading,
                    "versions": " ".join(doc.version_markers),
                    "text": chunk.text,
                    "position": chunk.position,
                }
            )
        rows.append(
            {
                "node_id": f"{doc.doc_id}#digest",
                "doc_id": doc.doc_id,
                "corpus": doc.corpus,
                "kind": "digest",
                "url": doc.canonical_url or doc.url,
                "title": doc.title,
                "heading": doc.heading,
                "versions": " ".join(doc.version_markers),
                "text": doc.digest.searchable_text(),
                "position": -1,
            }
        )
        return rows

    # -- index ------------------------------------------------------------------------
    #
    # One SQLite file per snapshot: a plain `nodes` table for the rows retrieval reads, and a
    # sqlite-vec virtual table for the dense side. The design doc named this as the fallback
    # "if dependency weight matters" — it does, because LanceDB and pyarrow together are 290 MB
    # against a 250 MB serverless function limit, and this is 1 MB.
    #
    # The sparse side stays `rank_bm25` in process: sqlite-vec covers vectors only, and BM25 over
    # a snapshot-sized corpus is cheap enough to build per search.

    def _connect(self) -> Any:
        """A connection with the vector extension loaded."""
        import sqlite3

        import sqlite_vec

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.index_path)
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.execute(
            """CREATE TABLE IF NOT EXISTS nodes (
                   node_id  TEXT PRIMARY KEY,
                   doc_id   TEXT NOT NULL,
                   corpus   TEXT NOT NULL,
                   kind     TEXT NOT NULL,
                   url      TEXT,
                   title    TEXT,
                   heading  TEXT,
                   versions TEXT,
                   text     TEXT NOT NULL,
                   position INTEGER
               )"""
        )
        db.execute("CREATE INDEX IF NOT EXISTS nodes_by_doc ON nodes (doc_id)")
        db.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
                    node_id TEXT PRIMARY KEY,
                    embedding float[{get_embeddings().dim}]
                )"""
        )
        return db

    def _write_rows(self, db: Any, rows: list[dict[str, Any]]) -> None:
        """Embed and insert one batch of index rows."""
        import sqlite_vec

        vectors = get_embeddings().embed([r["text"] for r in rows])
        db.executemany(
            """INSERT OR REPLACE INTO nodes
                   (node_id, doc_id, corpus, kind, url, title, heading, versions, text, position)
               VALUES
                   (:node_id, :doc_id, :corpus, :kind, :url, :title, :heading, :versions,
                    :text, :position)""",
            rows,
        )
        db.executemany(
            "INSERT OR REPLACE INTO vectors (node_id, embedding) VALUES (?, ?)",
            [
                (r["node_id"], sqlite_vec.serialize_float32(v))
                for r, v in zip(rows, vectors, strict=True)
            ],
        )

    def index_doc(self, doc: Doc) -> int:
        """Add or replace one doc's rows in the index.

        `save_note` calls this so a snapshot is queryable as soon as it is written. Without it
        the agent-driven path would produce a snapshot whose docs exist and whose index is
        empty, which looks like a retrieval bug rather than a missing step.
        """
        rows = self.doc_rows(doc)
        if not rows:
            return 0
        db = self._connect()
        try:
            # Replace rather than append: a re-digested page must not leave its old chunks
            # behind to be retrieved alongside the new ones.
            self._delete_doc(db, doc.doc_id)
            self._write_rows(db, rows)
            db.commit()
        finally:
            db.close()
        return len(rows)

    @staticmethod
    def _delete_doc(db: Any, doc_id: str) -> int:
        node_ids = [
            r[0] for r in db.execute("SELECT node_id FROM nodes WHERE doc_id = ?", (doc_id,))
        ]
        if node_ids:
            marks = ",".join("?" * len(node_ids))
            db.execute(f"DELETE FROM vectors WHERE node_id IN ({marks})", node_ids)  # noqa: S608
            db.execute("DELETE FROM nodes WHERE doc_id = ?", (doc_id,))
        return len(node_ids)

    def unindex_doc(self, doc_id: str) -> int:
        """Drop one doc's rows from the index without touching its record on disk.

        What `forget` needs: the doc stays in `docs.jsonl` as a tombstone so provenance still
        resolves, while retrieval stops returning it.
        """
        if not self.index_path.exists():
            return 0
        db = self._connect()
        try:
            removed = self._delete_doc(db, doc_id)
            db.commit()
        finally:
            db.close()
        return removed

    def reindex(self) -> int:
        """Rebuild the index from docs.jsonl. Safe to call any time."""
        self.refresh()
        rows = self.index_rows()
        self.index_path.unlink(missing_ok=True)
        if not rows:
            return 0
        db = self._connect()
        try:
            # Embedding is the slow part, so batch it rather than going doc by doc.
            for start in range(0, len(rows), INDEX_BATCH):
                self._write_rows(db, rows[start : start + INDEX_BATCH])
            db.commit()
        finally:
            db.close()
        self.write_manifest(indexed_at=_now(), node_count=len(rows))
        return len(rows)

    def search_vectors(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """Nearest neighbours by embedding: `(node_id, distance)`, smaller distance first.

        Replaces exposing a table handle, so the store owns how vectors are queried and
        retrieval only has to know what a neighbour is.
        """
        if not self.index_path.exists():
            return []
        import sqlite_vec

        db = self._connect()
        try:
            return [
                (row["node_id"], float(row["distance"]))
                for row in db.execute(
                    "SELECT node_id, distance FROM vectors "
                    "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (sqlite_vec.serialize_float32(vector), k),
                )
            ]
        finally:
            db.close()

    def all_nodes(self) -> list[dict[str, Any]]:
        """Every indexed row as plain dicts. BM25 runs in-process over these."""
        if not self.index_path.exists():
            return []
        db = self._connect()
        try:
            return [dict(row) for row in db.execute("SELECT * FROM nodes")]
        finally:
            db.close()

    def stats(self) -> dict[str, Any]:
        manifest = self.manifest()
        return {
            "snapshot_id": self.snapshot_id,
            "corpus": self.corpus,
            "page_count": len(self.live_docs()),
            "node_count": manifest.get("node_count", 0),
            "embedding_model": manifest.get("embedding_model", ""),
            "created_at": manifest.get("created_at", ""),
            "updated_at": manifest.get("updated_at", ""),
            "crawl": manifest.get("crawl", {}),
        }


# ----------------------------------------------------------------------------------------
# Opening snapshots
# ----------------------------------------------------------------------------------------


def snapshot_root(corpus: str, snapshot_id: str) -> Path:
    return get_settings().snapshots_dir / corpus / snapshot_id


def create_snapshot(corpus: str, *, snapshot_id: str | None = None, source: str = "") -> Snapshot:
    sid = snapshot_id or new_snapshot_id()
    root = snapshot_root(corpus, sid)
    root.mkdir(parents=True, exist_ok=True)
    snap = Snapshot(corpus, sid, root)
    snap.write_manifest(source=source)
    registry = CorpusRegistry()
    entry = registry.get(corpus) or CorpusEntry(name=corpus, source=source)
    if source and not entry.source:
        entry.source = source
    registry.upsert(entry)
    return snap


def open_snapshot(corpus: str, snapshot_id: str | None = None) -> Snapshot:
    """Open a snapshot for reading. Verifies the embedding model before anything reads it."""
    sid = snapshot_id or CorpusRegistry().default_snapshot(corpus)
    root = snapshot_root(corpus, sid)
    if not root.exists():
        raise StoreError(f"No snapshot at {root}")
    snap = Snapshot(corpus, sid, root)
    snap.check_embedding_compatibility()
    return snap


def resolve_snapshots(corpus: str = "", snapshot_id: str = "") -> list[Snapshot]:
    """The snapshots a tool should look in, from its `corpus` and `snapshot_id` arguments.

    Both are optional on most tools, which gives three cases: a named snapshot of a named
    corpus, every snapshot of one corpus, or every active corpus. Every tool resolved this the
    same way, so it is resolved here once.
    """
    if corpus and snapshot_id:
        return [open_snapshot(corpus, snapshot_id)]
    return open_active([corpus] if corpus else None)


def find_doc(
    corpus: str = "", snapshot_id: str = "", *, doc_id: str = "", url: str = ""
) -> tuple[Doc, Snapshot]:
    """Locate one doc and the snapshot holding it, by id or by URL.

    Raises `StoreError` rather than returning None: every caller wanted the error, and three of
    them were spelling out the same loop to produce it.
    """
    if not doc_id and not url:
        raise StoreError("find_doc needs either a doc_id or a url")
    for snapshot in resolve_snapshots(corpus, snapshot_id):
        doc = snapshot.get_doc(doc_id) if doc_id else snapshot.find_by_url(url)
        if doc is not None:
            return doc, snapshot
    raise StoreError(f"No doc matching {doc_id or url!r} in the active snapshots")


def publish_snapshot(corpus: str, snapshot_id: str) -> CorpusEntry:
    """Make a snapshot the corpus default, which is what retrieval reads."""
    return CorpusRegistry().add_snapshot(corpus, snapshot_id, make_default=True)


def open_active(corpora: list[str] | None = None) -> list[Snapshot]:
    """Default snapshots for the named corpora, or for every corpus that has one."""
    registry = CorpusRegistry()
    names = corpora or registry.active_corpora()
    snapshots = []
    for name in names:
        entry = registry.get(name)
        if not entry or not entry.default_snapshot:
            continue
        snapshots.append(open_snapshot(name, entry.default_snapshot))
    return snapshots
