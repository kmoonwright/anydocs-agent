"""The corpus a serverless deploy ships with, and how it reaches a writable directory.

A serverless function's own files are read-only, and SQLite writes a journal even to read, so
the deployed knowledge store cannot be the deployed files. `export` builds `bundled/` from a
local store; `materialize` copies it onto the data directory when a process starts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REGISTRY = "corpora.json"
SNAPSHOTS = "snapshots"


class BundleError(RuntimeError):
    pass


def export(data_dir: Path, out_dir: Path, corpora: list[str]) -> list[str]:
    """Copy the default snapshot of each named corpus, and a registry naming only those.

    Spend, profiles and pending turns are per-deployment state, not corpus, so they stay behind.
    A corpus ingested from a local folder cites `file://` URLs that name the machine it was built
    on, and is refused rather than shipped.
    """
    registry = json.loads((data_dir / REGISTRY).read_text())
    missing = [name for name in corpora if not registry.get(name, {}).get("default_snapshot")]
    if missing:
        raise BundleError(f"No published snapshot for: {', '.join(missing)}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    entries = {}
    for name in corpora:
        entry = dict(registry[name])
        sid = entry["default_snapshot"]
        source = data_dir / SNAPSHOTS / name / sid
        _refuse_local_urls(name, source)
        shutil.copytree(source, out_dir / SNAPSHOTS / name / sid)
        entry["snapshots"] = [sid]
        entries[name] = entry
    (out_dir / REGISTRY).write_text(json.dumps(entries, indent=2, sort_keys=True))
    return corpora


def materialize(bundle_dir: Path, data_dir: Path) -> bool:
    """Copy the bundle onto `data_dir` unless it already holds a store. True if it copied.

    The registry is written last, so a copy that dies part-way never looks like a finished store
    to the next start.
    """
    if (data_dir / REGISTRY).exists() or not (bundle_dir / REGISTRY).exists():
        return False
    try:
        shutil.copytree(bundle_dir / SNAPSHOTS, data_dir / SNAPSHOTS, dirs_exist_ok=True)
        shutil.copy2(bundle_dir / REGISTRY, data_dir / REGISTRY)
    except OSError as exc:
        raise BundleError(f"Could not copy {bundle_dir} to {data_dir}: {exc}") from exc
    return True


def _refuse_local_urls(corpus: str, snapshot: Path) -> None:
    for name in ("docs.jsonl", "provenance.jsonl"):
        path = snapshot / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line) if line.strip() else {}
            for key in ("url", "canonical_url"):
                if str(record.get(key, "")).startswith("file://"):
                    raise BundleError(
                        f"{corpus} cites {record[key]!r}; a local-folder corpus names the machine "
                        "it was built on. Re-ingest it with --base-url, or leave it out."
                    )
