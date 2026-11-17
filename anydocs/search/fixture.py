"""Replays recorded search results from tests/fixtures/search/.

Keeps the unit suite key-free and makes the learn-then-answer loop reproducible: the same
question returns the same candidates every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from anydocs.config import REPO_ROOT
from anydocs.search.base import SearchResult, authority_hint, domain_of

DEFAULT_DIR = REPO_ROOT / "tests" / "fixtures" / "search"


class FixtureProvider:
    name = "fixture"

    def __init__(self, directory: Path | None = None) -> None:
        env_dir = os.getenv("ANYDOCS_SEARCH_FIXTURES")
        self.directory = directory or (Path(env_dir) if env_dir else DEFAULT_DIR)

    def search(
        self, query: str, *, site: str = "", recency: str = "", k: int = 8
    ) -> list[SearchResult]:
        payload = self._load(query, site)
        results = []
        for item in payload[:k]:
            url = item.get("url", "")
            results.append(
                SearchResult(
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    domain=item.get("domain") or domain_of(url),
                    authority_hint=float(
                        item.get("authority_hint", authority_hint(url, item.get("title", "")))
                    ),
                    published=item.get("published", ""),
                )
            )
        return results

    def _load(self, query: str, site: str) -> list[dict]:
        """Exact slug match first, then the keyword index, then empty.

        Empty is a legitimate answer — "the web had nothing" is a case the agent must handle.
        """
        slug = _slug(f"{query} {site}".strip())
        exact = self.directory / f"{slug}.json"
        if exact.exists():
            return json.loads(exact.read_text())

        index = self.directory / "index.json"
        if index.exists():
            mapping = json.loads(index.read_text())
            lowered = query.lower()
            for keyword, filename in mapping.items():
                if keyword.lower() in lowered:
                    candidate = self.directory / filename
                    if candidate.exists():
                        return json.loads(candidate.read_text())
        return []


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(cleaned) <= 60:
        return cleaned
    return f"{cleaned[:48]}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"
