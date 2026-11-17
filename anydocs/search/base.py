"""One interface over search APIs.

Result quality for documentation sites varies by vendor, so the provider stays swappable and
its name is recorded on the span — an eval can then tell a bad answer caused by bad search from
one caused by bad reading.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    domain: str = ""
    authority_hint: float = Field(
        default=0.0,
        description="0-1 guess at how authoritative the domain is for documentation. Vendor "
        "score where one exists, heuristic otherwise.",
    )
    published: str = ""


class SearchProvider(Protocol):
    name: str

    def search(
        self, query: str, *, site: str = "", recency: str = "", k: int = 8
    ) -> list[SearchResult]:
        ...


# Hosts that publish their own product's documentation rank above write-ups about it.
_DOC_HOST_HINTS = (
    "docs.",
    "developer.",
    "developers.",
    "/docs/",
    "readthedocs",
    "reference.",
    "api.",
)
_LOW_AUTHORITY = (
    "stackoverflow.com",
    "reddit.com",
    "medium.com",
    "dev.to",
    "quora.com",
    "github.io/blog",
)


def authority_hint(url: str, title: str = "") -> float:
    """Heuristic standing in for a vendor authority score.

    The Scout judges sources with a model; this only orders the candidate list it sees.
    """
    lowered = url.lower()
    score = 0.4
    if any(hint in lowered for hint in _DOC_HOST_HINTS):
        score += 0.35
    if any(bad in lowered for bad in _LOW_AUTHORITY):
        score -= 0.3
    if "blog" in lowered or "blog" in title.lower():
        score -= 0.15
    if lowered.rstrip("/").endswith(("/docs", "/documentation", "/reference")):
        score += 0.1
    return max(0.0, min(1.0, round(score, 2)))


def domain_of(url: str) -> str:
    from urllib.parse import urlparse

    return (urlparse(url).hostname or "").lower()


_provider: SearchProvider | None = None


def get_search_provider() -> SearchProvider:
    global _provider
    if _provider is not None:
        return _provider
    from anydocs.config import get_settings

    name = get_settings().search_provider
    if name == "tavily":
        from anydocs.search.tavily import TavilyProvider

        _provider = TavilyProvider()
    elif name == "fixture":
        from anydocs.search.fixture import FixtureProvider

        _provider = FixtureProvider()
    else:
        raise ValueError(
            f"Unknown SEARCH_PROVIDER={name!r}. Wire it in anydocs/search/ and add it here."
        )
    return _provider

