"""Tavily. Built for agent retrieval, so its snippets are usable without a second fetch."""

from __future__ import annotations

import os

import httpx

from anydocs.search.base import SearchResult, authority_hint, domain_of

ENDPOINT = "https://api.tavily.com/search"
TIMEOUT_S = 20.0


class TavilyProvider:
    name = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")

    def search(
        self, query: str, *, site: str = "", recency: str = "", k: int = 8
    ) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Set it, or run with SEARCH_PROVIDER=fixture."
            )
        payload: dict[str, object] = {
            "api_key": self.api_key,
            "query": f"{query} site:{site}" if site else query,
            "max_results": k,
            "search_depth": "advanced",
            "include_answer": False,
        }
        if recency:
            payload["time_range"] = recency
        if site:
            payload["include_domains"] = [site]

        with httpx.Client(timeout=TIMEOUT_S) as client:
            response = client.post(ENDPOINT, json=payload)
            response.raise_for_status()
            data = response.json()

        results: list[SearchResult] = []
        for item in data.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            # Tavily's own score is relevance, not authority; blend it with the doc-host hint.
            vendor_score = float(item.get("score") or 0.0)
            hint = authority_hint(url, title)
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=(item.get("content") or "")[:600],
                    domain=domain_of(url),
                    authority_hint=round(min(1.0, 0.5 * vendor_score + 0.5 * hint), 2),
                    published=item.get("published_date") or "",
                )
            )
        return results
