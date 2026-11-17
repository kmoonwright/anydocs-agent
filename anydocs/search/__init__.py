"""Web search providers, behind one interface so the provider is a config change."""

from anydocs.search.base import SearchProvider, SearchResult, get_search_provider

__all__ = ["SearchProvider", "SearchResult", "get_search_provider"]
