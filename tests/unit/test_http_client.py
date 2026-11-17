"""The HTTP client is the boundary: robots, rate limits and private-network refusal."""

from __future__ import annotations

import httpx
import pytest
import respx

from anydocs import policy as policy_mod
from anydocs.policy import UnsafeURL, default_policy, policy_with_overrides
from anydocs.tools import http_client

# Captured before any patching, so a test that needs the real check can put it back.
REAL_REFUSE = policy_mod._refuse_private_network


@pytest.fixture(autouse=True)
def _resolvable(monkeypatch):
    """The fixture hosts are not real. Address resolution is tested in test_policy.py."""
    monkeypatch.setattr(policy_mod, "_refuse_private_network", lambda url, host: None)
    monkeypatch.setattr(http_client.time, "sleep", lambda _s: None)
    http_client.reset_robots_cache()


ROBOTS_ALLOW_ALL = "User-agent: *\nAllow: /\n"
ROBOTS_DENY_DOCS = "User-agent: *\nDisallow: /private/\n"


@respx.mock
def test_a_page_is_fetched_and_the_robots_decision_recorded():
    respx.get("https://docs.relayr.example/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_ALLOW_ALL)
    )
    respx.get("https://docs.relayr.example/retries").mock(
        return_value=httpx.Response(200, html="<html><title>Retries</title><body>x</body></html>")
    )
    page = http_client.fetch("https://docs.relayr.example/retries")
    assert page.status == 200
    assert page.robots_decision == "allowed"


@respx.mock
def test_a_disallowed_path_raises_rather_than_being_fetched():
    respx.get("https://docs.relayr.example/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_DENY_DOCS)
    )
    route = respx.get("https://docs.relayr.example/private/secrets").mock(
        return_value=httpx.Response(200, text="should never be read")
    )
    with pytest.raises(http_client.RobotsBlocked):
        http_client.fetch("https://docs.relayr.example/private/secrets")
    assert not route.called


@respx.mock
def test_a_missing_robots_file_is_permission_by_omission():
    respx.get("https://docs.relayr.example/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://docs.relayr.example/x").mock(return_value=httpx.Response(200, text="ok"))
    assert http_client.fetch("https://docs.relayr.example/x").robots_decision == "allowed"


@respx.mock
def test_an_unreachable_robots_file_is_unknown_not_allowed():
    respx.get("https://docs.relayr.example/robots.txt").mock(
        side_effect=httpx.ConnectError("boom")
    )
    respx.get("https://docs.relayr.example/x").mock(return_value=httpx.Response(200, text="ok"))
    assert http_client.fetch("https://docs.relayr.example/x").robots_decision == "unknown"


@respx.mock
def test_robots_is_not_consulted_when_policy_turns_it_off():
    route = respx.get("https://docs.relayr.example/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_DENY_DOCS)
    )
    respx.get("https://docs.relayr.example/private/x").mock(
        return_value=httpx.Response(200, text="ok")
    )
    policy = policy_with_overrides({"crawl": {"respect_robots": False}})
    page = http_client.fetch("https://docs.relayr.example/private/x", policy=policy)
    assert page.robots_decision == "not_checked"
    assert not route.called


@respx.mock
def test_a_redirect_to_a_private_address_is_refused(monkeypatch):
    """A redirect is a new URL, so it gets the same checks the original did."""
    monkeypatch.setattr(policy_mod, "_refuse_private_network", REAL_REFUSE)
    respx.get("https://docs.relayr.example/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_ALLOW_ALL)
    )
    respx.get("https://docs.relayr.example/open").mock(
        return_value=httpx.Response(302, headers={"location": "http://169.254.169.254/meta-data"})
    )
    with pytest.raises(UnsafeURL):
        http_client.fetch("https://docs.relayr.example/open")


@respx.mock
def test_a_redirect_chain_terminates():
    respx.get("https://docs.relayr.example/robots.txt").mock(
        return_value=httpx.Response(200, text=ROBOTS_ALLOW_ALL)
    )
    respx.get(url__startswith="https://docs.relayr.example/loop").mock(
        return_value=httpx.Response(302, headers={"location": "/loop"})
    )
    with pytest.raises(UnsafeURL, match="too many redirects"):
        http_client.fetch("https://docs.relayr.example/loop")


def test_the_rate_limiter_is_shared_per_domain(monkeypatch):
    """A crawl and a one-off fetch to the same host share one budget."""
    slept: list[float] = []
    monkeypatch.setattr(http_client.time, "sleep", slept.append)
    limiter = http_client._RateLimiter()
    limiter.wait("docs.example", 1.0)
    limiter.wait("docs.example", 1.0)
    limiter.wait("other.example", 1.0)
    assert len(slept) == 1 and slept[0] == pytest.approx(1.0, abs=0.05)


def test_zero_rps_disables_the_limiter(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(http_client.time, "sleep", slept.append)
    http_client._RateLimiter().wait("docs.example", 0)
    assert slept == []


def test_policy_rps_is_what_the_client_uses():
    assert default_policy().per_domain_rps == 1


def test_the_crawler_only_follows_documentation_pages():
    """Both halves were found by crawling arize.com: `.md` twins of every page produced eleven
    "no readable content" failures, and favicons linked from the head wasted 23 requests."""
    from anydocs.tools.page_text import same_site_links

    html = """
    <a href="/docs/overview">Overview</a>
    <a href="/docs/overview.md">Overview (markdown)</a>
    <a href="/docs/exporter?v=2">Exporter</a>
    <a href="/llms.txt">Agent index</a>
    <a href="/docs/guide.pdf">Guide</a>
    <a href="/assets/favicon.ico">icon</a>
    <a href="/assets/apple-touch-icon.PNG">icon</a>
    <a href="/assets/style.css">style</a>
    <a href="/fonts/inter.woff2">font</a>
    """
    links = same_site_links(html, "https://docs.example.com/docs/overview")
    paths = [link.split("docs.example.com")[1] for link in links]
    assert paths == ["/docs/overview", "/docs/exporter?v=2"]
