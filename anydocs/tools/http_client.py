"""The only way this agent talks to the public web.

Three guarantees, all enforced here rather than in a prompt or a deny list:
robots.txt is honored, each domain is rate limited, and private-network or non-http URLs are
refused — including after a redirect.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from anydocs.policy import Policy, UnsafeURL, default_policy

USER_AGENT = "anydocs-agent/0.1 (+https://github.com/anydocs/anydocs-agent)"
TIMEOUT_S = 20.0
MAX_BYTES = 5_000_000


@dataclass(frozen=True)
class FetchedPage:
    url: str
    final_url: str
    status: int
    content_type: str
    last_modified: str
    html: str
    robots_decision: str  # allowed | disallowed | unknown | not_checked


class RobotsBlocked(PermissionError):
    def __init__(self, url: str) -> None:
        super().__init__(f"robots.txt disallows {url}")
        self.url = url


class _RateLimiter:
    """One token bucket per domain, shared process-wide.

    A crawl and a one-off fetch to the same host must not each get their own budget, so the
    limiter is module-level rather than per-client.
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str, rps: float) -> None:
        if rps <= 0:
            return
        interval = 1.0 / rps
        with self._lock:
            now = time.monotonic()
            earliest = self._last.get(host, 0.0) + interval
            delay = max(0.0, earliest - now)
            self._last[host] = max(now, earliest)
        if delay:
            time.sleep(delay)


_limiter = _RateLimiter()
_robots_cache: dict[str, RobotFileParser | None] = {}
_robots_lock = threading.Lock()


def _robots_for(url: str, client: httpx.Client) -> RobotFileParser | None:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    with _robots_lock:
        if origin in _robots_cache:
            return _robots_cache[origin]
    parser: RobotFileParser | None = RobotFileParser()
    try:
        response = client.get(urljoin(origin, "/robots.txt"), timeout=10.0)
        if response.status_code >= 400:
            # No robots.txt is permission by omission, which is the convention.
            parser.parse([])
        else:
            parser.parse(response.text.splitlines())
    except httpx.HTTPError:
        parser = None  # unknown, not permission
    with _robots_lock:
        _robots_cache[origin] = parser
    return parser


def reset_robots_cache() -> None:
    with _robots_lock:
        _robots_cache.clear()




def fetch(url: str, *, policy: Policy | None = None, method: str = "GET") -> FetchedPage:
    """Fetch one URL under the policy. Raises UnsafeURL or RobotsBlocked rather than returning
    a page the agent was not allowed to see."""
    pol = policy or default_policy()
    pol.check_domain(url)

    host = urlparse(url).hostname or ""
    _limiter.wait(host, pol.per_domain_rps)

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=False,
        timeout=TIMEOUT_S,
    ) as client:
        decision = "not_checked"
        if pol.respect_robots:
            parser = _robots_for(url, client)
            if parser is None:
                decision = "unknown"
            elif parser.can_fetch(USER_AGENT, url):
                decision = "allowed"
            else:
                raise RobotsBlocked(url)

        current = url
        response = None
        for _ in range(5):
            response = client.request(method, current)
            if response.status_code not in (301, 302, 303, 307, 308):
                break
            location = response.headers.get("location")
            if not location:
                break
            current = urljoin(current, location)
            # A redirect is a new URL, so it gets the same checks the original did.
            pol.check_domain(current)
            _limiter.wait(urlparse(current).hostname or "", pol.per_domain_rps)
        else:
            raise UnsafeURL(url, "too many redirects")

        assert response is not None
        content_type = response.headers.get("content-type", "")
        body = response.text if len(response.content) <= MAX_BYTES else response.text[:MAX_BYTES]
        return FetchedPage(
            url=url,
            final_url=current,
            status=response.status_code,
            content_type=content_type,
            last_modified=response.headers.get("last-modified", ""),
            html=body,
            robots_decision=decision,
        )
