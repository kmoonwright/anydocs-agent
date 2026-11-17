"""Permission policy, enforced in code rather than in a prompt.

Four side-effect classes — crawl, write, delete, web_search — each with a mode of
`off | ask | auto`. `ask` suspends the turn and returns a signed resume token; the pause and the
decision are both spans, so the approval moment is visible in the trace.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import hmac
import ipaddress
import json
import socket
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from anydocs import telemetry
from anydocs.budget import Ledger
from anydocs.config import get_settings

RESUME_TOKEN_TTL_S = 3600


class PolicyDenied(PermissionError):
    """The policy forbids this action outright. Not retryable."""

    def __init__(self, action: str, reason: str) -> None:
        super().__init__(f"{action} denied: {reason}")
        self.action = action
        self.reason = reason


class NeedsApproval(Exception):
    """The policy requires a human decision. Carries everything the client needs to resume."""

    def __init__(self, action: str, payload: dict[str, Any], resume_token: str) -> None:
        super().__init__(f"{action} needs approval")
        self.action = action
        self.payload = payload
        self.resume_token = resume_token


class UnsafeURL(PolicyDenied):
    """Refused at the HTTP client layer: private network, loopback, or a non-http scheme."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__("fetch", f"{url}: {reason}")
        self.url = url


@dataclass(frozen=True)
class Policy:
    name: str
    raw: dict[str, Any]

    # -- modes ------------------------------------------------------------------------

    def mode(self, key: str) -> str:
        """`off`, `ask` or `auto` for a side-effect class. Unknown classes are `off`: a tool
        gated by a policy key the YAML does not mention must not quietly run."""
        return str((self.raw.get(key) or {}).get("mode", "off")).lower()

    def section(self, key: str) -> dict[str, Any]:
        return self.raw.get(key) or {}

    @property
    def budget(self) -> dict[str, Any]:
        return self.section("budget")

    def new_ledger(self) -> Ledger:
        b = self.budget
        return Ledger(
            max_tool_calls=int(b.get("max_tool_calls_per_turn", 24)),
            max_tokens=int(b.get("max_tokens_per_turn", 60_000)),
            max_wall_ms=int(b.get("max_wall_ms", 45_000)),
            daily_usd_cap=float(b.get("daily_usd_cap", 25.0)),
        )

    # -- the gate ---------------------------------------------------------------------

    def enforce(
        self,
        policy_key: str,
        *,
        action: str,
        payload: dict[str, Any] | None = None,
        auto_approve: bool = False,
    ) -> None:
        """Raise unless `action` may proceed.

        `auto_approve` lets a caller that already satisfied a quantitative cap (a crawl under
        `auto_approve_pages`, say) skip the human step in `ask` mode.
        """
        if policy_key == "none":
            return
        mode = self.mode(policy_key)
        payload = dict(payload or {})
        payload.setdefault("action", action)
        payload.setdefault("policy_key", policy_key)

        if mode == "auto" or (mode == "ask" and auto_approve):
            return
        if mode == "off":
            with telemetry.approval_span(
                action=action, mode=mode, pending=False, decision="denied"
            ):
                pass
            raise PolicyDenied(action, f"policy {self.name}: {policy_key}.mode is off")
        if mode == "ask":
            token = mint_resume_token({"action": action, "policy_key": policy_key, **payload})
            with telemetry.approval_span(
                action=action,
                mode=mode,
                pending=True,
                **{
                    telemetry.Attr.POLICY_ESTIMATE_PAGES: payload.get("estimated_pages", 0),
                    telemetry.Attr.POLICY_ESTIMATE_USD: payload.get("estimated_usd", 0.0),
                },
            ):
                pass
            raise NeedsApproval(action, payload, token)
        raise PolicyDenied(action, f"policy {self.name}: unknown mode {mode!r} for {policy_key}")

    # -- crawl specifics ---------------------------------------------------------------

    def crawl_caps(self) -> tuple[int, int]:
        c = self.section("crawl")
        return int(c.get("auto_approve_pages", 0)), int(c.get("max_pages_per_crawl", 500))

    @property
    def per_domain_rps(self) -> float:
        return float(self.section("crawl").get("per_domain_rps", 1))

    @property
    def respect_robots(self) -> bool:
        return bool(self.section("crawl").get("respect_robots", True))

    def check_domain(self, url: str) -> None:
        """Allow/deny lists plus the private-network refusal. Called by the HTTP client, so it
        applies to every fetch including redirects — not only to crawl roots."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise UnsafeURL(url, f"scheme {parsed.scheme!r} is not http(s)")
        host = (parsed.hostname or "").lower()
        if not host:
            raise UnsafeURL(url, "no host")

        crawl = self.section("crawl")
        for pattern in crawl.get("deny_domains") or []:
            if host == pattern or fnmatch.fnmatch(host, pattern):
                raise UnsafeURL(url, f"host matches deny_domains pattern {pattern!r}")
        allow = crawl.get("allow_domains") or []
        if allow and not any(host == p or fnmatch.fnmatch(host, p) for p in allow):
            raise UnsafeURL(url, "host is not in allow_domains")

        _refuse_private_network(url, host)


def _refuse_private_network(url: str, host: str) -> None:
    """Resolve and refuse loopback, link-local, private and reserved addresses.

    The deny list is a convenience; this is the actual boundary, so a public hostname that
    resolves to 127.0.0.1 is still refused.
    """
    try:
        addr = ipaddress.ip_address(host)
        addresses = [addr]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise UnsafeURL(url, f"host does not resolve ({exc})") from None
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue
    for ip in addresses:
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UnsafeURL(url, f"resolves to non-public address {ip}")


# ----------------------------------------------------------------------------------------
# Resume tokens
# ----------------------------------------------------------------------------------------


def mint_resume_token(payload: dict[str, Any]) -> str:
    """HMAC-signed, so a client cannot forge an approval for an action we never proposed."""
    body = dict(payload)
    body["issued_at"] = int(time.time())
    blob = json.dumps(body, sort_keys=True, default=str).encode()
    sig = hmac.new(get_settings().signing_key.encode(), blob, hashlib.sha256).digest()
    return (
        base64.urlsafe_b64encode(blob).decode().rstrip("=")
        + "."
        + base64.urlsafe_b64encode(sig).decode().rstrip("=")
    )


def verify_resume_token(token: str) -> dict[str, Any]:
    try:
        blob_b64, sig_b64 = token.split(".", 1)
        blob = base64.urlsafe_b64decode(blob_b64 + "=" * (-len(blob_b64) % 4))
        sig = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
    except (ValueError, TypeError) as exc:
        raise PolicyDenied("approve", "malformed resume token") from exc

    expected = hmac.new(get_settings().signing_key.encode(), blob, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise PolicyDenied("approve", "resume token signature does not verify")

    payload = json.loads(blob)
    age = int(time.time()) - int(payload.get("issued_at", 0))
    if age > RESUME_TOKEN_TTL_S:
        raise PolicyDenied("approve", f"resume token expired ({age}s old)")
    return payload


# ----------------------------------------------------------------------------------------
# Loading and per-request override
# ----------------------------------------------------------------------------------------


def default_policy() -> Policy:
    raw = _clamp(get_settings().policy)
    return Policy(name=str(raw.get("policy", "default")), raw=raw)


def _clamp(raw: dict[str, Any]) -> dict[str, Any]:
    """Lower each side-effect class to the process ceiling, if one is set.

    Applied after every merge, so a request can tighten a mode but never loosen it: a public
    deployment must not depend on the caller remembering not to ask for a crawl.
    """
    ceiling = get_settings().mode_ceiling
    if not ceiling:
        return raw
    rank = {"off": 0, "ask": 1, "auto": 2}
    clamped = dict(raw)
    for key, cap in ceiling.items():
        section = dict(clamped.get(key) or {})
        if rank.get(str(section.get("mode", "off")).lower(), 0) > rank[cap]:
            section["mode"] = cap
        clamped[key] = section
    return clamped


def policy_with_overrides(overrides: dict[str, Any] | None) -> Policy:
    """Merge a per-request policy override one level deep, so a request can flip
    `crawl.mode` without restating the whole file."""
    base = default_policy()
    if not overrides:
        return base
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in base.raw.items()}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return Policy(name=str(merged.get("policy", base.name)) + "+override", raw=_clamp(merged))
