"""Policy is enforced in code. These tests are the proof."""

from __future__ import annotations

import time

import pytest

from anydocs import policy as policy_mod
from anydocs.policy import (
    PolicyDenied,
    UnsafeURL,
    default_policy,
    mint_resume_token,
    policy_with_overrides,
    verify_resume_token,
)


def test_modes_come_from_the_yaml():
    policy = default_policy()
    assert policy.mode("crawl") == "ask"
    assert policy.mode("write") == "auto"
    assert policy.mode("delete") == "ask"
    assert policy.mode("web_search") == "auto"


def test_an_unknown_policy_key_is_off_not_auto():
    """A tool gated by a key the YAML does not mention must not quietly run."""
    assert default_policy().mode("telepathy") == "off"


def test_overrides_merge_one_level_deep():
    policy = policy_with_overrides({"crawl": {"mode": "auto"}})
    assert policy.mode("crawl") == "auto"
    # Untouched keys in the same section survive the merge.
    assert policy.section("crawl")["respect_robots"] is True
    assert policy.mode("delete") == "ask"
    assert "override" in policy.name


def test_auto_approve_skips_the_human_step_in_ask_mode():
    policy = policy_with_overrides({"crawl": {"mode": "ask"}})
    policy.enforce("crawl", action="crawl_site", auto_approve=True)  # does not raise


@pytest.mark.parametrize(
    "url,fragment",
    [
        ("file:///etc/passwd", "not http"),
        ("http://localhost:8000/x", "deny_domains"),
        ("http://127.0.0.1/x", "deny_domains"),
        # Not in the deny list; caught one layer down by address resolution.
        ("http://[::1]/x", "non-public"),
        ("http://192.168.1.10/x", "non-public"),
        ("http://169.254.169.254/latest/meta-data/", "non-public"),
        ("http://10.0.0.5/", "non-public"),
    ],
)
def test_private_and_non_http_urls_are_refused(url, fragment):
    with pytest.raises(UnsafeURL) as excinfo:
        default_policy().check_domain(url)
    assert fragment in str(excinfo.value)


def test_a_public_hostname_resolving_to_loopback_is_still_refused(monkeypatch):
    """The deny list is a convenience; address resolution is the actual boundary."""
    monkeypatch.setattr(
        policy_mod.socket,
        "getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("127.0.0.1", 0))],
    )
    with pytest.raises(UnsafeURL, match="non-public"):
        default_policy().check_domain("https://totally-public.example/")


def test_allow_domains_excludes_everything_else(monkeypatch):
    policy = policy_with_overrides({"crawl": {"allow_domains": ["docs.relayr.example"]}})
    monkeypatch.setattr(
        policy_mod, "_refuse_private_network", lambda url, host: None
    )
    policy.check_domain("https://docs.relayr.example/retries")
    with pytest.raises(UnsafeURL, match="allow_domains"):
        policy.check_domain("https://docs.other.example/")


def test_resume_tokens_round_trip():
    token = mint_resume_token({"action": "crawl_site", "estimated_pages": 40})
    payload = verify_resume_token(token)
    assert payload["action"] == "crawl_site"
    assert payload["estimated_pages"] == 40


def test_a_tampered_resume_token_is_rejected():
    token = mint_resume_token({"action": "crawl_site", "estimated_pages": 40})
    body, sig = token.split(".", 1)
    with pytest.raises(PolicyDenied, match="signature"):
        verify_resume_token(f"{body}.{sig[:-4]}AAAA")


def test_a_forged_payload_cannot_be_signed_by_the_client():
    import base64
    import json

    forged = base64.urlsafe_b64encode(
        json.dumps({"action": "crawl_site", "issued_at": int(time.time())}).encode()
    ).decode().rstrip("=")
    with pytest.raises(PolicyDenied):
        verify_resume_token(f"{forged}.{forged}")


def test_an_expired_resume_token_is_rejected(monkeypatch):
    token = mint_resume_token({"action": "crawl_site"})
    later = time.time() + 7200
    monkeypatch.setattr(policy_mod.time, "time", lambda: later)
    with pytest.raises(PolicyDenied, match="expired"):
        verify_resume_token(token)


def test_a_malformed_token_is_rejected():
    with pytest.raises(PolicyDenied, match="malformed"):
        verify_resume_token("not-a-token")


def test_crawl_caps_come_from_policy():
    auto, hard = default_policy().crawl_caps()
    assert (auto, hard) == (50, 500)


@pytest.fixture
def ceiling(monkeypatch):
    """The env a public deployment sets: no crawl, no writes, no deletes."""
    from anydocs import config

    for key in ("CRAWL", "WRITE", "DELETE"):
        monkeypatch.setenv(f"ANYDOCS_{key}_MODE", "off")
    config.reset_settings_cache()
    yield
    config.reset_settings_cache()


def test_the_env_ceiling_lowers_the_yaml(ceiling):
    policy = default_policy()
    assert [policy.mode(k) for k in ("crawl", "write", "delete")] == ["off", "off", "off"]
    assert policy.mode("web_search") == "auto"


def test_a_request_cannot_raise_the_ceiling(ceiling):
    policy = policy_with_overrides({"crawl": {"mode": "auto"}, "write": {"mode": "auto"}})
    assert policy.mode("crawl") == "off"
    assert policy.mode("write") == "off"


def test_a_request_can_still_tighten_below_the_ceiling(monkeypatch):
    from anydocs import config

    monkeypatch.setenv("ANYDOCS_CRAWL_MODE", "ask")
    config.reset_settings_cache()
    try:
        assert policy_with_overrides({"crawl": {"mode": "auto"}}).mode("crawl") == "ask"
        assert policy_with_overrides({"crawl": {"mode": "off"}}).mode("crawl") == "off"
    finally:
        config.reset_settings_cache()


def test_a_bad_ceiling_fails_at_startup(monkeypatch):
    from anydocs import config

    monkeypatch.setenv("ANYDOCS_CRAWL_MODE", "never")
    config.reset_settings_cache()
    try:
        with pytest.raises(config.ConfigError, match="ANYDOCS_CRAWL_MODE"):
            config.get_settings()
    finally:
        config.reset_settings_cache()


def test_the_root_is_wherever_the_config_actually_is(tmp_path):
    """An installed (non-editable) package's parent is site-packages, which has no `config/`.
    The root must then come from the working directory, or startup dies before the bundle copy."""
    from anydocs.config import find_root

    site_packages, source = tmp_path / "site-packages", tmp_path / "project"
    site_packages.mkdir()
    (source / "config").mkdir(parents=True)
    (source / "config" / "models.yaml").write_text("roles: {}\n")

    assert find_root([site_packages, source]) == source
    assert find_root([source, site_packages]) == source
    # Neither holds config: keep the package parent, so the error names the expected path.
    assert find_root([site_packages, tmp_path / "elsewhere"]) == site_packages
