"""Role -> LangChain chat model.

`config/models.yaml` still maps a role to a provider and a model, and the role is still what lands
on the span, so experiments compare roles rather than model strings. What changed is who makes the
call: LangChain's provider packages replace the two SDK adapters this module used to hold, so the
`langchain` instrumentor emits the provider-level spans and we stop maintaining that part.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Any

from langchain_core.language_models import BaseChatModel

from anydocs import telemetry
from anydocs.config import RoleConfig, get_settings

_models: dict[tuple[str, str, str], BaseChatModel] = {}

# Per-request role overrides, e.g. {"judge": "claude-haiku-4-5-20251001"}. Set by the server from
# `overrides.model_roles` and recorded on the turn span.
_role_overrides: ContextVar[dict[str, str] | None] = ContextVar(
    "anydocs_role_overrides", default=None
)

# Swapped in by tests so the suite needs no API key. Keyed by role, like everything else here.
# A plain dict rather than a ContextVar: a test installs these from its own context, and the
# server runs a request in a context the test body never touches.
_fakes: dict[str, BaseChatModel] = {}


def set_role_overrides(overrides: dict[str, str] | None) -> None:
    _role_overrides.set(dict(overrides) if overrides else None)


def set_fake_models(models: dict[str, BaseChatModel] | None) -> None:
    """Install stand-in chat models for the whole process. Tests only."""
    _fakes.clear()
    _fakes.update(models or {})


# `effort` is a Claude 5 feature. Claude 4.x answers a request carrying it with
# `400 This model does not support the effort parameter` — every call, not the first one.
_EFFORT_CAPABLE = re.compile(r"claude-[a-z]+-5")


def resolve_role(role: str) -> RoleConfig:
    """The config for a role, with any per-request model override applied.

    The override is the reason `effort` is re-checked here rather than trusted from the file: a
    role configured for a Claude 5 model carries an effort, and `--judge-model` pointing it at a
    4.x model would otherwise send that effort to a model that rejects it and fail every turn.
    """
    cfg = get_settings().role(role)
    override = (_role_overrides.get() or {}).get(role)
    if override is not None:
        cfg = RoleConfig(**{**vars(cfg), "model": override})
    if cfg.effort and not _EFFORT_CAPABLE.search(cfg.model):
        # Degrade rather than fail, and say so on the span: a model that cannot be asked for an
        # effort level still answers the question.
        telemetry.annotate(**{telemetry.Attr.MODEL_DEGRADED: f"{cfg.model} rejects effort"})
        cfg = RoleConfig(**{**vars(cfg), "effort": ""})
    return cfg


def chat_model(role: str) -> tuple[BaseChatModel, RoleConfig]:
    """The chat model for a role, and the config it came from.

    Cached per (provider, model, effort): constructing a client is cheap but not free, and a turn
    asks for the same role several times.
    """
    cfg = resolve_role(role)
    fake = _fakes.get(role)
    if fake is not None:
        return fake, cfg

    key = (cfg.provider, cfg.model, cfg.effort)
    model = _models.get(key)
    if model is None:
        model = _build(cfg)
        _models[key] = model
    return model, cfg


def _build(cfg: RoleConfig) -> BaseChatModel:
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {"model": cfg.model, "max_tokens": cfg.max_tokens}
        if cfg.effort:
            # Reasoning effort is a Claude 5 feature; Claude 4.x rejects it outright, so it is
            # only sent when the config asks for it. See `config/models.yaml`.
            kwargs["output_config"] = {"effort": cfg.effort}
        else:
            # The Messages API dropped `temperature` in favour of effort, but LangChain still
            # forwards it, and 4.x models still accept it.
            kwargs["temperature"] = cfg.temperature
        return ChatAnthropic(**kwargs)

    if cfg.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.model, max_completion_tokens=cfg.max_tokens, temperature=cfg.temperature
        )

    raise ValueError(
        f"Role {cfg.role!r} names provider {cfg.provider!r}; only anthropic and openai are wired."
    )
