"""Settings: environment variables plus the YAML files under config/.

One `get_settings()` singleton so the YAML is parsed once per process. Tests that need a
different policy call `reset_settings_cache()` or construct `Settings` directly rather than
mutating module state.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def find_root(candidates: list[Path]) -> Path:
    """The first candidate holding `config/models.yaml`, else the first candidate.

    The package's parent is the root when running from source. It is not when the package was
    installed non-editable — as Vercel's `uv sync --no-editable` does — because the wheel holds
    `anydocs/` and not `config/` or `bundled/`, so the parent is `site-packages`. The working
    directory is the fallback, since a host that ships the source tree runs from its root.
    Falling back to the first candidate keeps the "Missing config file" error naming the path a
    developer would expect.
    """
    for candidate in candidates:
        if (candidate / "config" / "models.yaml").is_file():
            return candidate
    return candidates[0]


REPO_ROOT = find_root([Path(__file__).resolve().parent.parent, Path.cwd()])
CONFIG_DIR = REPO_ROOT / "config"

load_dotenv(REPO_ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class RoleConfig:
    """One entry from config/models.yaml `roles`.

    `temperature` and `effort` are both here because the two providers expose different knobs:
    the Anthropic SDK takes `output_config.effort` and no longer accepts a temperature, while
    OpenAI takes a temperature and has no effort. Each adapter sends the one it supports and
    ignores the other, so a role can be moved between providers without rewriting its config.
    """

    role: str
    provider: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.0
    effort: str = ""  # anthropic only: low | medium | high | xhigh | max


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dim: int


@dataclass
class Settings:
    data_dir: Path
    bundle_dir: Path
    api_token: str
    signing_key: str
    project_name: str
    trace_target: str
    flush_spans_per_request: bool
    framework_spans: bool
    phoenix_endpoint: str
    arize_space_id: str
    arize_api_key: str
    arize_endpoint: str
    search_provider: str
    roles: dict[str, RoleConfig]
    embeddings: EmbeddingConfig
    pricing: dict[str, dict[str, float]]
    policy: dict[str, Any]
    operator_notes: str
    operator_notes_version: str
    # Side-effect classes this process may never exceed, whatever a request asks for.
    mode_ceiling: dict[str, str] = field(default_factory=dict)
    _raw_models: dict[str, Any] = field(default_factory=dict, repr=False)

    def role(self, name: str) -> RoleConfig:
        try:
            return self.roles[name]
        except KeyError:
            known = ", ".join(sorted(self.roles))
            raise ConfigError(f"Unknown model role {name!r}. Known roles: {known}") from None

    def price_per_mtok(self, model: str) -> dict[str, float]:
        """Input/output price per million tokens. Unknown models price at zero, loudly."""
        return self.pricing.get(model, {"input": 0.0, "output": 0.0})

    @property
    def corpora_registry(self) -> Path:
        return self.data_dir / "corpora.json"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def _operator_notes() -> tuple[str, str]:
    """Return (text, version). The version line is `version: N` anywhere in the file."""
    path = CONFIG_DIR / "operator-notes.md"
    if not path.exists():
        return "", "0"
    text = path.read_text()
    version = "0"
    for line in text.splitlines():
        if line.strip().startswith("version:"):
            version = line.split(":", 1)[1].strip()
            break
    return text, version


MODES = ("off", "ask", "auto")


def _mode_ceiling() -> dict[str, str]:
    """`ANYDOCS_<CLASS>_MODE`, for the side-effect classes a public deployment must lock.

    Read here rather than written into the YAML so the policy file stays one copy, and enforced
    as a ceiling in `policy.py` so a per-request override cannot raise it again.
    """
    ceiling: dict[str, str] = {}
    for key in ("crawl", "write", "delete"):
        value = os.getenv(f"ANYDOCS_{key.upper()}_MODE", "").strip().lower()
        if not value:
            continue
        if value not in MODES:
            raise ConfigError(f"ANYDOCS_{key.upper()}_MODE must be one of {MODES}; got {value!r}")
        ceiling[key] = value
    return ceiling


def build_settings() -> Settings:
    models = _load_yaml(CONFIG_DIR / "models.yaml")
    policy = _load_yaml(CONFIG_DIR / "policy.default.yaml")

    default_provider = models.get("default_provider", "anthropic")
    roles: dict[str, RoleConfig] = {}
    for name, cfg in (models.get("roles") or {}).items():
        if "model" not in cfg:
            raise ConfigError(f"Role {name!r} in models.yaml has no `model`")
        roles[name] = RoleConfig(
            role=name,
            provider=cfg.get("provider", default_provider),
            model=cfg["model"],
            max_tokens=int(cfg.get("max_tokens", 4096)),
            temperature=float(cfg.get("temperature", 0.0)),
            effort=str(cfg.get("effort", "")),
        )

    emb = models.get("embeddings") or {}
    if not emb.get("model"):
        raise ConfigError("models.yaml needs an `embeddings` block with a `model`")
    embeddings = EmbeddingConfig(
        provider=emb.get("provider", "openai"),
        model=emb["model"],
        dim=int(emb.get("dim", 1536)),
    )

    notes, notes_version = _operator_notes()

    return Settings(
        data_dir=Path(os.getenv("ANYDOCS_DATA_DIR", REPO_ROOT / "data")).resolve(),
        bundle_dir=Path(os.getenv("ANYDOCS_BUNDLE_DIR", REPO_ROOT / "bundled")).resolve(),
        api_token=os.getenv("ANYDOCS_API_TOKEN", "dev-token"),
        signing_key=os.getenv("ANYDOCS_SIGNING_KEY", "insecure-dev-key"),
        project_name=os.getenv("ANYDOCS_PROJECT_NAME", "anydocs-agent"),
        trace_target=os.getenv("TRACE_TARGET", "phoenix").lower(),
        # Serverless hosts freeze the process after the response, so a batched exporter never
        # reaches its timer. Set this on any platform that does not keep the process alive.
        flush_spans_per_request=os.getenv("ANYDOCS_FLUSH_SPANS", "").lower()
        in ("1", "true", "yes"),
        # LangChain's own spans, off by default. Measured on a real turn: because its callbacks
        # do not put their spans in the OpenTelemetry context, nothing of ours can nest inside
        # them, so the result is not extra detail but a second complete copy of the turn — every
        # specialist, tool and model call twice, 60 spans where 25 say the same thing, under
        # names like `model` and `tools` instead of `librarian` and `search_docs`. Our tree
        # already carries the OpenInference span kinds AX renders. Turn these on to see the graph
        # the way LangChain reports it; leave them off to read the turn.
        framework_spans=os.getenv("ANYDOCS_FRAMEWORK_SPANS", "").lower()
        in ("1", "true", "yes", "on"),
        phoenix_endpoint=os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006"),
        arize_space_id=os.getenv("ARIZE_SPACE_ID", ""),
        arize_api_key=os.getenv("ARIZE_API_KEY", ""),
        # The OTLP/HTTP traces path. Must end in `/v1/traces`: the collector answers 200 at
        # `/v1` and drops whatever it is sent.
        arize_endpoint=os.getenv("ARIZE_ENDPOINT", "https://otlp.arize.com/v1/traces"),
        search_provider=os.getenv("SEARCH_PROVIDER", "tavily").lower(),
        roles=roles,
        embeddings=embeddings,
        pricing=models.get("pricing") or {},
        policy=policy,
        operator_notes=notes,
        operator_notes_version=notes_version,
        mode_ceiling=_mode_ceiling(),
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return build_settings()


def reset_settings_cache() -> None:
    """Drop the memoized settings. For tests that change env or config files."""
    get_settings.cache_clear()
