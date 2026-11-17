"""Prompt loading.

Repo files are the source of truth. The harness pushes them to the prompt hub under a tag, and
a deployment may later pull by tag with the file as fallback — but nothing in this process
invents prompt text, so `prompt.version` on a span always names something a reader can go find.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

from anydocs.config import REPO_ROOT

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_VERSION = re.compile(r"^\s*(?:<!--\s*)?version:\s*(\S+?)\s*(?:-->)?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    version: str


@functools.lru_cache(maxsize=64)
def load_prompt(name: str) -> Prompt:
    """Load a prompt by name.

    `digest_page` -> anydocs/prompts/digest_page.md
    `librarian` -> anydocs/specialists/librarian/prompt.md
    `orchestrator` -> anydocs/orchestrator/prompt.md
    """
    for candidate in (
        PROMPTS_DIR / f"{name}.md",
        REPO_ROOT / "anydocs" / "specialists" / name / "prompt.md",
        REPO_ROOT / "anydocs" / "orchestrator" / "prompt.md" if name == "orchestrator" else None,
    ):
        if candidate and candidate.exists():
            raw = candidate.read_text()
            match = _VERSION.search(raw)
            version = match.group(1) if match else "0"
            body = _VERSION.sub("", raw, count=1).strip()
            return Prompt(name=name, text=body, version=f"{name}@{version}")
    raise FileNotFoundError(f"No prompt file for {name!r}")

