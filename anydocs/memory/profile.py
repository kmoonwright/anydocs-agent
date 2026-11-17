"""The opt-in user profile.

Keyed by `user_id`, and empty until someone says otherwise. The distinction that matters here is
between what a user *declared* and what the agent *inferred*: a declared preference changes
behaviour, an inferred one is recorded as a suggestion with the evidence behind it and only
becomes a preference when the user confirms it. The agent never silently starts behaving
differently because it noticed a pattern.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anydocs.config import get_settings

# What a user may declare. A key outside this set is refused rather than stored, so the profile
# cannot become a dumping ground that silently reshapes the orchestrator's prompt.
DECLARABLE = {
    "language": "Primary programming language or SDK, e.g. 'python'.",
    "versions": "Product versions in use, e.g. 'relayr v3'.",
    "verbosity": "brief | normal | detailed.",
    "allow_learning": "Whether the agent may ingest new sources without asking: 'yes' or 'no'.",
}


@dataclass
class Suggestion:
    """An inferred preference, with the evidence for it. Not in effect until confirmed."""

    key: str
    value: str
    evidence: list[str] = field(default_factory=list)
    observed: int = 1
    first_seen: str = ""
    last_seen: str = ""


@dataclass
class Profile:
    user_id: str
    preferences: dict[str, str] = field(default_factory=dict)
    suggestions: dict[str, Suggestion] = field(default_factory=dict)
    version: int = 0
    updated_at: str = ""

    @property
    def opted_in(self) -> bool:
        """A profile with nothing declared is not in effect, however much was inferred."""
        return bool(self.preferences)

    def as_prompt_block(self) -> str:
        """The short block injected into the orchestrator prompt.

        Only declared preferences. Suggestions are deliberately absent: injecting them would be
        acting on them, which is the thing this design exists to prevent.
        """
        if not self.preferences:
            return ""
        lines = [f"- {key}: {value}" for key, value in sorted(self.preferences.items())]
        return (
            "The user has told us the following. Honour it unless this turn contradicts it.\n"
            + "\n".join(lines)
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "suggestions": {k: asdict(v) for k, v in self.suggestions.items()},
            "version": self.version,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Profile:
        return cls(
            user_id=data["user_id"],
            preferences=dict(data.get("preferences") or {}),
            suggestions={
                k: Suggestion(**v) for k, v in (data.get("suggestions") or {}).items()
            },
            version=int(data.get("version", 0)),
            updated_at=str(data.get("updated_at", "")),
        )


class ProfileStore:
    """One JSON file per user under the data directory.

    On disk rather than in memory, because a profile is the one piece of memory that is supposed
    to outlive a session — and because a user who declared a preference last week should not have
    to declare it again after a restart.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        path = self._directory or (get_settings().data_dir / "profiles")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, user_id: str) -> Path:
        import hashlib

        # User ids reach this from a request body, so the filename is a hash rather than the id.
        return self.directory / f"{hashlib.sha256(user_id.encode()).hexdigest()[:24]}.json"

    def get(self, user_id: str) -> Profile:
        if not user_id:
            return Profile(user_id="")
        path = self._path(user_id)
        if not path.exists():
            return Profile(user_id=user_id)
        try:
            return Profile.from_json(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError):
            # A corrupt profile must not break a turn; an empty one is the safe default.
            return Profile(user_id=user_id)

    def _save(self, profile: Profile) -> Profile:
        profile.version += 1
        profile.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        with self._lock:
            self._path(profile.user_id).write_text(json.dumps(profile.to_json(), indent=2))
        return profile

    def declare(self, user_id: str, preferences: dict[str, str]) -> Profile:
        """Record what the user said about themselves. This is what changes behaviour."""
        unknown = [k for k in preferences if k not in DECLARABLE]
        if unknown:
            raise ValueError(
                f"Not a declarable preference: {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(DECLARABLE))}"
            )
        profile = self.get(user_id)
        for key, value in preferences.items():
            if value:
                profile.preferences[key] = value
            else:
                profile.preferences.pop(key, None)
            # A declaration settles the matter; any standing suggestion about it is moot.
            profile.suggestions.pop(key, None)
        return self._save(profile)

    def suggest(self, user_id: str, key: str, value: str, evidence: str) -> Profile | None:
        """Record something the agent noticed. Does not change behaviour.

        Repeated observations accumulate evidence rather than overwriting it, so a confirmation
        prompt can say *why* the agent thinks this — which is what makes it answerable.
        """
        if not user_id or key not in DECLARABLE:
            return None
        profile = self.get(user_id)
        if key in profile.preferences:
            return profile  # the user has already spoken on this

        now = datetime.now(UTC).isoformat(timespec="seconds")
        existing = profile.suggestions.get(key)
        if existing and existing.value == value:
            existing.observed += 1
            existing.last_seen = now
            if evidence and evidence not in existing.evidence:
                existing.evidence = (existing.evidence + [evidence])[-5:]
        else:
            profile.suggestions[key] = Suggestion(
                key=key,
                value=value,
                evidence=[evidence] if evidence else [],
                first_seen=now,
                last_seen=now,
            )
        return self._save(profile)

    def confirm(self, user_id: str, key: str, accept: bool = True) -> Profile:
        """Promote a suggestion to a preference, or drop it. Only the user does this."""
        profile = self.get(user_id)
        suggestion = profile.suggestions.pop(key, None)
        if suggestion and accept:
            profile.preferences[key] = suggestion.value
        return self._save(profile)

    def forget(self, user_id: str) -> None:
        with self._lock:
            self._path(user_id).unlink(missing_ok=True)


_store: ProfileStore | None = None


def get_profile_store() -> ProfileStore:
    global _store
    if _store is None:
        _store = ProfileStore()
    return _store


def reset_profile_store() -> None:
    global _store
    _store = None


# Inferred preferences the agent is allowed to notice. Deliberately narrow: each is something a
# question states outright, not something guessed from tone or topic.
_LANGUAGE_HINTS = {
    "python": ("python", "pip install", "pydantic", "def ", "asyncio"),
    "typescript": ("typescript", "npm install", "tsconfig", "node.js"),
    "go": ("golang", "go get", "goroutine"),
    "ruby": ("ruby", "gem install", "rails"),
    "java": ("java", "maven", "gradle"),
}


def infer_from_turn(user_id: str, message: str) -> str:
    """Notice a stated preference in a question and file it as a suggestion.

    Returns the key it suggested, or "". Called once per turn; it never changes the turn.
    """
    if not user_id:
        return ""
    lowered = message.lower()
    for language, hints in _LANGUAGE_HINTS.items():
        if any(hint in lowered for hint in hints):
            get_profile_store().suggest(
                user_id, "language", language, evidence=message[:160]
            )
            return "language"
    return ""
