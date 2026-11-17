"""Session memory, keyed by session_id.

In-process for now; a Redis backend would sit behind this same interface. What it
holds is deliberately small: the turns, the corpora in play, the versions the user mentioned, and
the doc ids already cited — the last of which is pinned into retrieval so a follow-up like "and
how do I configure that?" lands on the right page.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from anydocs import telemetry

SESSION_TTL_S = 24 * 3600
SUMMARY_TOKEN_BUDGET = 1500  # rough: 4 chars per token
SUMMARY_SPAN = telemetry.SESSION_SUMMARY


@dataclass
class Turn:
    message: str
    answer: str = ""
    confidence: float = 0.0
    citations: list[str] = field(default_factory=list)
    at: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    user_id: str = ""
    turns: list[Turn] = field(default_factory=list)
    corpora: list[str] = field(default_factory=list)
    versions: list[str] = field(default_factory=list)
    cited_doc_ids: list[str] = field(default_factory=list)
    summary: str = ""
    updated_at: float = field(default_factory=time.time)

    @property
    def expired(self) -> bool:
        return time.time() - self.updated_at > SESSION_TTL_S

    def context_blurb(self) -> str:
        """What the planner sees of the conversation so far."""
        if self.summary:
            recent = self.turns[-1:] if self.turns else []
            tail = "; ".join(f"asked {t.message!r}" for t in recent)
            return f"{self.summary} Most recently: {tail}" if tail else self.summary
        return "; ".join(f"asked {t.message!r}" for t in self.turns[-3:])

    def pinned_docs(self) -> list[str]:
        """Doc ids already cited this session, newest first."""
        return list(reversed(self.cited_doc_ids))[:10]


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str, user_id: str = "") -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.expired:
                session = Session(session_id=session_id, user_id=user_id)
                self._sessions[session_id] = session
            if user_id and not session.user_id:
                session.user_id = user_id
            return session

    def record_turn(
        self,
        session_id: str,
        turn: Turn,
        *,
        corpora: list[str] | None = None,
        versions: list[str] | None = None,
    ) -> Session:
        session = self.get(session_id)
        with self._lock:
            session.turns.append(turn)
            for corpus in corpora or []:
                if corpus not in session.corpora:
                    session.corpora.append(corpus)
            for version in versions or extract_versions(turn.message):
                if version not in session.versions:
                    session.versions.append(version)
            for doc_id in turn.citations:
                if doc_id not in session.cited_doc_ids:
                    session.cited_doc_ids.append(doc_id)
            session.updated_at = time.time()
        return session

    def needs_summary(self, session: Session) -> bool:
        raw = sum(len(t.message) + len(t.answer) for t in session.turns)
        return raw > SUMMARY_TOKEN_BUDGET * 4

    def set_summary(self, session_id: str, summary: str) -> None:
        session = self.get(session_id)
        with self._lock:
            session.summary = summary
            # The rolling summary replaces the raw turns it covers; the last turn stays so a
            # follow-up still has literal context.
            session.turns = session.turns[-1:]

    def roll_up(self, session_id: str) -> str:
        """Replace the raw turns with a summary when they pass the token budget.

        Returns the summary, or "" if none was needed. The summary goes on the turn span, so a
        session eval can see what the agent thought the conversation was about — which is the
        thing that goes wrong when a follow-up lands on the wrong page.
        """
        session = self.get(session_id)
        if not self.needs_summary(session):
            return ""

        from langchain_core.messages import HumanMessage, SystemMessage

        from anydocs.models.roles import call_role
        from anydocs.prompts import load_prompt

        prompt = load_prompt("session_summary")
        transcript = "\n\n".join(
            f"User: {turn.message}\nAgent: {turn.answer[:600]}" for turn in session.turns
        )
        if session.summary:
            transcript = f"Summary so far: {session.summary}\n\n{transcript}"

        completion = call_role(
            SUMMARY_SPAN,
            "summarizer",
            [SystemMessage(prompt.text), HumanMessage(transcript)],
            prompt_version=prompt.version,
        )
        summary = completion.text.strip()
        if summary:
            self.set_summary(session_id, summary)
        return summary

    def clear(self, session_id: str = "") -> None:
        with self._lock:
            if session_id:
                self._sessions.pop(session_id, None)
            else:
                self._sessions.clear()


_VERSION_MENTION = re.compile(r"\bv?(\d+\.\d+(?:\.\d+)?|\d{4}-\d{2}-\d{2})\b")


def extract_versions(text: str) -> list[str]:
    return list(dict.fromkeys(_VERSION_MENTION.findall(text)))[:5]


_store = SessionStore()


def get_session_store() -> SessionStore:
    return _store
