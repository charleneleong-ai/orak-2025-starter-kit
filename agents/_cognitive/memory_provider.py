"""Abstract base for pluggable agent memory providers.

A memory provider gives an agent persistent recall across turns and (optionally)
across sessions. The interface is deliberately small and modeled on
hermes-agent's pattern, with the lifecycle hooks an in-game agent actually uses:

* ``initialize(session_id)`` — connect, allocate, warm up
* ``prefetch(query)`` — recall before the next turn (returns formatted text)
* ``sync_turn(user, assistant)`` — persist the turn after the LLM call
* ``add_event(content, metadata)`` — record significant events outside the
  per-turn flow (badges, map transitions, success conditions, etc.)
* ``on_session_end()`` — final extraction / flush at episode end

Concrete providers (``VectorMemoryProvider``, future ``BuiltinMemoryProvider``,
etc.) implement this and can be plugged into any agent — pokemon, 2048, mario,
future cognitive-loop variants — without re-implementing semantic memory.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """Abstract base for in-game memory providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. ``'vector'``, ``'builtin'``)."""

    # ── Required lifecycle ───────────────────────────────────────────

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is configured and ready.

        Called during agent setup to decide whether to activate. Should not
        make network calls — just check config / installed deps.
        """

    @abstractmethod
    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """One-time setup at session start.

        ``kwargs`` may include ``game_name``, ``run_id``, etc. — providers
        ignore what they don't need.
        """

    # ── Per-turn recall + persist ────────────────────────────────────

    def prefetch(self, query: str) -> str:
        """Recall context for the upcoming turn.

        Return formatted text to inject into the prompt, or empty string for
        nothing relevant. Default is no-op so providers that don't recall
        per-turn (event-only stores) can opt out.
        """
        return ""

    def sync_turn(self, user_content: str, assistant_content: str) -> None:
        """Persist a completed turn. Non-blocking; queue if backend is slow."""

    def add_event(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Record a significant event (badge, map transition, success, etc.)."""

    # ── Session boundaries ───────────────────────────────────────────

    def on_session_end(self) -> None:
        """Flush / extract at episode end. Default is no-op."""

    # ── Optional: prompt block for system message ────────────────────

    def system_prompt_block(self) -> str:
        """Return text to inject into the agent's system prompt (e.g. a
        usage note for the model). Empty string skips."""
        return ""

    # ── Stats for observability ─────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return runtime stats (memory count, retrieval hit rate, etc.)
        for trajectory logging."""
        return {}
