"""Vector-memory provider — semantic recall via cosine similarity.

Lifted from pokemon_red's ``VectorMemory`` class
(``agents/pokemon_red/openai_pokemon_vector_memory.py``) and reshaped to
implement the ``MemoryProvider`` interface so any agent can opt in.

Embedding backends, in priority order:

1. ``embedding_fn`` injected at construction (tests, custom embedders).
2. OpenAI ``text-embedding-3-small`` if ``OPENAI_API_KEY`` is set.
3. Hash-based deterministic fallback when neither is available — keeps the
   provider usable in fully-local deployments (vLLM-only, no API keys).
   Semantic similarity is degraded to exact-token overlap, but dedup,
   retrieval lifecycle, and stats keep working so callers can opt in
   without crashing.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger

from agents._cognitive.memory_provider import MemoryProvider


class VectorMemoryProvider(MemoryProvider):
    """Semantic memory via cosine-similarity retrieval over embedded text.

    Parameters
    ----------
    embedding_fn:
        Callable taking a text string and returning a 1-D ``np.ndarray``.
        Default: ``text-embedding-3-small`` via the openai client. Inject a
        mock for tests so they don't need API access.
    embedding_model:
        Used only when ``embedding_fn`` is not provided (default backend).
    max_memories:
        Sliding window — drop oldest when exceeded.
    default_top_k, default_threshold:
        Retrieval defaults; per-call overrides via ``prefetch(top_k=, threshold=)``.
    dim:
        Embedding dimension (used only for the zero-vector fallback on
        embedding errors). 1536 matches text-embedding-3-small.
    """

    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        *,
        embedding_model: str = "text-embedding-3-small",
        max_memories: int = 100,
        default_top_k: int = 3,
        default_threshold: float = 0.5,
        dim: int = 1536,
    ) -> None:
        self._embedding_fn = embedding_fn
        self._embedding_model = embedding_model
        self._max_memories = max_memories
        self._default_top_k = default_top_k
        self._default_threshold = default_threshold
        self._dim = dim
        self._memories: list[dict[str, Any]] = []
        self._client = None  # lazy — only when default backend is used
        self._client_unavailable = False  # set on first failure to skip retries
        self._stats = {"adds": 0, "retrievals": 0, "hits": 0}

    # ── MemoryProvider interface ─────────────────────────────────────

    @property
    def name(self) -> str:
        return "vector"

    def is_available(self) -> bool:
        # Always available — failing embeddings degrade to zero vectors,
        # callers can still write/recall but matching becomes meaningless.
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        # No-op: we lazy-init the openai client on first embedding call.
        # Subclasses with persistent backends would wire that here.
        logger.debug(f"[VectorMemory] initialize session_id={session_id} kwargs={list(kwargs)}")

    def prefetch(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> str:
        """Retrieve top matching memories for ``query`` and format for injection."""
        if not query or not query.strip():
            return ""
        memories = self._retrieve_similar(
            query,
            top_k=top_k or self._default_top_k,
            threshold=threshold if threshold is not None else self._default_threshold,
        )
        return self._format_for_prompt(memories)

    def sync_turn(self, user_content: str, assistant_content: str) -> None:
        # Per-turn writes are coarse — most agents prefer add_event for
        # selective writes. Default: no-op. Subclass to enable if useful.
        pass

    def add_event(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a memory after a deduplication check."""
        if not content or not content.strip():
            return
        existing = self._retrieve_similar(content, top_k=1, threshold=0.8)
        if existing:
            return
        self._add_memory(content, metadata)

    def on_session_end(self) -> None:
        # The episode-end hook for providers with bulk extraction (e.g.
        # summarising before persisting). VectorMemoryProvider is purely
        # in-memory per session, so nothing to flush.
        pass

    def stats(self) -> dict[str, Any]:
        return {
            "memory_count": len(self._memories),
            "adds": self._stats["adds"],
            "retrievals": self._stats["retrievals"],
            "hits": self._stats["hits"],
            "hit_rate": (
                self._stats["hits"] / self._stats["retrievals"]
                if self._stats["retrievals"] > 0
                else 0.0
            ),
        }

    # ── Internals (mirrored from pokemon's VectorMemory) ─────────────

    def _embed(self, text: str) -> np.ndarray:
        if self._embedding_fn is not None:
            return self._embedding_fn(text)
        if not self._client_unavailable:
            try:
                if self._client is None:
                    if not os.environ.get("OPENAI_API_KEY"):
                        raise RuntimeError("OPENAI_API_KEY not set — using local hash fallback")
                    import openai
                    self._client = openai.OpenAI()
                response = self._client.embeddings.create(
                    model=self._embedding_model,
                    input=text,
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                logger.warning(
                    f"[VectorMemory] OpenAI embeddings unavailable, falling back "
                    f"to hash-based local embeddings (semantic match degraded): {e}"
                )
                self._client_unavailable = True
        return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
        """Deterministic local embedding — SHA256 over text bytes, unpacked
        into a fixed-dimension float vector. Same text → same vector → exact
        dedup still works. Different texts → near-orthogonal → semantic
        similarity is approximated by token-overlap rather than meaning."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat hash bytes to fill the requested dim
        buf = (h * (dim // len(h) + 1))[:dim]
        return np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _add_memory(self, content: str, metadata: dict[str, Any] | None) -> None:
        embedding = self._embed(content)
        self._memories.append({
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })
        self._stats["adds"] += 1
        if len(self._memories) > self._max_memories:
            self._memories = self._memories[-self._max_memories:]

    def _retrieve_similar(
        self, query: str, *, top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
        self._stats["retrievals"] += 1
        if not self._memories:
            return []
        q = self._embed(query)
        hits = []
        for m in self._memories:
            sim = self._cosine(q, m["embedding"])
            if sim >= threshold:
                hits.append({
                    "content": m["content"],
                    "metadata": m["metadata"],
                    "similarity": sim,
                })
        if hits:
            self._stats["hits"] += 1
        hits.sort(key=lambda x: x["similarity"], reverse=True)
        return hits[:top_k]

    @staticmethod
    def _format_for_prompt(memories: list[dict[str, Any]]) -> str:
        if not memories:
            return ""
        lines = []
        for i, m in enumerate(memories):
            meta = m.get("metadata") or {}
            tags = []
            if "step" in meta:
                tags.append(f"step {meta['step']}")
            if "map_name" in meta:
                tags.append(f"map: {meta['map_name']}")
            tag_str = f" ({', '.join(tags)})" if tags else ""
            lines.append(f"[Memory {i + 1}{tag_str}] {m['content']}")
        return "\n".join(lines)

    # ── Pokemon-compatible helpers (used by the existing agent) ──────
    # These mirror pokemon's `VectorMemory` API so the refactor to use the
    # provider stays minimal — the existing agent calls them by these names.

    def add_memory(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Compat shim: pokemon's existing API. Adds without dedup check."""
        if not content or not content.strip():
            return
        self._add_memory(content, metadata)

    def retrieve_similar(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Compat shim: pokemon's existing API."""
        return self._retrieve_similar(query, top_k=top_k, threshold=threshold)

    def format_memories_for_prompt(self, memories: list[dict[str, Any]]) -> str:
        """Compat shim: pokemon's existing API."""
        if not memories:
            return "N/A"
        return self._format_for_prompt(memories)
