"""Vector-memory provider — semantic recall via cosine similarity.

Lifted from pokemon_red's ``VectorMemory`` class
(``agents/pokemon_red/openai_pokemon_vector_memory.py``) and reshaped to
implement the ``MemoryProvider`` interface so any agent can opt in.

Embedding backends, in priority order:

1. ``embedding_fn`` injected at construction (tests, custom embedders).
2. OpenAI ``text-embedding-3-small`` if ``OPENAI_API_KEY`` is set.
3. ``sentence-transformers`` local model (default ``all-MiniLM-L6-v2``,
   384-dim, ~80MB, CPU-fast) — fully local, no API needed. This is the
   happy path for vLLM-Gemma deployments.
4. SHA256 hash-based fallback when no semantic backend is available —
   degrades semantic similarity to exact-string overlap but keeps dedup,
   retrieval lifecycle, and stats working without crashing.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import openai
from loguru import logger
from sentence_transformers import SentenceTransformer

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
        embedding_fn: Callable[[str], np.ndarray] | None = None,
        *,
        embedding_model: str = "text-embedding-3-small",
        local_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_memories: int = 100,
        default_top_k: int = 3,
        default_threshold: float = 0.5,
        dim: int = 1536,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        repetition_decay_alpha: float = 0.0,
        repetition_decay_window: int = 20,
    ) -> None:
        self._embedding_fn = embedding_fn
        self._embedding_model = embedding_model
        self._local_model_name = local_model
        self._max_memories = max_memories
        self._default_top_k = default_top_k
        self._default_threshold = default_threshold
        self._dim = dim
        # MMR rerank: λ=1 → relevance only (default top-k), λ=0 → diversity only.
        self._use_mmr = use_mmr
        self._mmr_lambda = mmr_lambda
        # Repetition decay: each retrieval call multiplies every memory's
        # recent_hits by (1 - 1/W); selected memories get +1. Effective score
        # = sim / (1 + α · recent_hits). α=0 disables.
        self._repetition_decay_alpha = repetition_decay_alpha
        self._repetition_decay_window = max(1, repetition_decay_window)
        self._memories: list[dict[str, Any]] = []
        self._client = None  # lazy — only when OpenAI backend is used
        self._client_unavailable = False
        self._local_model = None  # lazy — only when sentence-transformers is used
        self._local_model_unavailable = False
        self._backend_in_use: str = "none"  # "openai" | "local" | "hash" | "injected"
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
        top_k: int | None = None,
        threshold: float | None = None,
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
            "backend": self._backend_in_use,
        }

    # ── Internals (mirrored from pokemon's VectorMemory) ─────────────

    def _embed(self, text: str) -> np.ndarray:
        if self._embedding_fn is not None:
            self._backend_in_use = "injected"
            return self._embedding_fn(text)

        # 1. OpenAI cloud (if API key is set).
        if not self._client_unavailable:
            try:
                if self._client is None:
                    if not os.environ.get("OPENAI_API_KEY"):
                        raise RuntimeError("OPENAI_API_KEY not set")
                    self._client = openai.OpenAI()
                response = self._client.embeddings.create(
                    model=self._embedding_model,
                    input=text,
                )
                self._backend_in_use = "openai"
                return np.array(response.data[0].embedding)
            except Exception as e:
                logger.info(
                    f"[VectorMemory] OpenAI embeddings unavailable ({e}), "
                    f"trying local sentence-transformers"
                )
                self._client_unavailable = True

        # 2. sentence-transformers local model.
        if not self._local_model_unavailable:
            try:
                if self._local_model is None:
                    logger.info(
                        f"[VectorMemory] loading local embedding model "
                        f"{self._local_model_name!r} (first call only)"
                    )
                    self._local_model = SentenceTransformer(self._local_model_name)
                vec = self._local_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
                self._backend_in_use = "local"
                return np.asarray(vec)
            except Exception as e:
                logger.warning(
                    f"[VectorMemory] sentence-transformers unavailable ({e}), "
                    f"falling back to hash embeddings (semantic match degraded)"
                )
                self._local_model_unavailable = True

        # 3. Hash-based deterministic fallback.
        self._backend_in_use = "hash"
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
        self._memories.append(
            {
                "content": content,
                "embedding": embedding,
                "metadata": metadata or {},
                "timestamp": time.time(),
                "recent_hits": 0.0,
            }
        )
        self._stats["adds"] += 1
        if len(self._memories) > self._max_memories:
            self._memories = self._memories[-self._max_memories :]

    def _retrieve_similar(
        self, query: str, *, top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
        self._stats["retrievals"] += 1
        if not self._memories:
            return []

        # Decay every memory's recent-hit count once per retrieval call.
        # Done before scoring so the call that follows a long absence sees
        # near-zero penalty even on memories that were heavily retrieved.
        if self._repetition_decay_alpha > 0:
            decay = 1.0 - 1.0 / self._repetition_decay_window
            for m in self._memories:
                m["recent_hits"] *= decay

        q = self._embed(query)
        candidates = []
        for m in self._memories:
            sim = self._cosine(q, m["embedding"])
            penalty = 1.0 + self._repetition_decay_alpha * m.get("recent_hits", 0.0)
            score = sim / penalty if penalty > 0 else sim
            if score >= threshold:
                candidates.append({"_memory": m, "similarity": sim, "score": score})

        if candidates:
            self._stats["hits"] += 1

        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = (
            self._mmr_rerank(candidates, q, top_k=top_k) if self._use_mmr else candidates[:top_k]
        )

        # Mark every memory we are about to surface as recently retrieved so
        # the next call sees its decay applied.
        if self._repetition_decay_alpha > 0:
            for c in selected:
                c["_memory"]["recent_hits"] += 1.0

        return [
            {
                "content": c["_memory"]["content"],
                "metadata": c["_memory"]["metadata"],
                "similarity": c["similarity"],
            }
            for c in selected
        ]

    def _mmr_rerank(
        self,
        candidates: list[dict[str, Any]],
        query_emb: np.ndarray,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Greedy maximal-marginal-relevance rerank over already-scored candidates.

        Each step picks the candidate maximising
        ``λ·score(c) − (1−λ)·max_sim(c, already_selected)`` —
        keeps the high-score winners but penalises clustering near them.
        """
        if not candidates or top_k <= 0:
            return []
        remaining = candidates[:]
        selected: list[dict[str, Any]] = [remaining.pop(0)]  # global max first
        lam = self._mmr_lambda
        while remaining and len(selected) < top_k:
            best_idx, best_mmr = 0, -float("inf")
            for i, c in enumerate(remaining):
                max_div = max(
                    self._cosine(c["_memory"]["embedding"], s["_memory"]["embedding"])
                    for s in selected
                )
                mmr = lam * c["score"] - (1.0 - lam) * max_div
                if mmr > best_mmr:
                    best_mmr, best_idx = mmr, i
            selected.append(remaining.pop(best_idx))
        return selected

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
