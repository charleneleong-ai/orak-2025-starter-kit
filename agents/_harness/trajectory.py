"""Trajectory writer — ShareGPT-shaped per-episode log.

Two upgrades over orak's existing ``raw_requests.jsonl``:

1. Per-step records are appended live (same behaviour today), but the writer
   also tracks whether the action came from a real LLM response vs a fallback
   (exception caught). At episode end the records are rolled up into one of
   two files:
   * ``trajectory_samples.jsonl`` — episodes that completed with no fallbacks
   * ``failed_trajectories.jsonl`` — episodes with any fallback or crash

2. Each step is recorded in ShareGPT-shaped form (``conversations`` array of
   ``{from, value}`` turns) so the data is reusable for SFT later without
   reformatting. ``<REASONING_SCRATCHPAD>`` tags become ``<think>`` for
   compatibility with thinking-mode datasets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def convert_scratchpad_to_think(content: str) -> str:
    """Replace ``<REASONING_SCRATCHPAD>`` with ``<think>``."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return (
        content.replace("<REASONING_SCRATCHPAD>", "<think>")
        .replace("</REASONING_SCRATCHPAD>", "</think>")
    )


def has_incomplete_scratchpad(content: str) -> bool:
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


@dataclass
class StepRecord:
    step: int
    system_prompt: Optional[str]
    user_prompt: str
    assistant_output: str
    action: str
    reasoning: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    cached_tokens: int = 0
    is_fallback: bool = False  # True if action came from exception handler
    fallback_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_sharegpt(self) -> dict[str, Any]:
        """ShareGPT-shaped per-step entry."""
        convs: list[dict[str, str]] = []
        if self.system_prompt:
            convs.append({"from": "system", "value": self.system_prompt})
        convs.append({"from": "human", "value": self.user_prompt})
        convs.append({
            "from": "gpt",
            "value": convert_scratchpad_to_think(self.assistant_output),
        })
        return {
            "step": self.step,
            "action": self.action,
            "reasoning": self.reasoning,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
            "tokens": {
                "prompt": self.tokens_prompt,
                "completion": self.tokens_completion,
                "total": self.tokens_total,
                "cached": self.cached_tokens,
            },
            "conversations": convs,
            "timestamp": self.timestamp,
        }


class TrajectoryWriter:
    """Per-episode buffer of ``StepRecord``s, flushed at episode end.

    Coexists with orak's existing ``raw_requests.jsonl`` writer — this only
    emits the rolled-up ``trajectory_samples.jsonl`` / ``failed_trajectories.jsonl``
    files. Use ``add_step`` to buffer in-memory, ``flush_episode`` to write.
    """

    def __init__(self, log_dir: str | Path, *, model: str = "unknown") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.success_path = self.log_dir / "trajectory_samples.jsonl"
        self.failed_path = self.log_dir / "failed_trajectories.jsonl"
        self._buffer: list[StepRecord] = []

    def add_step(self, record: StepRecord) -> None:
        self._buffer.append(record)

    def flush_episode(
        self,
        episode_id: int,
        *,
        completed: bool,
        final_score: float,
        game_name: str,
    ) -> Path:
        """Write the buffered episode to success/failed file. Returns the path."""
        any_fallback = any(r.is_fallback for r in self._buffer)
        is_success = completed and not any_fallback
        target = self.success_path if is_success else self.failed_path

        entry = {
            "episode_id": episode_id,
            "game_name": game_name,
            "model": self.model,
            "completed": completed,
            "final_score": final_score,
            "n_steps": len(self._buffer),
            "n_fallbacks": sum(1 for r in self._buffer if r.is_fallback),
            "total_cached_tokens": sum(r.cached_tokens for r in self._buffer),
            "total_input_tokens": sum(r.tokens_prompt for r in self._buffer),
            "total_output_tokens": sum(r.tokens_completion for r in self._buffer),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "steps": [r.to_sharegpt() for r in self._buffer],
        }
        try:
            with target.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(
                f"trajectory[{game_name} ep={episode_id}] flushed → {target.name} "
                f"(success={is_success}, steps={len(self._buffer)}, fallbacks={entry['n_fallbacks']})"
            )
        except Exception as e:
            logger.warning(f"failed to write episode trajectory: {e}")

        self._buffer.clear()
        return target
