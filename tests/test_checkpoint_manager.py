"""Tests for CheckpointManager.cleanup_old_checkpoints — keep_recent integration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evaluation_utils.checkpoint_manager import CheckpointManager


def _make_checkpoint_pair(directory: Path, name: str, mtime: float) -> None:
    pkl = directory / f"{name}.pkl"
    pkl.parent.mkdir(parents=True, exist_ok=True)
    pkl.write_bytes(b"")
    (pkl.with_suffix(".json")).write_text("{}")
    os.utime(pkl, (mtime, mtime))


@pytest.fixture
def manager(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(checkpoint_dir=str(tmp_path))


class TestCleanupOldCheckpoints:
    """Rolling-window checkpoint cleanup via autoresearch.files.keep_recent."""

    def test_keeps_n_newest_and_removes_sidecar_json(
        self, manager: CheckpointManager, tmp_path: Path
    ) -> None:
        game_dir = tmp_path / "pokemon_red"
        for i in range(5):
            _make_checkpoint_pair(game_dir, f"Agent_step{i}", mtime=100.0 + i)

        manager.cleanup_old_checkpoints("Agent", game_name="pokemon_red", keep_last_n=2)

        remaining = sorted(p.name for p in game_dir.iterdir())
        assert remaining == [
            "Agent_step3.json",
            "Agent_step3.pkl",
            "Agent_step4.json",
            "Agent_step4.pkl",
        ]

    def test_no_op_when_below_limit(self, manager: CheckpointManager, tmp_path: Path) -> None:
        game_dir = tmp_path / "pokemon_red"
        for i in range(2):
            _make_checkpoint_pair(game_dir, f"Agent_step{i}", mtime=100.0 + i)

        manager.cleanup_old_checkpoints("Agent", game_name="pokemon_red", keep_last_n=5)

        assert len(list(game_dir.iterdir())) == 4  # 2 pkl + 2 json

    def test_missing_game_dir_is_silent(self, manager: CheckpointManager) -> None:
        # Does not raise even though pokemon_red/ doesn't exist.
        manager.cleanup_old_checkpoints("Agent", game_name="pokemon_red", keep_last_n=2)

    def test_only_matches_agent_specific_files(
        self, manager: CheckpointManager, tmp_path: Path
    ) -> None:
        # Two agents share a game dir; cleanup must only touch the named agent.
        game_dir = tmp_path / "pokemon_red"
        for i in range(3):
            _make_checkpoint_pair(game_dir, f"AgentA_step{i}", mtime=100.0 + i)
        for i in range(3):
            _make_checkpoint_pair(game_dir, f"AgentB_step{i}", mtime=200.0 + i)

        manager.cleanup_old_checkpoints("AgentA", game_name="pokemon_red", keep_last_n=1)

        remaining = sorted(p.name for p in game_dir.iterdir())
        # AgentA: 1 newest pair kept. AgentB: untouched (3 pairs).
        agent_a = [n for n in remaining if n.startswith("AgentA_")]
        agent_b = [n for n in remaining if n.startswith("AgentB_")]
        assert agent_a == ["AgentA_step2.json", "AgentA_step2.pkl"]
        assert len(agent_b) == 6  # 3 pairs
