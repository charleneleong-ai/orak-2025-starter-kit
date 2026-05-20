"""Stage S — checkpoint hygiene: rolling N-most-recent cleanup + /tmp warning.

Motivation: Stage R v5 iter5 was killed at launch by `/tmp` ENOSPC because
4 iters × 600 steps of per-step `UnifiedMaclaAgent_step_*.pkl` pickles
saturated `/tmp` (the much smaller scratch partition vs `/workspace`'s
199G). Cleanup existed (`CheckpointManager.cleanup_old_checkpoints`) but
was never auto-called. These tests pin the wiring + the loud warning when
a sweep is configured to write under `/tmp/`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest


class _StubAgent:
    """Minimal Checkpointable stub — keeps the tests independent of the
    real MACLA agent surface."""

    def get_state(self) -> dict[str, object]:
        return {"weights": [1, 2, 3]}

    def load_state(self, state: dict[str, object]) -> None:
        pass

    def get_checkpoint_metadata(self) -> dict[str, object]:
        return {"total_steps": 0}


class TestRollingCleanup:
    """``CheckpointManager(keep_last_n=N)`` auto-cleans on save so per-step
    pickles never accumulate past N + the in-flight write."""

    def test_keep_last_n_defaults_to_zero_unbounded(self, tmp_path: Path):
        from evaluation_utils.checkpoint_manager import CheckpointManager

        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        agent = _StubAgent()
        for i in range(10):
            mgr.save_agent_checkpoint(
                agent=agent,
                game_state={"total_steps": i},
                game_name="g",
                checkpoint_id=f"step_{i:03d}",
            )
        assert len(list((tmp_path / "g").glob("*.pkl"))) == 10  # no auto-prune

    def test_keep_last_n_caps_pkls_on_save(self, tmp_path: Path):
        from evaluation_utils.checkpoint_manager import CheckpointManager

        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), keep_last_n=3)
        agent = _StubAgent()
        for i in range(10):
            mgr.save_agent_checkpoint(
                agent=agent,
                game_state={"total_steps": i},
                game_name="g",
                checkpoint_id=f"step_{i:03d}",
            )
        pkls = sorted((tmp_path / "g").glob("*.pkl"))
        assert len(pkls) == 3
        # The 3 most recent (007, 008, 009) survive
        assert [p.stem.split("_")[-1] for p in pkls] == ["007", "008", "009"]

    def test_kept_checkpoints_are_loadable(self, tmp_path: Path):
        """After auto-cleanup the surviving pickles must still deserialise."""
        from evaluation_utils.checkpoint_manager import CheckpointManager

        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), keep_last_n=2)
        agent = _StubAgent()
        for i in range(5):
            mgr.save_agent_checkpoint(
                agent=agent,
                game_state={"total_steps": i},
                game_name="g",
                checkpoint_id=f"step_{i:03d}",
            )
        for pkl in (tmp_path / "g").glob("*.pkl"):
            with pkl.open("rb") as f:
                data = pickle.load(f)
            assert "agent_state" in data and "timestamp" in data

    def test_companion_json_summary_pruned_alongside_pkl(self, tmp_path: Path):
        """Each save writes a sibling .json summary; cleanup must drop both
        so we don't leave dangling .json files."""
        from evaluation_utils.checkpoint_manager import CheckpointManager

        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), keep_last_n=2)
        agent = _StubAgent()
        for i in range(5):
            mgr.save_agent_checkpoint(
                agent=agent,
                game_state={"total_steps": i},
                game_name="g",
                checkpoint_id=f"step_{i:03d}",
            )
        pkls = list((tmp_path / "g").glob("*.pkl"))
        jsons = list((tmp_path / "g").glob("*.json"))
        assert len(pkls) == 2
        assert len(jsons) == 2
        assert {p.stem for p in pkls} == {j.stem for j in jsons}


class TestTmpWarning:
    """Emit a loud loguru warning at runner start-up when ``GAME_DATA_DIR``
    points at ``/tmp/`` — that's the exact misconfiguration that killed
    Stage R v5 iter5."""

    @pytest.mark.parametrize(
        "path,expect_warn",
        [
            ("/tmp/orak-anything", True),
            ("/tmp/sub/dir", True),
            ("/workspace/orak-x", False),
            ("/var/data", False),
        ],
    )
    def test_warns_on_tmp_root(self, path: str, expect_warn: bool):
        from evaluation_utils.commons import warn_if_tmp_data_dir

        warnings: list[str] = []
        try:
            warn_if_tmp_data_dir(path, _warn=warnings.append)
        except Exception as exc:  # pragma: no cover - regression net
            pytest.fail(f"warn_if_tmp_data_dir raised: {exc!r}")
        assert bool(warnings) == expect_warn
        if expect_warn:
            assert "/tmp" in warnings[0]
