"""Stage S — checkpoint hygiene + ``move_to`` boundary detection.

Two sub-features, one file per the stage convention:

1. **Checkpoint hygiene** (`561eec7`). ``CheckpointManager(keep_last_n=N)``
   auto-prunes after every save so per-step pickles never accumulate past
   N + in-flight write. Stage R v5 iter5 was ENOSPC-killed by 4 × 600 step
   pickles in ``/tmp``; the rolling-window cap prevents recurrence. Runner
   wires ``autoresearch.files.warn_if_tmp_data_dir`` on each checkpoint
   dir so future sweeps see the misconfig before it bites.

2. **``move_to`` boundary detection** (`8a955ae`, F4). The pokemon nav
   tool was reporting ``(False, "Unable to move to (12, 0)…")`` when the
   agent successfully walked off PalletTown's north edge into Route1 —
   verdict was comparing the new map's entry coords against the old
   map's target. ``classify_post_move_outcome`` now returns a structured
   ``(True, "Crossed map boundary …")`` for that case.
"""

from __future__ import annotations

import importlib.util
import inspect
import pickle
import sys
import types
from pathlib import Path

import pytest

from evaluation_utils.checkpoint_manager import CheckpointManager

_REPO = Path(__file__).resolve().parent.parent


# ── shared StubAgent for checkpoint tests ─────────────────────────────


class _StubAgent:
    """Minimal Checkpointable — keeps tests independent of real MACLA."""

    def get_state(self) -> dict[str, object]:
        return {"weights": [1, 2, 3]}

    def load_state(self, state: dict[str, object]) -> None:
        pass

    def get_checkpoint_metadata(self) -> dict[str, object]:
        return {"total_steps": 0}


@pytest.fixture
def agent() -> _StubAgent:
    return _StubAgent()


def _save_n(mgr: CheckpointManager, agent: _StubAgent, n: int, game: str = "g") -> None:
    """Save ``n`` checkpoints with monotonically increasing ids."""
    for i in range(n):
        mgr.save_agent_checkpoint(
            agent=agent,
            game_state={"total_steps": i},
            game_name=game,
            checkpoint_id=f"step_{i:03d}",
        )


# ── load pokemon_tools.py off-graph (its top-level wildcard import on a
# runtime-only sibling package can't resolve from the tests/ context;
# same pattern as test_pokemon_milestones.py) ─────────────────────────────

_pkg = types.ModuleType("mcp_game_servers")
_pkg.__path__ = []
sys.modules.setdefault("mcp_game_servers", _pkg)
sys.modules.setdefault(
    "mcp_game_servers.pokemon_red.game.utils.map_utils",
    types.ModuleType("mcp_game_servers.pokemon_red.game.utils.map_utils"),
)
_TOOLS_PATH = _REPO / "evaluation_utils/mcp_game_servers/pokemon_red/game/utils/pokemon_tools.py"
_spec = importlib.util.spec_from_file_location("pokemon_tools_under_test", _TOOLS_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
classify_post_move_outcome = _module.classify_post_move_outcome


def _classify(**overrides):
    """Helper with sensible PalletTown defaults — tests override only
    the field(s) they care about."""
    args = dict(
        prev_map="PalletTown",
        current_map="PalletTown",
        current_pos=(12, 0),
        target=(12, 0),
        state="Field",
        x_dest=12,
        y_dest=0,
    )
    args.update(overrides)
    return classify_post_move_outcome(**args)


# ─────────────────────────────────────────────────────────────────────────
# Checkpoint hygiene
# ─────────────────────────────────────────────────────────────────────────


class TestRollingCleanup:
    """``CheckpointManager(keep_last_n=N)`` auto-cleans on save so per-step
    pickles never accumulate past N + the in-flight write."""

    @pytest.mark.parametrize(
        "keep_last_n, save_count, expected",
        [
            (0, 10, 10),  # default 0 → unbounded
            (3, 10, 3),  # cap to N most recent
            (2, 5, 2),
        ],
    )
    def test_keep_last_n_caps_pkls_on_save(
        self, tmp_path: Path, agent: _StubAgent, keep_last_n, save_count, expected
    ):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), keep_last_n=keep_last_n)
        _save_n(mgr, agent, save_count)
        pkls = sorted((tmp_path / "g").glob("*.pkl"))
        assert len(pkls) == expected
        if keep_last_n > 0:
            # The N most recent survive (highest step ids)
            survivors = [p.stem.split("_")[-1] for p in pkls]
            assert survivors == [f"{i:03d}" for i in range(save_count - keep_last_n, save_count)]

    def test_kept_checkpoints_are_loadable(self, tmp_path: Path, agent: _StubAgent):
        """After auto-cleanup the surviving pickles must still deserialise."""
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), keep_last_n=2)
        _save_n(mgr, agent, 5)
        for pkl in (tmp_path / "g").glob("*.pkl"):
            with pkl.open("rb") as f:
                data = pickle.load(f)
            assert "agent_state" in data and "timestamp" in data

    def test_companion_json_summary_pruned_alongside_pkl(self, tmp_path: Path, agent: _StubAgent):
        """Each save writes a sibling .json summary; cleanup drops both
        so dangling .json files don't accumulate."""
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path), keep_last_n=2)
        _save_n(mgr, agent, 5)
        pkls = list((tmp_path / "g").glob("*.pkl"))
        jsons = list((tmp_path / "g").glob("*.json"))
        assert len(pkls) == 2
        assert len(jsons) == 2
        assert {p.stem for p in pkls} == {j.stem for j in jsons}


class TestTmpWarningWired:
    """``runner.py`` calls ``autoresearch.files.warn_if_tmp_data_dir`` on
    each game's checkpoint dir at start-up. The helper itself lives in
    autoresearch (PR #102, v0.27.0)."""

    def test_runner_imports_and_calls_warn(self):
        from evaluation_utils import runner

        src = inspect.getsource(runner)
        assert "from autoresearch.files import warn_if_tmp_data_dir" in src
        assert "warn_if_tmp_data_dir(checkpoint_dir)" in src


# ─────────────────────────────────────────────────────────────────────────
# move_to boundary detection
# ─────────────────────────────────────────────────────────────────────────


class TestClassifyPostMoveOutcome:
    """``classify_post_move_outcome`` returns one of:
    - ``(True, "Successfully Move to …")`` — landed on target.
    - ``(True, "Crossed map boundary …")`` — F4: walked into adjacent map.
    - ``(False, "Interrupt by …")`` — dialog/battle interrupted the walk.
    - ``None`` — no verdict yet; the caller should retry.
    """

    def test_landed_on_target_is_success(self):
        ok, msg = _classify()
        assert ok is True
        assert "Successfully Move to (12, 0)" in msg

    def test_map_changed_is_boundary_crossing_success(self):
        """v2 symptom: agent calls move_to(12, 0) at the Pallet→Route1
        edge. Old verdict said False ("unable to move"); F4 says True
        with boundary-crossing context."""
        ok, msg = _classify(current_map="Route1", current_pos=(12, 18))
        assert ok is True
        for fragment in ("Crossed", "PalletTown", "Route1", "(12, 18)"):
            assert fragment in msg

    def test_boundary_crossing_takes_priority_over_target_match(self):
        """If the new map's entry coords coincidentally equal the original
        target, the map-change is still the more informative signal."""
        ok, msg = _classify(current_map="Route1")  # target & pos both (12,0)
        assert ok is True
        assert "Crossed" in msg

    @pytest.mark.parametrize(
        "state, fragment",
        [
            ("Dialog", "Dialog"),
            ("Battle", "Battle"),
            ("WildBattle", "Battle"),
        ],
    )
    def test_dialog_or_battle_interrupt_is_failure(self, state, fragment):
        ok, msg = _classify(current_pos=(5, 5), state=state)
        assert ok is False
        assert fragment in msg

    def test_same_map_not_at_target_returns_none_for_retry(self):
        """``None`` = "no resolution yet"; the caller's retry loop continues."""
        assert _classify(current_pos=(11, 5)) is None


class TestMoveToWiring:
    """Both the sync ``PokemonTools.move_to`` and the async MCP variant
    must dispatch through ``classify_post_move_outcome`` rather than
    re-implementing the verdict inline."""

    @pytest.mark.parametrize(
        "path",
        [
            "evaluation_utils/mcp_game_servers/pokemon_red/game/utils/pokemon_tools.py",
            "evaluation_utils/mcp_game_servers/pokemon_red/game/utils/pokemon_tools_mcp.py",
        ],
    )
    def test_move_to_dispatches_through_helper(self, path):
        src = (_REPO / path).read_text()
        assert "classify_post_move_outcome(" in src, (
            f"{path} must invoke classify_post_move_outcome rather than "
            "re-implementing the post-walk verdict inline."
        )
