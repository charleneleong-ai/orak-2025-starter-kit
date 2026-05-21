"""GSPO data collation: game_states.jsonl + evaluation_summary.json -> per-step
{prompt, completion, reward, group_id} JSONL records.

Sequence-level reward: every step in the same iter shares the trajectory's
final score / 7 (normalised to 0-1, matches GSPO's sequence-relative
advantage assumption). Group id defaults to the run_id - cheap placeholder
that lets a single sweep flow through the pipeline; real multi-rollout
groups come from re-rolling K trajectories from a shared checkpoint and
sharing the resulting group_id across all K of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.gspo.collate import GSPOSample, collate_iter, collate_sweep

_ = GSPOSample  # keep import live — currently asserted via dataclass attrs below


def _write_iter_dir(parent: Path, name: str, n_steps: int, final_score: float) -> Path:
    """Create a synthetic iter dir with `n_steps` rows + an eval summary.
    Hoisted so multi-iter sweep tests can build their fixtures inline."""
    d = parent / name
    d.mkdir()
    rows = [
        {
            "iteration": i + 1,
            "obs": {"obs_str": f"obs at step {i + 1}", "game_info": {"map_name": "X"}},
            "action": f"use_tool(move_to, (x_dest={i}, y_dest=0))",
        }
        for i in range(n_steps)
    ]
    (d / "game_states.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "evaluation_summary.json").write_text(
        json.dumps({"episodes": [{"final_score": final_score}]})
    )
    return d


@pytest.fixture
def iter_dir(tmp_path: Path) -> Path:
    """Synthesise a 3-step iter dir with final_score=5/7 (matches v4 iter1)."""
    d = tmp_path / "stage_r_v5_iter1_synth"
    d.mkdir()
    rows = [
        {
            "iteration": 1,
            "obs": {
                "obs_str": "obs at step 1",
                "game_info": {"score": "0", "map_name": "RedsHouse2f"},
            },
            "action": "use_tool(move_to, (x_dest=6, y_dest=4))",
            "current_score": 0.0,
        },
        {
            "iteration": 2,
            "obs": {
                "obs_str": "obs at step 2",
                "game_info": {"score": "1", "map_name": "PalletTown"},
            },
            "action": "use_tool(move_to, (x_dest=12, y_dest=5))",
            "current_score": 1.0,
        },
        {
            "iteration": 3,
            "obs": {
                "obs_str": "obs at step 3",
                "game_info": {"score": "5", "map_name": "ViridianCity"},
            },
            "action": "use_tool(interact_with_object, (object_name='MART_CLERK'))",
            "current_score": 5.0,
        },
    ]
    (d / "game_states.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "evaluation_summary.json").write_text(
        json.dumps({"episodes": [{"episode_id": 1, "final_score": 5.0}]})
    )
    return d


class TestCollateIter:
    """One ``GSPOSample`` per game_states.jsonl row, sharing trajectory reward."""

    def test_emits_one_sample_per_row(self, iter_dir: Path):
        samples = collate_iter(iter_dir)
        assert len(samples) == 3

    def test_reward_is_normalised_trajectory_final_score(self, iter_dir: Path):
        # final_score 5/7 = 0.7143 — sequence-level reward shared by all steps.
        for s in collate_iter(iter_dir):
            assert s.reward == pytest.approx(5.0 / 7.0)

    def test_group_id_is_run_id_by_default(self, iter_dir: Path):
        samples = collate_iter(iter_dir)
        assert all(s.group_id == iter_dir.name for s in samples)

    def test_prompt_is_obs_str(self, iter_dir: Path):
        samples = collate_iter(iter_dir)
        assert samples[0].prompt == "obs at step 1"
        assert samples[2].prompt == "obs at step 3"

    def test_completion_is_action_string(self, iter_dir: Path):
        samples = collate_iter(iter_dir)
        assert samples[1].completion == "use_tool(move_to, (x_dest=12, y_dest=5))"

    def test_iter_step_is_1_indexed(self, iter_dir: Path):
        samples = collate_iter(iter_dir)
        assert [s.iter_step for s in samples] == [1, 2, 3]

    def test_run_id_threaded_through(self, iter_dir: Path):
        samples = collate_iter(iter_dir)
        assert all(s.run_id == iter_dir.name for s in samples)


class TestErrors:
    """Defensive: missing files raise rather than silently emitting zero rows."""

    def test_missing_game_states_raises(self, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        (d / "evaluation_summary.json").write_text(json.dumps({"episodes": [{"final_score": 0.0}]}))
        with pytest.raises(FileNotFoundError):
            collate_iter(d)

    def test_missing_eval_summary_raises(self, tmp_path: Path):
        d = tmp_path / "no_summary"
        d.mkdir()
        (d / "game_states.jsonl").write_text("")
        with pytest.raises(FileNotFoundError):
            collate_iter(d)


class TestNormalisation:
    """``score_max`` scales the reward — pokemon=7, mario varies, 2048=different."""

    @pytest.mark.parametrize(
        "final, score_max, expected", [(7.0, 7.0, 1.0), (3.5, 7.0, 0.5), (0.0, 7.0, 0.0)]
    )
    def test_reward_scales_with_score_max(self, tmp_path: Path, final, score_max, expected):
        d = tmp_path / "iter"
        d.mkdir()
        (d / "game_states.jsonl").write_text(
            json.dumps({"iteration": 1, "obs": {"obs_str": "x"}, "action": "y"}) + "\n"
        )
        (d / "evaluation_summary.json").write_text(
            json.dumps({"episodes": [{"final_score": final}]})
        )
        samples = collate_iter(d, score_max=score_max)
        assert samples[0].reward == pytest.approx(expected)


class TestCollateSweep:
    """``collate_sweep`` walks an entire game-data root (one dir per iter)
    and aggregates into a single sample list. Defensive on incomplete
    iters (in-flight, crashed)."""

    def test_aggregates_all_complete_iters_in_order(self, tmp_path: Path):
        root = tmp_path / "pokemon_red"
        root.mkdir()
        _write_iter_dir(root, "sweep_iter1_T1", n_steps=2, final_score=5.0)
        _write_iter_dir(root, "sweep_iter2_T2", n_steps=3, final_score=4.0)
        _write_iter_dir(root, "sweep_iter3_T3", n_steps=1, final_score=7.0)
        samples = collate_sweep(root)
        # 2 + 3 + 1 = 6 samples across 3 iters
        assert len(samples) == 6
        # group_ids preserve per-iter identity (one group per iter,
        # default placeholder until a re-roll launcher fixes K rollouts)
        assert len({s.group_id for s in samples}) == 3
        # rewards reflect each iter's final_score / score_max
        per_iter = {s.run_id: s.reward for s in samples}
        assert per_iter["sweep_iter1_T1"] == pytest.approx(5.0 / 7.0)
        assert per_iter["sweep_iter2_T2"] == pytest.approx(4.0 / 7.0)
        assert per_iter["sweep_iter3_T3"] == pytest.approx(7.0 / 7.0)

    def test_iters_emitted_in_sorted_order(self, tmp_path: Path):
        """Stable ordering — downstream batching code assumes
        deterministic sample sequence per (game_root, score_max)."""
        root = tmp_path / "pokemon_red"
        root.mkdir()
        # Names ordered to expose sort: T3 written first but should
        # appear last after sorting.
        _write_iter_dir(root, "sweep_iter1_T3", n_steps=1, final_score=1.0)
        _write_iter_dir(root, "sweep_iter1_T1", n_steps=1, final_score=2.0)
        _write_iter_dir(root, "sweep_iter1_T2", n_steps=1, final_score=3.0)
        samples = collate_sweep(root)
        run_ids = [s.run_id for s in samples]
        assert run_ids == sorted(run_ids)

    def test_skips_iter_missing_evaluation_summary(self, tmp_path: Path):
        """In-flight sweep with an iter still running has only
        game_states.jsonl, no evaluation_summary.json. Skip silently;
        don't crash."""
        root = tmp_path / "pokemon_red"
        root.mkdir()
        _write_iter_dir(root, "sweep_iter1_T1", n_steps=2, final_score=5.0)
        in_flight = root / "sweep_iter2_T2"
        in_flight.mkdir()
        (in_flight / "game_states.jsonl").write_text(
            json.dumps({"iteration": 1, "obs": {"obs_str": "x"}, "action": "y"}) + "\n"
        )
        samples = collate_sweep(root)
        assert len(samples) == 2
        assert all(s.run_id == "sweep_iter1_T1" for s in samples)

    def test_skips_iter_missing_game_states(self, tmp_path: Path):
        """Defensive against a launcher that wrote eval_summary but
        the game_states.jsonl was lost (cleanup script, ENOSPC, etc)."""
        root = tmp_path / "pokemon_red"
        root.mkdir()
        _write_iter_dir(root, "sweep_iter1_T1", n_steps=2, final_score=5.0)
        no_states = root / "sweep_iter2_T2"
        no_states.mkdir()
        (no_states / "evaluation_summary.json").write_text(
            json.dumps({"episodes": [{"final_score": 0.0}]})
        )
        samples = collate_sweep(root)
        assert len(samples) == 2

    def test_skips_non_directory_entries(self, tmp_path: Path):
        """game-data roots sometimes have stray files (eval.log,
        results.jsonl). Don't treat them as iter dirs."""
        root = tmp_path / "pokemon_red"
        root.mkdir()
        (root / "stray.txt").write_text("not an iter dir")
        _write_iter_dir(root, "sweep_iter1_T1", n_steps=1, final_score=5.0)
        samples = collate_sweep(root)
        assert len(samples) == 1

    def test_empty_root_returns_empty_list(self, tmp_path: Path):
        root = tmp_path / "pokemon_red"
        root.mkdir()
        assert collate_sweep(root) == []

    def test_nonexistent_root_raises(self, tmp_path: Path):
        with pytest.raises(NotADirectoryError):
            collate_sweep(tmp_path / "does_not_exist")

    def test_score_max_passed_through(self, tmp_path: Path):
        """score_max overrides reach each per-iter collate_iter call."""
        root = tmp_path / "mario"
        root.mkdir()
        _write_iter_dir(root, "iter1", n_steps=1, final_score=50.0)
        samples = collate_sweep(root, score_max=100.0)
        assert samples[0].reward == pytest.approx(0.5)


class TestGroupIdSidecar:
    """The re-roll launcher writes ``gspo_group.json`` into each iter dir
    to tag it with a shared group_id. Collator picks it up — without
    this, every iter is its own group (variance=0, no gradient signal)."""

    def test_sidecar_overrides_default_group_id(self, tmp_path: Path):
        d = _write_iter_dir(tmp_path, "iter1", n_steps=2, final_score=5.0)
        (d / "gspo_group.json").write_text(json.dumps({"group_id": "shared_g"}))
        samples = collate_iter(d)
        assert all(s.group_id == "shared_g" for s in samples)
        # run_id is independent — still the dir name (for traceability).
        assert all(s.run_id == "iter1" for s in samples)

    def test_no_sidecar_falls_back_to_run_id(self, tmp_path: Path):
        """Existing collation behavior preserved when no sidecar present."""
        d = _write_iter_dir(tmp_path, "iter1", n_steps=2, final_score=5.0)
        samples = collate_iter(d)
        assert all(s.group_id == "iter1" for s in samples)

    def test_sidecar_missing_group_id_field_falls_back(self, tmp_path: Path):
        """Defensive: a malformed sidecar (file present but no group_id
        key) shouldn't crash; falls back to run_id default."""
        d = _write_iter_dir(tmp_path, "iter1", n_steps=2, final_score=5.0)
        (d / "gspo_group.json").write_text(json.dumps({"comment": "no group_id key"}))
        samples = collate_iter(d)
        assert all(s.group_id == "iter1" for s in samples)

    def test_sweep_aggregates_sidecar_groups(self, tmp_path: Path):
        """Re-roll case: 3 iter dirs, all tagged with the same group_id
        via sidecar → collate_sweep produces 1 group, n=3 samples-per-iter
        × 3 iters = 9 samples sharing a group_id."""
        root = tmp_path / "pokemon_red"
        root.mkdir()
        for name, score in [("k1", 5.0), ("k2", 3.0), ("k3", 7.0)]:
            d = _write_iter_dir(root, name, n_steps=3, final_score=score)
            (d / "gspo_group.json").write_text(json.dumps({"group_id": "reroll_g"}))
        samples = collate_sweep(root)
        assert len(samples) == 9
        assert {s.group_id for s in samples} == {"reroll_g"}
        # Three distinct run_ids, one per K rollout
        assert {s.run_id for s in samples} == {"k1", "k2", "k3"}
        # Rewards still per-trajectory (each rollout's final_score)
        assert {round(s.reward, 4) for s in samples} == {
            round(5.0 / 7, 4),
            round(3.0 / 7, 4),
            round(7.0 / 7, 4),
        }
