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

from experiments.gspo.collate import GSPOSample, collate_iter

_ = GSPOSample  # keep import live — currently asserted via dataclass attrs below


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
