"""StarCraft II milestone-ladder progress metric (star_craft.progress)."""

import pytest

from evaluation_utils.mcp_game_servers.star_craft.progress import (
    episode_peaks,
    extract_metrics,
    merge_peaks,
    milestone_score,
    run_progress,
    split_episodes,
)

# Real SC2 obs_str field formats (from seed-1 game_states). Supply left omitted
# on purpose — it is absent in many real states and must default to 0.
_OBS = (
    "At 03:43 game time, our situation. Game time: 03:43 . "
    "Mineral: 970 . Supply used: 23 . Supply cap: 31 . Worker supply: 21 . "
    "Nexus count: 1 Pylon count: 1 Gas buildings count: 2 Gateway count: 3 . "
    "Probe count: 18 . Enemy unittypeid.zergling: 4 Enemy unittypeid.roach: 2"
)

_ZERO = {
    "building_count": 0,
    "supply_cap": 0,
    "worker_supply": 0,
    "supply_used": 0,
    "enemy_unit_count": 0,
}
# Peaks that clear all 7 non-victory rungs.
_ALL_MET = {
    "building_count": 8,
    "supply_cap": 23,
    "worker_supply": 16,
    "supply_used": 34,
    "enemy_unit_count": 1,
}


class TestMilestoneScore:
    @pytest.mark.parametrize(
        "peaks,victory,expected",
        [
            (_ZERO, False, 0.0),  # nothing reached
            (_ZERO, True, 12.5),  # victory alone = 1 of 8
            (_ALL_MET, False, 87.5),  # all 7 economic rungs, no win
            (_ALL_MET, True, 100.0),  # everything incl. victory
        ],
    )
    def test_headline_boundaries(self, peaks, victory, expected):
        assert milestone_score(peaks, victory) == expected

    @pytest.mark.parametrize(
        "field,at,rungs_at",
        [
            ("building_count", 2, 1),  # M1 only
            ("building_count", 8, 2),  # M1 + M5
            ("supply_cap", 23, 1),  # M2
            ("worker_supply", 16, 1),  # M3
            ("supply_used", 20, 1),  # M4
            ("supply_used", 34, 2),  # M4 + M6
            ("enemy_unit_count", 1, 1),  # M7
        ],
    )
    def test_threshold_is_inclusive_and_credits_expected_rungs(self, field, at, rungs_at):
        # just-at the threshold credits the rung(s); one below credits one fewer
        assert milestone_score({**_ZERO, field: at}, victory=False) == rungs_at / 8 * 100
        assert milestone_score({**_ZERO, field: at - 1}, victory=False) == (rungs_at - 1) / 8 * 100

    @pytest.mark.parametrize(
        "field",
        ["building_count", "supply_used", "worker_supply", "supply_cap", "enemy_unit_count"],
    )
    def test_monotonic_in_each_field(self, field):
        scores = [
            milestone_score({**_ZERO, field: v}, victory=False) for v in (0, 1, 5, 10, 20, 40)
        ]
        assert scores == sorted(scores)  # raising a single field never lowers the score

    def test_missing_field_treated_as_zero(self):
        # only supply_used present → M4 credited (1 rung), no KeyError
        assert milestone_score({"supply_used": 25}, victory=False) == 12.5


class TestSplitEpisodes:
    def test_splits_on_game_time_reset(self):
        steps = [
            {"game_time_sec": 10},
            {"game_time_sec": 20},
            {"game_time_sec": 5},  # reset → new episode
            {"game_time_sec": 15},
        ]
        eps = split_episodes(steps)
        assert [len(e) for e in eps] == [2, 2]
        assert eps[1][0]["game_time_sec"] == 5

    def test_small_dip_does_not_split(self):
        # 1s decrease is noise, not an episode boundary (>=2s guard)
        steps = [{"game_time_sec": 10}, {"game_time_sec": 9}, {"game_time_sec": 12}]
        assert len(split_episodes(steps)) == 1

    def test_monotonic_run_is_single_episode(self):
        steps = [{"game_time_sec": t} for t in (0, 5, 10, 30, 60)]
        assert len(split_episodes(steps)) == 1

    def test_empty_input_returns_no_episodes(self):
        assert split_episodes([]) == []


class TestEpisodePeaks:
    def test_peak_is_max_over_episode_not_terminal(self):
        ep = [{"supply_used": 10}, {"supply_used": 38}, {"supply_used": 12}]
        assert episode_peaks(ep)["supply_used"] == 38

    def test_missing_fields_default_zero(self):
        assert episode_peaks([{"supply_used": 5}])["building_count"] == 0

    def test_empty_episode_is_all_zero(self):
        peaks = episode_peaks([])
        assert peaks["supply_used"] == 0 and peaks["enemy_unit_count"] == 0


class TestMergePeaks:
    def test_keeps_running_max_either_direction(self):
        assert merge_peaks({"supply_used": 30}, {"supply_used": 10})["supply_used"] == 30
        assert merge_peaks({"supply_used": 10}, {"supply_used": 30})["supply_used"] == 30

    def test_missing_fields_default_zero(self):
        assert merge_peaks({}, {"supply_used": 5})["building_count"] == 0

    def test_folding_merge_matches_episode_peaks(self):
        steps = [{"supply_used": 10, "building_count": 3}, {"supply_used": 38, "building_count": 1}]
        running = {}
        for s in steps:
            running = merge_peaks(running, s)
        assert running == episode_peaks(steps)


class TestRunProgress:
    def test_aggregates_mean_best_and_winrate_across_episodes(self):
        steps = [
            # episode 1: peak supply_used 20 → M4 only = 1/8 = 12.5, no victory
            {"game_time_sec": 5, "supply_used": 20},
            {"game_time_sec": 10, "supply_used": 20},
            # episode 2 (game_time reset): building_count 8 → M1+M5, plus victory = 3/8 = 37.5
            {"game_time_sec": 2, "building_count": 8, "victory": True},
        ]
        out = run_progress(steps)
        assert out["n_episodes"] == 2
        assert out["starcraft_progress"] == pytest.approx(25.0)  # mean(12.5, 37.5)
        assert out["starcraft_progress_best"] == pytest.approx(37.5)
        assert out["star_craft_victory"] == pytest.approx(0.5)  # 1 of 2 episodes

    def test_empty_run_scores_zero(self):
        out = run_progress([])
        assert out["starcraft_progress"] == 0.0
        assert out["n_episodes"] == 0


class TestExtractMetrics:
    def test_parses_all_fields_from_real_obs_format(self):
        m = extract_metrics(_OBS)
        assert m["game_time_sec"] == 3 * 60 + 43
        assert m["mineral"] == 970
        assert m["supply_used"] == 23
        assert m["supply_cap"] == 31
        assert m["worker_supply"] == 21

    def test_building_count_sums_structures_excluding_workers(self):
        # Nexus 1 + Pylon 1 + Gas buildings 2 + Gateway 3 = 7; Probe count excluded
        assert extract_metrics(_OBS)["building_count"] == 7

    def test_enemy_units_sum_across_types(self):
        assert extract_metrics(_OBS)["enemy_unit_count"] == 6  # zergling 4 + roach 2

    def test_absent_field_defaults_to_zero(self):
        assert extract_metrics(_OBS)["supply_left"] == 0  # not present in _OBS

    def test_empty_text_yields_all_zero_metrics(self):
        m = extract_metrics("")
        assert m["supply_used"] == 0 and m["building_count"] == 0 and m["game_time_sec"] == 0
