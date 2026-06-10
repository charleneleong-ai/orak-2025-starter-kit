"""Interaction-sweep mechanism: milestone-stall detector (macla_lib) + interactable
parser (pokemon adapter) + graduated hint→override controller (interaction_sweep).

The detector and controller are game-agnostic; only the parser is per-game. See
docs/specs/2026-06-09-interaction-sweep-design.md.
"""

import inspect
import pickle

import pytest

from agents.macla import unified
from agents.macla.interaction_sweep import decide_interaction_sweep, render_sweep_hint
from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem
from agents.pokemon_red.game_adapter import (
    interaction_action,
    interaction_targets,
    make_observation_preprocessor,
)

_TARGETS = [
    ("SPRITE_GIRL_4", 17, 9),
    ("SPRITE_GAMBLER_ASLEEP_5", 18, 9),
    ("Warp→ViridianNicknameHouse", 21, 9),
]


def _decide(stall_steps, **kw):
    base = dict(
        stall_steps=stall_steps,
        looping=True,
        targets=_TARGETS,
        tried=set(),
        player_pos=(19, 10),
    )
    base.update(kw)
    return decide_interaction_sweep(**base)


# A trimmed Viridian frame: two NPC sprites, one warp, the rest walkable/blocked.
_GRID_OBS = """State: Field

[Map Info]
Map Name: ViridianCity, (x_max , y_max): (39, 35)
Your position (x, y): (19, 10)

Map on Screen:
(15,  9): X\t(16,  9): X\t(17,  9): SPRITE_GIRL_4\t(18,  9): SPRITE_GAMBLER_ASLEEP_5\t(19,  9): O\t(21,  9): Warp→ViridianNicknameHouse
(15, 10): X\t(16, 10): X\t(17, 10): O\t(18, 10): O\t(19, 10): O\t(20, 10): O
"""


@pytest.fixture
def mem() -> EnhancedHierarchicalMemorySystem:
    return EnhancedHierarchicalMemorySystem()


class TestMilestoneStallDetector:
    """Counts steps since the milestone score last increased (the story-stall signal)."""

    def test_stall_counts_steps_since_last_gain(self, mem):
        mem.record_milestone_step(5)  # reach M5 — baseline, stall resets to 0
        for _ in range(5):
            mem.record_milestone_step(5)  # 5 flat steps since the gain
        assert mem.milestone_stall_steps == 5

    def test_stall_steps_reset_when_score_increases(self, mem):
        for _ in range(5):
            mem.record_milestone_step(5)
        mem.record_milestone_step(6)
        assert mem.milestone_stall_steps == 0


class TestInteractableParser:
    """interaction_targets pulls SPRITE_* (talk) and Warp→* (enter) tiles + coords from the grid."""

    def test_parses_sprites_and_warps_with_coords(self):
        targets = interaction_targets(_GRID_OBS)
        assert ("SPRITE_GIRL_4", 17, 9) in targets
        assert ("SPRITE_GAMBLER_ASLEEP_5", 18, 9) in targets
        assert ("Warp→ViridianNicknameHouse", 21, 9) in targets

    def test_ignores_walkable_and_blocked_tiles(self):
        labels = {label for label, _, _ in interaction_targets(_GRID_OBS)}
        assert "O" not in labels and "X" not in labels
        assert len(interaction_targets(_GRID_OBS)) == 3

    @pytest.mark.parametrize("obs", ["", "State: Field\nMap on Screen:\n", "garbage"])
    def test_no_targets_on_empty_or_gridless_obs(self, obs):
        assert interaction_targets(obs) == []

    def test_preprocessor_strips_grid_so_sweep_needs_raw_obs(self):
        # The pokemon obs preprocessor rewrites the obs into a structured
        # summary that DROPS the "Map on Screen" tile grid. _base_fallback only
        # sees the preprocessed obs, so Stage S must parse the RAW (pre-process)
        # obs — feeding it the preprocessed one finds zero interactables and the
        # sweep never fires (the second inertness bug after the regex fix).
        pre = make_observation_preprocessor()
        assert interaction_targets(_GRID_OBS), "raw obs must expose interactables"
        assert interaction_targets(pre.preprocess(_GRID_OBS)) == []

    def test_parses_space_padded_single_digit_coords(self):
        # Interior maps (OaksLab is 8 wide) render single-digit tiles space-padded
        # to width: "( 5,  2): SPRITE_OAK_5". The grid aligns columns, so the gate
        # NPC the agent must talk to is always padded — the parser must see it.
        obs = (
            "Map on Screen:\n"
            "( 2,  1): SPRITE_POKEDEX_6\t( 5,  2): SPRITE_OAK_5\t( 7,  1): Warp→RedsHouse1f\n"
        )
        targets = interaction_targets(obs)
        assert ("SPRITE_OAK_5", 5, 2) in targets
        assert ("Warp→RedsHouse1f", 7, 1) in targets


class TestSweepController:
    """Graduated decision: none below threshold → hint → override, gated by looping + untried targets."""

    def test_none_below_hint_threshold(self):
        assert _decide(29, hint_after=30).mode == "none"

    def test_none_when_not_looping(self):
        assert _decide(100, looping=False).mode == "none"

    def test_hint_between_thresholds(self):
        d = _decide(40, hint_after=30, override_after=60)
        assert d.mode == "hint"
        assert "SPRITE_GIRL_4" in d.hint and "ViridianNicknameHouse" in d.hint

    def test_override_at_escalation_threshold(self):
        d = _decide(60, hint_after=30, override_after=60)
        assert d.mode == "override"

    def test_override_picks_nearest_untried(self):
        # player at (19,10): SPRITE_GAMBLER (18,9) is dist 2, GIRL (17,9) dist 3, warp dist 3.
        d = _decide(60, hint_after=30, override_after=60)
        assert d.target == ("SPRITE_GAMBLER_ASLEEP_5", 18, 9)

    def test_override_skips_tried_targets(self):
        d = _decide(60, hint_after=30, override_after=60, tried={(18, 9)})
        assert d.target == ("SPRITE_GIRL_4", 17, 9)

    def test_none_when_all_targets_tried(self):
        tried = {(x, y) for _, x, y in _TARGETS}
        assert _decide(100, tried=tried).mode == "none"


class TestSweepHintRendering:
    """Hint differentiates talk-to (SPRITE) from enter (Warp) and reports the stall."""

    def test_hint_labels_talk_and_enter(self):
        hint = render_sweep_hint(_TARGETS, stall_steps=42)
        assert "talk to SPRITE_GIRL_4 at (17, 9)" in hint
        assert "enter Warp→ViridianNicknameHouse at (21, 9)" in hint
        assert "42" in hint


class TestOverrideAction:
    """interaction_action emits the atomic high-level tool per interactable kind."""

    def test_npc_uses_interact_with_object(self):
        assert interaction_action(("SPRITE_GAMBLER_ASLEEP_5", 18, 9)) == (
            "use_tool(interact_with_object, (object_name='SPRITE_GAMBLER_ASLEEP_5'))"
        )

    def test_warp_uses_warp_with_warp_point(self):
        assert interaction_action(("Warp→ViridianNicknameHouse", 21, 9)) == (
            "use_tool(warp_with_warp_point, (x_dest=21, y_dest=9))"
        )


class TestUnifiedWiring:
    """The controller + per-episode reset are actually wired into the act loop."""

    @pytest.mark.parametrize(
        "needle",
        [
            "decide_interaction_sweep(",  # Phase 1/2 controller invoked
            "reset_interaction_sweep()",  # per-episode reset at the subgoal gate
            "interaction-sweep override",  # Phase 2 short-circuits the LLM
            "self._raw_cur_state = cur_state_str",  # raw obs stashed pre-preprocess
            "itargets_fn(raw_obs)",  # sweep parses the RAW grid, not the summary
            "record_milestone_step(raw_score)",  # milestone tracked via harness score
        ],
    )
    def test_sweep_is_wired(self, needle):
        assert needle in inspect.getsource(unified)

    def test_dead_text_score_parser_removed(self):
        # _extract_raw_score scanned the text obs for "Score:" — but the pokemon
        # obs never carries it, so it always returned 0. The milestone source is
        # now the harness score; the dead helper must be gone.
        assert "_extract_raw_score" not in inspect.getsource(unified)


class TestEpisodeResetAndPickle:
    """Stall counter follows the position-counter discipline: survives round-trips, resets per episode."""

    def test_stall_steps_survive_pickle_roundtrip(self, mem):
        mem.record_milestone_step(5)  # baseline
        for _ in range(20):
            mem.record_milestone_step(5)
        restored = pickle.loads(pickle.dumps(mem))
        assert restored.milestone_stall_steps == 20

    def test_reset_clears_stall_and_tried(self, mem):
        for _ in range(20):
            mem.record_milestone_step(5)
        mem.record_interaction_tried("ViridianCity", 17, 9)
        mem.reset_interaction_sweep()
        assert mem.milestone_stall_steps == 0
        assert mem.interaction_tried("ViridianCity") == set()

    def test_tried_set_is_map_keyed(self, mem):
        mem.record_interaction_tried("ViridianCity", 17, 9)
        assert mem.interaction_tried("ViridianCity") == {(17, 9)}
        assert mem.interaction_tried("PalletTown") == set()
