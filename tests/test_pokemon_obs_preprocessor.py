"""Tests for the Pokemon obs preprocessor wired into UnifiedMaclaAgent.

The legacy ``PokemonRedMaclaAgent`` (``agents/pokemon_red/base.py``) maintained
``_map_memory`` across steps and called ``replace_map_on_screen_with_full_map``
on every observation so the LLM saw the full explored map, not just the
screen window the env emits.

That pipeline was lost when ``UnifiedMaclaAgent`` was refactored to be
game-agnostic. These tests pin the contract that the pokemon game adapter
now exposes an ``ObservationPreprocessor`` factory which the unified agent
instantiates per-agent and calls per-step.
"""

from __future__ import annotations

import textwrap


def _sample_obs(map_name: str, pos_xy: tuple[int, int], map_block: str) -> str:
    """Build a minimal-but-realistic raw obs string. Mirrors the format
    that ``run.py`` feeds the agent (see logs/raw_requests.jsonl from any
    pr31 run)."""
    x, y = pos_xy
    return textwrap.dedent(
        f"""
        [Map Info]
        Map Name: {map_name}, (x_max , y_max): (7, 7)
        Map type: reds_house
        Expansion direction: $00
        Your position (x, y): ({x}, {y})
        Your facing direction: up
        Action instruction
         - up: (x, y) -> (x, y-1)

        Map on Screen:
        {map_block.strip()}

        """
    ).lstrip()


# Top half of RedsHouse1f visible when agent stands at (7,1): rows 0-5 only.
_TRUNCATED_TOP_HALF = """
( 3,  0): X	( 4,  0): X	( 5,  0): X	( 6,  0): X	( 7,  0): X
( 3,  1): SIGN_REDSHOUSE1F_TV	( 4,  1): O	( 5,  1): O	( 6,  1): O	( 7,  1): Warp→RedsHouse2f
( 3,  2): O	( 4,  2): O	( 5,  2): O	( 6,  2): O	( 7,  2): O
( 3,  3): O	( 4,  3): O	( 5,  3): O	( 6,  3): O	( 7,  3): O
( 3,  4): X	( 4,  4): X	( 5,  4): OBJ_1_1	( 6,  4): O	( 7,  4): O
( 3,  5): X	( 4,  5): X	( 5,  5): O	( 6,  5): O	( 7,  5): O
"""

# Bottom half of RedsHouse1f visible after agent moves down — rows 4-7
# expose the exit door at (3,7)/(2,7) labelled Warp→PalletTown.
_TRUNCATED_BOTTOM_HALF = """
( 3,  4): X	( 4,  4): X	( 5,  4): O	( 6,  4): O	( 7,  4): O
( 3,  5): X	( 4,  5): X	( 5,  5): O	( 6,  5): O	( 7,  5): O
( 3,  6): O	( 4,  6): O	( 5,  6): O	( 6,  6): X	( 7,  6): O
( 2,  7): Warp→PalletTown	( 3,  7): Warp→PalletTown	( 4,  7): O	( 5,  7): O	( 6,  7): X	( 7,  7): O
"""


# ── factory + contract ──────────────────────────────────────────────


def test_make_observation_preprocessor_factory_exists():
    """Pokemon adapter exports a ``make_observation_preprocessor`` factory."""
    from agents.pokemon_red import game_adapter

    factory = getattr(game_adapter, "make_observation_preprocessor", None)
    assert factory is not None, (
        "agents/pokemon_red/game_adapter.py must export make_observation_preprocessor()"
    )
    p = factory()
    assert hasattr(p, "preprocess")


def test_preprocessor_no_op_when_obs_has_no_map_section():
    """Plain text obs (no Map Info block) round-trips unchanged."""
    from agents.pokemon_red import game_adapter

    p = game_adapter.make_observation_preprocessor()
    obs = "Score: 0\nNothing to see here."
    assert p.preprocess(obs) == obs


# ── single-step expansion ──────────────────────────────────────────


def test_preprocessor_replaces_truncated_screen_with_full_map_marker():
    """First call must rewrite 'Map on Screen:' to a full-map block."""
    from agents.pokemon_red import game_adapter

    p = game_adapter.make_observation_preprocessor()
    obs = _sample_obs("RedsHouse1f", (7, 1), _TRUNCATED_TOP_HALF)
    out = p.preprocess(obs)
    # replace_map_on_screen_with_full_map swaps the truncated 'Map on
    # Screen:' block for a labelled '[Full Map]' grid + a separate
    # '[Notable Objects]' section.
    assert "[Full Map]" in out
    assert "Map on Screen:" not in out
    # Off-screen rows (y >= 6) must be filled with the unexplored
    # placeholder so the LLM sees the full grid shape.
    assert "?" in out


# ── multi-step memory accumulation ──────────────────────────────────


def test_preprocessor_accumulates_explored_map_across_calls():
    """Second obs (different screen window of same map) merges with the
    first — agent sees BOTH the staircase from step 1 AND the exit door
    from step 2 in the second call's output."""
    from agents.pokemon_red import game_adapter

    p = game_adapter.make_observation_preprocessor()
    p.preprocess(_sample_obs("RedsHouse1f", (7, 1), _TRUNCATED_TOP_HALF))
    out = p.preprocess(_sample_obs("RedsHouse1f", (3, 6), _TRUNCATED_BOTTOM_HALF))
    # Both warps must be in the merged map.
    assert "RedsHouse2f" in out, "Lost the staircase tile from step 1"
    assert "PalletTown" in out, "Missing the exit door from step 2"


def test_preprocessor_isolates_state_across_maps():
    """A second map's exploration doesn't leak into the first map's grid."""
    from agents.pokemon_red import game_adapter

    p = game_adapter.make_observation_preprocessor()
    p.preprocess(_sample_obs("RedsHouse1f", (7, 1), _TRUNCATED_TOP_HALF))
    p.preprocess(_sample_obs("RedsHouse2f", (5, 4), _TRUNCATED_TOP_HALF))
    # Memory must be keyed per-map: state on disk should have both keys
    assert "RedsHouse1f" in p._map_memory
    assert "RedsHouse2f" in p._map_memory


# ── failure modes degrade gracefully ────────────────────────────────


def test_preprocessor_returns_raw_obs_on_parse_failure():
    """Bad / partial obs returns the original string instead of raising."""
    from agents.pokemon_red import game_adapter

    p = game_adapter.make_observation_preprocessor()
    out = p.preprocess("[Map Info]\nMap Name: foo, (x_max , y_max): broken")
    # Doesn't raise; returns *some* string (raw or partially-rewritten).
    assert isinstance(out, str)


# ── UnifiedMaclaAgent wiring (contract tests, no live LLM) ──────────


def test_other_game_adapters_do_not_export_factory():
    """Mario / 2048 don't have a viewport-truncation problem so they
    don't expose the factory — UnifiedMaclaAgent's getattr-with-default
    pattern degrades gracefully for them."""
    from agents.super_mario import game_adapter as mario
    from agents.twenty_fourty_eight import game_adapter as tfe

    assert getattr(mario, "make_observation_preprocessor", None) is None
    assert getattr(tfe, "make_observation_preprocessor", None) is None


def test_unified_agent_init_wires_obs_preprocessor():
    """``UnifiedMaclaAgent.__init__`` reads the adapter's factory and
    stores the result on ``self._obs_preprocessor`` for ``_get_action``
    to call per step."""
    import inspect

    from agents.macla import unified

    init_src = inspect.getsource(unified.UnifiedMaclaAgent.__init__)
    get_action_src = inspect.getsource(unified.UnifiedMaclaAgent._get_action)

    assert "make_observation_preprocessor" in init_src, (
        "UnifiedMaclaAgent.__init__ must read self._adapter.make_observation_preprocessor"
    )
    assert "_obs_preprocessor" in init_src, (
        "UnifiedMaclaAgent.__init__ must store the preprocessor on self"
    )
    assert "_obs_preprocessor" in get_action_src, (
        "UnifiedMaclaAgent._get_action must call the preprocessor on cur_state_str"
    )
