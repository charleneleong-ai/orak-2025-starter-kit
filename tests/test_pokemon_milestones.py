"""Unit tests for the Pokemon Red milestone evaluator.

Targets the ``PokemonRedEnv.evaluate`` state machine — specifically the
cutscene-aware fix to milestone 1->2. The original branch checked
``'SPRITE_OAK' in map_screen_raw``, which goes blind when Oak's intro
replaces the tile grid with a textbox. The fix ORs that with a
RAM-derived ``'OaksLab' in map_name`` signal so the milestone fires once
the cutscene has provably warped the player into Oak's lab.

We don't construct a full ``PokemonRedEnv`` (constructor starts a PyBoy
thread). Instead we drive ``evaluate`` directly against a minimal
state-dict harness.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

# Stub the mcp_game_servers package so loading pokemon_red_env doesn't
# trigger the runtime-only base_env / utils imports.
_REPO = Path(__file__).resolve().parent.parent

_pkg_root = types.ModuleType("mcp_game_servers")
_pkg_root.__path__ = []
sys.modules.setdefault("mcp_game_servers", _pkg_root)

_base_env_mod = types.ModuleType("mcp_game_servers.base_env")


class _BaseEnv:
    def __init__(self, *args, **kwargs):
        pass


_base_env_mod.BaseEnv = _BaseEnv
sys.modules.setdefault("mcp_game_servers.base_env", _base_env_mod)

_types_pkg = types.ModuleType("mcp_game_servers.utils")
_types_pkg.__path__ = []
sys.modules.setdefault("mcp_game_servers.utils", _types_pkg)
_types_inner = types.ModuleType("mcp_game_servers.utils.types")
_types_inner.__path__ = []
sys.modules.setdefault("mcp_game_servers.utils.types", _types_inner)

_game_io = types.ModuleType("mcp_game_servers.utils.types.game_io")


class _Action:
    pass


class _Obs:
    pass


_game_io.Action = _Action
_game_io.Obs = _Obs
sys.modules.setdefault("mcp_game_servers.utils.types.game_io", _game_io)

# Stub the pokemon_red game subpackage so importing pokemon_red_env doesn't
# also pull in PyBoyRunner / pokemon_tools / map_utils (each of which has
# its own runtime-only deps).
_pokemon_pkg = types.ModuleType("mcp_game_servers.pokemon_red")
_pokemon_pkg.__path__ = []
sys.modules.setdefault("mcp_game_servers.pokemon_red", _pokemon_pkg)
_pokemon_game_pkg = types.ModuleType("mcp_game_servers.pokemon_red.game")
_pokemon_game_pkg.__path__ = []
sys.modules.setdefault("mcp_game_servers.pokemon_red.game", _pokemon_game_pkg)

_runner_stub = types.ModuleType("mcp_game_servers.pokemon_red.game.pyboy_runner")


class _PyBoyRunner:
    def __init__(self, *_args, **_kwargs):
        pass


_runner_stub.PyBoyRunner = _PyBoyRunner
sys.modules.setdefault("mcp_game_servers.pokemon_red.game.pyboy_runner", _runner_stub)

_utils_stub = types.ModuleType("mcp_game_servers.pokemon_red.game.utils")
_utils_stub.__path__ = []
sys.modules.setdefault("mcp_game_servers.pokemon_red.game.utils", _utils_stub)
_tools_stub = types.ModuleType("mcp_game_servers.pokemon_red.game.utils.pokemon_tools")
_tools_stub.PokemonToolset = object
_tools_stub.execute_action_response = lambda *a, **kw: None
sys.modules.setdefault("mcp_game_servers.pokemon_red.game.utils.pokemon_tools", _tools_stub)
_map_utils_stub = types.ModuleType("mcp_game_servers.pokemon_red.game.utils.map_utils")
_map_utils_stub.construct_init_map = lambda *a, **kw: ""
_map_utils_stub.refine_current_map = lambda *a, **kw: ""
sys.modules.setdefault("mcp_game_servers.pokemon_red.game.utils.map_utils", _map_utils_stub)

_ENV_PATH = _REPO / "evaluation_utils/mcp_game_servers/pokemon_red/game/pokemon_red_env.py"
_spec = importlib.util.spec_from_file_location("pokemon_red_env_under_test", _ENV_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
PokemonRedEnv = _module.PokemonRedEnv


class _MilestoneHarness:
    """Bare-minimum stand-in for PokemonRedEnv used to exercise evaluate()."""

    def __init__(self) -> None:
        self.score = 0
        self.prev_state_dict: dict = {}
        self.state_dict: dict = {}
        # evaluate() reads runner.quit_flag for the terminal-state check.
        self.runner = type("R", (), {"quit_flag": False})()

    def step(self, prev: dict, curr: dict) -> int:
        self.prev_state_dict = prev
        self.state_dict = curr
        # Bind the unbound method so it operates on this harness.
        PokemonRedEnv.evaluate(self, obs=None)
        return self.score


def _state(
    *,
    map_name: str = "RedsHouse1f",
    map_screen_raw: str = "",
    state: str = "Field",
    your_party: str = "",
    inventory: str = "",
) -> dict:
    return {
        "state": state,
        "your_party": your_party,
        "inventory": inventory,
        "map_info": {
            "map_name": map_name,
            "map_screen_raw": map_screen_raw,
        },
    }


def test_milestone_0_to_1_fires_on_leaving_redshouse():
    h = _MilestoneHarness()
    assert h.step(_state(map_name="RedsHouse1f"), _state(map_name="PalletTown")) == 1


def test_milestone_0_to_1_skips_internal_redshouse_warp():
    h = _MilestoneHarness()
    assert h.step(_state(map_name="RedsHouse2f"), _state(map_name="RedsHouse1f")) == 0


def test_milestone_1_to_2_via_sprite_oak_on_route():
    h = _MilestoneHarness()
    h.score = 1
    curr = _state(map_name="Route1", map_screen_raw="...SPRITE_OAK...")
    assert h.step(_state(map_name="PalletTown"), curr) == 2


def test_milestone_1_to_2_via_oakslab_during_cutscene():
    """The fix: when Oak's intro hides the tile grid, OaksLab map_name fires."""
    h = _MilestoneHarness()
    h.score = 1
    # map_screen_raw is None — the cutscene replaced it with a textbox.
    curr = _state(map_name="OaksLab", map_screen_raw="")
    curr["map_info"]["map_screen_raw"] = None
    assert h.step(_state(map_name="PalletTown"), curr) == 2


def test_milestone_1_to_2_does_not_fire_in_palettown_without_oak():
    h = _MilestoneHarness()
    h.score = 1
    curr = _state(map_name="PalletTown", map_screen_raw="...trees...")
    assert h.step(_state(map_name="RedsHouse1f"), curr) == 1


def test_milestone_2_to_3_fires_on_party_named():
    h = _MilestoneHarness()
    h.score = 2
    curr = _state(map_name="OaksLab", your_party="Name: BULBASAUR\nLevel: 5")
    assert h.step(_state(map_name="OaksLab"), curr) == 3


def test_milestone_3_to_4_fires_on_battle_exit():
    h = _MilestoneHarness()
    h.score = 3
    prev = _state(map_name="Route1", state="WildBattle")
    curr = _state(map_name="Route1", state="Field")
    assert h.step(prev, curr) == 4


def test_milestone_4_to_5_fires_on_reaching_viridian():
    h = _MilestoneHarness()
    h.score = 4
    assert h.step(_state(map_name="Route1"), _state(map_name="ViridianCity")) == 5


def test_milestone_5_to_6_fires_on_parcel_acquired():
    h = _MilestoneHarness()
    h.score = 5
    curr = _state(map_name="ViridianMart", inventory="- OAK's PARCEL × 1")
    assert h.step(_state(map_name="ViridianMart"), curr) == 6


def test_milestone_6_to_7_fires_on_parcel_delivered():
    h = _MilestoneHarness()
    h.score = 6
    prev = _state(map_name="OaksLab", inventory="- OAK's PARCEL × 1")
    curr = _state(map_name="OaksLab", inventory="")
    assert h.step(prev, curr) == 7
