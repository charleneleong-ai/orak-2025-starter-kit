"""Game-specific adapter for Pokemon Red — used by UnifiedMaclaAgent."""

import re

from loguru import logger
from pydantic import BaseModel, Field

# Re-exports read by UnifiedMaclaAgent via self._adapter.<NAME>.
from agents.pokemon_red.base import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: F401
from agents.pokemon_red.openai_pokemon_memory_utils import (
    get_map_memory_dict,
    parse_game_state,
    replace_map_on_screen_with_full_map,
)

# Compiled once for the per-step lookup in extract_loop_state. Mirrors the
# regexes in pokemon_red/macla.py so the LoopDetector wiring works for both
# UnifiedMaclaAgent (uses this adapter) and PokemonRedMaclaAgent (uses its
# own override) without requiring a refactor.
_LOOP_MAP_RE = re.compile(r"Map Name:\s*([^,\s]+)")
_LOOP_POS_RE = re.compile(r"Your position \(x, y\):\s*\((\d+),\s*(\d+)\)")


class PokemonAction(BaseModel):
    reasoning: str = Field(description="Detailed explanation of why this action was chosen")
    action: str = Field(
        description="The action to take. PREFER high-level tool actions like use_tool(move_to, (x_dest=X, y_dest=Y)), use_tool(interact_with_object, (object_name='NAME')), use_tool(continue_dialog, ()). Only use low-level buttons (up, down, left, right, a, b, start, select) if no tool applies."
    )
    current_goal: str | None = Field(
        default=None, description="Inferred next milestone or sub-goal"
    )


VALID_ACTIONS = ["up", "down", "left", "right", "a", "b", "start", "select"]
DEFAULT_ACTION = "a"
DEFAULT_GOAL = "Become the Pokemon Champion by progressing through the storyline."

SCORE_PATTERN = r"[Ss]core:?\s*(\d+)"
PROGRESS_PATTERN = None
PROGRESS_THRESHOLD = 0.0
LIVES_PATTERN = None
SUCCESS_KEYWORDS = [
    "Badge obtained",
    "defeated",
    "Level up",
    "evolved",
    "learned",
    "caught",
    "received",
    "Thank you",
]
FATAL_KEYWORDS: list[str] = []

METRIC_FIELDS = ["badges", "level", "team_size", "score"]

CONTEXT_EXTRACTION_MODE = "dict_fields"
CONTEXT_FIELDS = {
    "fields": [
        {"name": "map_name", "pattern": r"Map Name:\s*([^\s,]+)", "type": "str"},
        {
            "name": "position",
            "pattern": r"Your position \(x, y\): \((\d+), (\d+)\)",
            "type": "tuple",
        },
        {"name": "facing", "pattern": r"Your facing direction:\s*(\w+)", "type": "str"},
    ],
}


def extract_action(result: PokemonAction) -> str:
    return result.action


def extract_loop_state(obs: dict) -> tuple | None:
    """Lift ``(map, x, y)`` out of pokemon obs for the LoopDetector.

    UnifiedMaclaAgent's adapter dispatch reads this; mirrors the
    method on PokemonRedMaclaAgent so both agent classes feed the
    detector identically. Returns ``None`` during battles/menus
    where ``[Map Info]`` is replaced by a textbox.
    """
    text = obs.get("obs_str", "")
    if not text:
        return None
    m_map = _LOOP_MAP_RE.search(text)
    m_pos = _LOOP_POS_RE.search(text)
    if not m_map or not m_pos:
        return None
    return (m_map.group(1), int(m_pos.group(1)), int(m_pos.group(2)))


class PokemonObservationPreprocessor:
    """Maintains per-agent map exploration memory and expands the env's
    screen-window 'Map on Screen' block to the full explored map.

    Pokemon Red emits a viewport ~5 columns × 6 rows centred on the
    player, so tiles outside that window — including the RedsHouse1f
    exit door at (3, 7) when the agent stands at the top of the map —
    are invisible to the LLM. Without an off-screen memory layer the
    agent can be unable to even consider an action whose target is
    physically off the obs grid.

    Mirrors the legacy ``PokemonRedMaclaAgent._preprocess_observation``
    pipeline (``agents/pokemon_red/base.py``) so ``UnifiedMaclaAgent``
    runs see the same expanded obs as the legacy entrypoint.
    """

    def __init__(self) -> None:
        self._map_memory: dict = {}

    def preprocess(self, obs_str: str) -> str:
        try:
            parsed = parse_game_state(obs_str)
            map_info = parsed.get("map_info") or {}
            map_name = map_info.get("map_name")
            if not map_name or map_info.get("x_max") is None:
                return obs_str
            self._map_memory = get_map_memory_dict(parsed, self._map_memory)
            map_current = self._map_memory.get(map_name, {}).get("explored_map", [])
            if not map_current:
                return obs_str
            return replace_map_on_screen_with_full_map(obs_str, map_current, {})
        except Exception as e:
            logger.warning(f"[pokemon obs preprocessor] failed; returning raw obs: {e}")
            return obs_str


def make_observation_preprocessor() -> PokemonObservationPreprocessor:
    """Factory hook read by ``UnifiedMaclaAgent`` — instantiate per agent."""
    return PokemonObservationPreprocessor()


def calculate_metrics(game_info: dict) -> dict:
    metrics = {}
    for key in METRIC_FIELDS:
        if key in game_info:
            metrics[key] = game_info[key]
    if "evaluation_score" in game_info:
        metrics["evaluation_score"] = float(game_info["evaluation_score"])
    else:
        raw_flags = float(game_info.get("score", 0))
        metrics["evaluation_score"] = (raw_flags / 7.0) * 100.0
    metrics["score"] = float(game_info.get("score", 0))
    return metrics


# ── Self-reflection recommendation (PR #64 cross-game retro) ──────────
# Cross-game test (n=1, 300 steps) found self-reflection on pokemon enables
# deeper stuck-state recoveries (Charmander naming + trainer-battle progression)
# even when the headline score ties the Stage D baseline. Keep enabled for
# long-horizon dialog-heavy progression.
RECOMMENDED_USE_SELF_REFLECTION = True
RECOMMENDED_REFLECTION_EVERY = 10


# ── Map-graph hint (Stage P/Q generalisable adapter surface) ──────────
# UnifiedMaclaAgent reads this via ``getattr(self._adapter, 'graph_hint',
# None)``. Games without spatial navigation (2048, currently mario) don't
# export the symbol — getattr returns None and the planner sees no hint.
#
# Stage Q (2026-05-17): extends the Stage P hint with per-neighbour exit
# tile coordinates. The Stage P n=5 verdict (FLAT, 57.14% × 5) confirmed
# the planner consumed the map names but couldn't translate them into a
# move_to(x, y) call — the agent stalled at Route 1 (10, 35) every iter,
# never finding the north-edge transition tile to Viridian. The exit-tile
# hint surfaces that coordinate directly:
#
#     ### Exit tiles
#       → OaksLab: walk to (12, 11)
#       → Route1: walk off the north edge
#
# Indoor warps render as ``walk to (x, y)`` from the warp_event tile;
# outdoor connections render as ``walk off the <direction> edge`` from
# the header connection. Both auto-extracted from pokered .asm.
from pathlib import Path  # noqa: E402

from agents.macla.pokered_map_extractor import build_exit_tiles, build_map_graph  # noqa: E402

# pokered repo path matches the runtime hard-fail location.
_POKERED = Path("evaluation_utils/mcp_game_servers/pokemon_red/game/pokered")


def _load_graph_and_exits() -> tuple[dict[str, set[str]], dict]:
    """Load auto-extracted MAP_GRAPH + EXIT_TILES once at module import.

    Falls back to empty dicts if pokered isn't present (e.g. CI without
    the submodule symlinked); the hint then just returns None and the
    planner sees no graph block — safe degradation.
    """
    if not (_POKERED / "data/maps/headers").is_dir():
        return {}, {}
    return build_map_graph(_POKERED), build_exit_tiles(_POKERED)


_MAP_GRAPH, _EXIT_TILES = _load_graph_and_exits()


def _render_exit(target: str, info) -> str:
    """Render a single exit-tile line. Tuple = indoor coord; str = direction."""
    if isinstance(info, tuple):
        x, y = info
        return f"  → {target}: walk to ({x}, {y})"
    return f"  → {target}: walk off the {info} edge"


def graph_hint(current_map: str | None, visited_maps: set[str]) -> str | None:
    """Render the Stage Q map-graph + exit-tile hint for pokemon_red.

    Extends the Stage P hint with a per-unvisited-neighbour exit-tile
    section. Returns None when there's nothing useful to say (unknown
    map, outside the graph, no neighbours and nothing visited).
    """
    if not current_map or current_map == "unknown":
        return None
    if current_map not in _MAP_GRAPH:
        return None
    neighbours = _MAP_GRAPH[current_map]
    unvisited = sorted(n for n in neighbours if n not in visited_maps)
    visited_sorted = sorted(visited_maps)
    if not unvisited and not visited_sorted:
        return None

    lines = ["### Map graph"]
    if unvisited:
        lines.append(f"Unvisited maps reachable from {current_map}: " + ", ".join(unvisited))
    if visited_sorted:
        lines.append(f"Visited so far ({len(visited_sorted)}): " + ", ".join(visited_sorted))

    exit_lines = [
        _render_exit(n, _EXIT_TILES[(current_map, n)])
        for n in unvisited
        if (current_map, n) in _EXIT_TILES
    ]
    if exit_lines:
        lines.append("")
        lines.append("### Exit tiles")
        lines.extend(exit_lines)

    return "\n".join(lines)
