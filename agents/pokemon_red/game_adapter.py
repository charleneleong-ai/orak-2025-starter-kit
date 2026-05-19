"""Game-specific adapter for Pokemon Red — used by UnifiedMaclaAgent."""

import json
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


# ── Trajectory introspection adapter (introspect) ─────────
# Consumed by `introspect --adapter agents.pokemon_red.game_adapter`.
# Mario / 2048 ship their own equivalent blocks when they need introspection.

from autoresearch.trajectory import ActionSpec, DwellSpec, MilestoneSpec  # noqa: E402

_MOVE_TO_RE = re.compile(r"move_to[^()]*\(\s*x_dest=(-?\d+)\s*,\s*y_dest=(-?\d+)\s*\)")


def _traj_score(row: dict) -> float:
    gi = row.get("obs", {}).get("game_info", {})
    try:
        return float(int(gi.get("score", 0)))
    except (TypeError, ValueError):
        return 0.0


def _traj_zone(row: dict) -> str:
    return row.get("obs", {}).get("game_info", {}).get("map_name", "?") or "?"


def _traj_move_target(row: dict) -> tuple[int, int] | None:
    s = row.get("action", "")
    if not isinstance(s, str):
        s = json.dumps(s)
    m = _MOVE_TO_RE.search(s)
    return (int(m.group(1)), int(m.group(2))) if m else None


TRAJECTORY_MILESTONES: list[MilestoneSpec] = [
    MilestoneSpec(f"M{i}", lambda row, i=i: _traj_score(row) >= i) for i in range(1, 8)
]
TRAJECTORY_DWELL_SPECS: list[DwellSpec] = [
    DwellSpec("Route1", lambda row: _traj_zone(row) == "Route1"),
    DwellSpec("Viridian", lambda row: "Viridian" in _traj_zone(row)),
]
TRAJECTORY_ACTION_SPEC = ActionSpec(extract_target=_traj_move_target)
TRAJECTORY_SCORE_EXTRACTOR = _traj_score
TRAJECTORY_ZONE_EXTRACTOR = _traj_zone
TRAJECTORY_SCORE_MAX = 7.0

# Stage Q2: minimum raw score an iter must reach for its procedures to
# survive the next iter's checkpoint-load prune. M4 (4/7 — Charmander
# nickname dialog crossed) is the gate that distinguishes a productive
# iter from a stuck-in-Pallet-Town iter (the Stage Q n=5 failure mode).
PROC_CACHE_MIN_ITER_SCORE = 4.0


# ── hierarchical subgoal templates ──────────────────────────────────
# Each subgoal carries an explicit completion predicate. The planner
# emits a stack of these; per step the executor checks the top
# subgoal's completion(obs) and pops on fire.
#
# Predicates must be picklable (existing checkpoint code uses pickle),
# so we use module-level functions + functools.partial — not lambdas.
from functools import partial as _partial  # noqa: E402

from agents.macla.macla_lib import (  # noqa: E402
    Subgoal,
    build_score_milestone_stack,
    completes_when_score_at_least,
    make_score_milestone_subgoal,
)


def _completes_when_map_is(target: str, obs: dict) -> bool:
    return obs.get("map_name") == target


def _completes_when_dialog_mentions(npc: str, obs: dict) -> bool:
    return npc in (obs.get("recent_dialog", "") or "")


def _navigate_to_map(target: str) -> Subgoal:
    return Subgoal(
        name=f"NavigateToMap({target})",
        description=f"Walk until the current map is {target}.",
        completion=_partial(_completes_when_map_is, target),
        suggested_tools=["move_to"],
    )


def _talk_to(npc: str) -> Subgoal:
    return Subgoal(
        name=f"TalkTo({npc})",
        description=f"Interact with {npc} (dialog should mention them).",
        completion=_partial(_completes_when_dialog_mentions, npc),
        suggested_tools=["interact_with_object", "continue_dialog", "a"],
    )


def _defeat_trainer(trainer: str, score_after: int) -> Subgoal:
    return Subgoal(
        name=f"DefeatTrainer({trainer})",
        description=f"Win the battle vs {trainer} — score should reach {score_after}.",
        completion=_partial(completes_when_score_at_least, score_after),
        suggested_tools=["a", "interact_with_object"],
    )


# Per-milestone registry: score-threshold -> (name, description, suggested_tools).
# Mirrors pokemon_red_env.py:276-304's 7-point ladder (M1..M7). M1-M4 are
# cutscene-paced; M5+ requires navigation, so only those land in the
# initial stack. Adding M8+ = one dict entry — no other edits.
_POKEMON_MILESTONE_LIBRARY: dict[int, tuple[str, str, list[str]]] = {
    5: (
        "EnterViridian",
        "Walk into Viridian City (any Viridian-named map). Route 1 leads "
        "directly north from Pallet Town.",
        ["move_to"],
    ),
    6: (
        "GetOaksParcel",
        "Pick up Oak's Parcel from the Viridian Mart — enter the Mart "
        "(blue-roofed building in Viridian City) and talk to the clerk "
        "at the counter.",
        ["move_to", "interact_with_object", "continue_dialog", "a"],
    ),
    7: (
        "DeliverOaksParcel",
        "Return to Pallet Town and deliver Oak's Parcel to Professor Oak "
        "in his lab (south side of Pallet Town).",
        ["move_to", "interact_with_object", "continue_dialog", "a"],
    ),
}


def _reach_pokemon_milestone(idx: int) -> Subgoal:
    """Thin pokemon-side wrapper over ``make_score_milestone_subgoal`` that
    looks up descriptive metadata from ``_POKEMON_MILESTONE_LIBRARY``."""
    name, description, tools = _POKEMON_MILESTONE_LIBRARY[idx]
    return make_score_milestone_subgoal(idx, name, description, tools)


SUBGOAL_TEMPLATES = {
    "NavigateToMap": _navigate_to_map,
    "TalkTo": _talk_to,
    "DefeatTrainer": _defeat_trainer,
    "ReachMilestone": _reach_pokemon_milestone,
}


def initial_subgoal_stack() -> list[Subgoal]:
    """Per-game default subgoal stack pushed at fresh-iter start.

    For pokemon the critical path through the 7-point env scoring ladder
    is: Route1 (top, immediate) → M5 EnterViridian → M6 GetOaksParcel →
    M7 DeliverOaksParcel (bottom). M1-M4 (leave RedsHouse → encounter Oak
    → get starter → win rival) are cutscene-paced and don't need stack
    entries; M5+ are the navigation-bound ones the agent has historically
    ceiling-bound on.

    Built via the generic ``build_score_milestone_stack`` framework helper
    so any other game with a monotone integer score can plug in by
    declaring its own ``MILESTONE_LIBRARY`` + preamble.
    """
    return build_score_milestone_stack(
        _POKEMON_MILESTONE_LIBRARY,
        preamble=[_navigate_to_map("Route1")],  # immediate next from Pallet
    )
