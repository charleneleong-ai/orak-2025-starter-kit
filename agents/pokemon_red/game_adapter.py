"""Game-specific adapter for Pokemon Red — used by UnifiedMaclaAgent."""
import re
from typing import Optional

from pydantic import BaseModel, Field

from agents._cognitive import Milestone
from agents.pokemon_red.base import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class PokemonAction(BaseModel):
    reasoning: str = Field(description="Detailed explanation of why this action was chosen")
    action: str = Field(description="The action to take. PREFER high-level tool actions like use_tool(move_to, (x_dest=X, y_dest=Y)), use_tool(interact_with_object, (object_name='NAME')), use_tool(continue_dialog, ()). Only use low-level buttons (up, down, left, right, a, b, start, select) if no tool applies.")
    current_goal: Optional[str] = Field(default=None, description="Inferred next milestone or sub-goal")


VALID_ACTIONS = ["up", "down", "left", "right", "a", "b", "start", "select"]
DEFAULT_ACTION = "a"
DEFAULT_GOAL = "Become the Pokemon Champion by progressing through the storyline."

SCORE_PATTERN = r"[Ss]core:?\s*(\d+)"
PROGRESS_PATTERN = None
PROGRESS_THRESHOLD = 0.0
LIVES_PATTERN = None
SUCCESS_KEYWORDS = ["Badge obtained", "defeated", "Level up", "evolved", "learned", "caught", "received", "Thank you"]
FATAL_KEYWORDS: list[str] = []

METRIC_FIELDS = ["badges", "level", "team_size", "score"]

CONTEXT_EXTRACTION_MODE = "dict_fields"
CONTEXT_FIELDS = {
    "fields": [
        {"name": "map_name", "pattern": r"Map Name:\s*([^\s,]+)", "type": "str"},
        {"name": "position", "pattern": r"Your position \(x, y\): \((\d+), (\d+)\)", "type": "tuple"},
        {"name": "facing", "pattern": r"Your facing direction:\s*(\w+)", "type": "str"},
    ],
}


def extract_action(result: PokemonAction) -> str:
    return result.action


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


# ── Waypoint curriculum for the planner's `goal=` argument ────────────
#
# Pokemon's static DEFAULT_GOAL ("Become Pokémon Champion") is too abstract
# for the LLM subtask planner to decompose well — PR #28 Stage D evidence:
# the planner generated abstract subgoals ("explore upstairs") that trapped
# the agent in the starter house (0/4 maps reached vs Stage A's 4/4).
#
# WAYPOINTS replace the abstract goal with an ordered curriculum of
# concrete milestones. WaypointGoalProvider walks the list and emits the
# first unmet milestone as the planner's goal, so the planner gets a
# state-aware concrete sub-goal at every step rather than having to
# re-derive "which milestone applies" from the observation each time.
#
# Layered with the abstract default planner system prompt (PR #31):
# WAYPOINTS data is the *what* (milestone selection, code-driven), the
# planner's heuristics are the *how* (action-level decomposition,
# LLM-driven). Together: waypoint provider picks the milestone, planner
# uses exploration heuristics to expand it into a step-level subtask.
# Pokemon adapter is the only game opted in here — mario / 2048 don't
# need this (their substrates work without explicit curriculum).
_MAP_NAME_RE = re.compile(r"Map Name:\s*([A-Za-z0-9_]+)")


def _current_map(observation: str) -> str:
    m = _MAP_NAME_RE.search(observation)
    return m.group(1) if m else ""


def _party_empty(observation: str) -> bool:
    return "No more Pokemons" in observation or "No more Pokemon" in observation


WAYPOINTS: list[Milestone] = [
    Milestone(
        name="exit_starter_house",
        goal=(
            "Walk south to leave Red's House. Staircases and exit doors are both "
            "labelled WarpPoint, but the exit door is always the bottom-edge "
            "WarpPoint of RedsHouse1f (largest y). Use the staircase from 2f to "
            "reach 1f, then walk to that bottom WarpPoint."
        ),
        precondition=lambda obs: _current_map(obs) in {"RedsHouse2f", "RedsHouse1f"},
    ),
    Milestone(
        name="reach_oaks_lab",
        goal=(
            "Walk south through PalletTown to enter Oak's Lab. Oak's Lab is the "
            "building at the bottom of PalletTown."
        ),
        precondition=lambda obs: _current_map(obs) == "PalletTown" and _party_empty(obs),
    ),
    Milestone(
        name="get_starter_pokemon",
        goal=(
            "Approach Professor Oak in OaksLab and interact with him; then "
            "interact with a Poké Ball on the table to receive a starter Pokémon."
        ),
        precondition=lambda obs: _current_map(obs) == "OaksLab" and _party_empty(obs),
    ),
    Milestone(
        name="defeat_rival",
        goal="Win the first rival battle inside Oak's Lab.",
        precondition=lambda obs: _current_map(obs) == "OaksLab" and not _party_empty(obs),
    ),
    Milestone(
        name="enter_route_1",
        goal="Walk north out of PalletTown onto Route 1.",
        precondition=lambda obs: _current_map(obs) == "PalletTown" and not _party_empty(obs),
    ),
    Milestone(
        name="reach_viridian_city",
        goal="Walk north through Route 1 to enter Viridian City.",
        precondition=lambda obs: _current_map(obs) == "Route1",
    ),
]
