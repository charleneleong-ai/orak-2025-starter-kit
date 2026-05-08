"""Game-specific adapter for Pokemon Red — used by UnifiedMaclaAgent."""

import re

from pydantic import BaseModel, Field

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
