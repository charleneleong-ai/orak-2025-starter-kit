"""Game-specific adapter for 2048 — used by UnifiedMaclaAgent."""

from pydantic import BaseModel, Field

from agents.twenty_fourty_eight._metrics import normalize_2048_score


class TwentyFortyEightAction(BaseModel):
    reasoning: str = Field(description="Explanation of why this action was chosen")
    action: str = Field(description="The action: up, down, left, or right")


VALID_ACTIONS = ["up", "down", "left", "right"]
DEFAULT_ACTION = "left"
DEFAULT_GOAL = "Merge tiles to create the 2048 tile and maximize score."

SCORE_PATTERN = r"[Ss]core:?\s*(\d+)"
PROGRESS_PATTERN = None
PROGRESS_THRESHOLD = 0.0
LIVES_PATTERN = None
SUCCESS_KEYWORDS: list[str] = []
FATAL_KEYWORDS: list[str] = []

METRIC_FIELDS = ["score", "max_tile"]

CONTEXT_EXTRACTION_MODE = "strategic_grid"
CONTEXT_FIELDS = {
    "score_pattern": r"Score:\s*(\d+)",
    # Match the 4×4 board literal in the observation (any of:
    # [[a, b, c, d], [...], [...], [...]] format). Falls back to default
    # in StrategicGridExtractor if absent.
    "grid_pattern": (
        r"\[\[\s*\d+(?:\s*,\s*\d+){3}\s*\][\s,]*"
        r"\[\s*\d+(?:\s*,\s*\d+){3}\s*\][\s,]*"
        r"\[\s*\d+(?:\s*,\s*\d+){3}\s*\][\s,]*"
        r"\[\s*\d+(?:\s*,\s*\d+){3}\s*\]"
    ),
    "grid_size": 4,
}


def extract_action(result: TwentyFortyEightAction) -> str:
    return result.action


def calculate_metrics(game_info: dict) -> dict:
    current_game_score = int(float(game_info.get("score", 0)))
    try:
        max_tile = int(game_info.get("max_tile", 0))
    except (ValueError, TypeError):
        max_tile = 0
    return {
        "evaluation_score": normalize_2048_score(max_tile),
        "max_tile": max_tile,
        "score": current_game_score,
    }
