"""Game-specific adapter for Super Mario — used by UnifiedMaclaAgent."""

from pydantic import BaseModel, Field

# Re-exports read by UnifiedMaclaAgent via self._adapter.<NAME>.
from agents.super_mario.base import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: F401


class MarioAction(BaseModel):
    reasoning: str = Field(description="Explanation of why this action was chosen")
    jump_level: int = Field(description="The jump level: 0 to 6", ge=0, le=6)


VALID_ACTIONS = [f"Jump Level: {i}" for i in range(7)]
DEFAULT_ACTION = "Jump Level: 0"
DEFAULT_GOAL = (
    "Move right to reach the goal, gain points and power-ups and avoid obstacles and enemies."
)

SCORE_PATTERN = r"[Ss]core:?\s*(\d+)"
PROGRESS_PATTERN = r"x_pos:?\s*(\d+)"
PROGRESS_THRESHOLD = 2.0
LIVES_PATTERN = r"[Ll]ives:?\s*(\d+)"
SUCCESS_KEYWORDS: list[str] = []
FATAL_KEYWORDS: list[str] = []

METRIC_FIELDS = ["coins", "lives", "time", "world", "stage", "x_pos", "score"]

CONTEXT_EXTRACTION_MODE = "regex_spatial"
CONTEXT_FIELDS = {
    "player_position_pattern": r"Position of Mario:\s*\((\d+),\s*(\d+)\)",
    "entities": [
        {"keywords": ["pit"], "label": "pit"},
        {"keywords": ["monster goomba", "goomba", "enemy"], "label": "goomba"},
        {"keywords": ["monster koopas", "koopa", "turtle"], "label": "koopa"},
        {"keywords": ["warp pipe", "pipe"], "label": "pipe"},
        {"keywords": ["item mushrooms"], "label": "mushroom"},
        {"keywords": ["bricks"], "label": "brick"},
        {"keywords": ["question blocks"], "label": "question"},
        {"keywords": ["inactivated blocks"], "label": "block"},
    ],
    "distance_bins": {"near": 60, "mid": 140, "far": 180},
    "filter_behind": -20,
    "filter_ahead": 180,
}


def extract_action(result: MarioAction) -> str:
    return f"Jump Level: {result.jump_level}"


def calculate_metrics(game_info: dict) -> dict:
    metrics = {}
    for key in METRIC_FIELDS:
        if key in game_info:
            metrics[key] = game_info[key]
    if "evaluation_score" in game_info:
        metrics["evaluation_score"] = float(game_info["evaluation_score"])
    else:
        x_pos = float(game_info.get("x_pos", 40))
        x_start, x_flag = 40, 3161
        metrics["evaluation_score"] = (x_pos - x_start) / (x_flag - x_start) * 100.0
    metrics["score"] = float(game_info.get("score", 0))
    return metrics


# ── Self-reflection recommendation (PR #64 cross-game retro) ──────────
# Mario is reactive; cross-game test (n=1, 300 steps) showed self-reflection
# at reflect_every=10 ties Stage D baseline within noise (-0.38 pp), and
# critiques are game-aware ("Goomba at (189, 47), stop jumping in place").
# Keep enabled but sparser to amortise the extra LLM-call overhead.
RECOMMENDED_USE_SELF_REFLECTION = True
RECOMMENDED_REFLECTION_EVERY = 30
