"""Game-specific configuration for the UnifiedMaclaAgent."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GameConfig:
    """Describes everything game-specific so UnifiedMaclaAgent can adapt."""

    game_name: str
    valid_actions: list[str]
    default_action: str
    default_goal: str
    system_prompt: str
    user_prompt_template: str
    action_schema_name: str  # lookup key in GAME_ACTION_SCHEMAS

    # Success detection
    score_pattern: str = r"Score:\s*(\d+)"
    progress_pattern: str | None = None
    progress_threshold: float = 2.0
    success_keywords: list[str] = field(default_factory=list)
    fatal_keywords: list[str] = field(default_factory=list)
    lives_pattern: str | None = None

    # Context extraction
    context_extraction_mode: str = "dict_fields"  # "regex_spatial" | "dict_fields" | "geometric"
    context_fields: dict[str, Any] = field(default_factory=dict)

    # Metrics
    metric_fields: list[str] = field(default_factory=list)


def load_game_config(path: str | Path) -> GameConfig:
    """Load a GameConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return GameConfig(**data)
