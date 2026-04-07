"""
Config-driven context extractors for UnifiedMaclaAgent.

Three strategies matching the three games:
- RegexSpatialExtractor: Mario-style entity-relative-to-player positioning
- DictFieldExtractor: Pokemon Red-style named field extraction
- GeometricExtractor: 2048-style grid/density analysis
"""
import re
from typing import Any, Protocol

from loguru import logger


class ContextExtractor(Protocol):
    """Protocol for game-agnostic context extraction."""

    def extract(self, observation: str) -> str | dict: ...
    def extract_preconditions(self, context_key: str, observation: str) -> list[str]: ...


class RegexSpatialExtractor:
    """
    Extracts entity positions relative to a player position.
    Used for spatial games like Super Mario.

    Config shape:
        player_position_pattern: str  # regex with (x, y) groups
        entities: list[dict]          # [{keywords: [...], label: str}]
        distance_bins: dict           # {near: 60, mid: 140, far: 180}
        filter_behind: int            # ignore entities this far behind (-20)
        filter_ahead: int             # ignore entities this far ahead (180)
    """

    def __init__(self, config: dict[str, Any]):
        self.player_pattern = config["player_position_pattern"]
        self.entities = config["entities"]
        self.bins = config.get("distance_bins", {"near": 60, "mid": 140, "far": 180})
        self.filter_behind = config.get("filter_behind", -20)
        self.filter_ahead = config.get("filter_ahead", 180)

    def extract(self, observation: str | list) -> str:
        if isinstance(observation, list):
            observation = "\n".join(str(item) for item in observation)
        # Get player position
        player_x, player_y = 0, 0
        m = re.search(self.player_pattern, observation)
        if m:
            player_x, player_y = int(m.group(1)), int(m.group(2))

        context_parts = []
        for entity_def in self.entities:
            tokens = self._extract_entity(observation, entity_def, player_x)
            context_parts.extend(tokens)

        if not context_parts:
            return "clear_run"
        return "_".join(sorted(set(context_parts)))

    def _extract_entity(self, observation: str, entity_def: dict, player_x: int) -> list[str]:
        keywords = entity_def["keywords"]
        label = entity_def["label"]
        tokens = []

        for kw in keywords:
            line_pattern = rf"-\s*{kw}[^\n]*?:\s*(.+?)(?:\n|$)"
            line_match = re.search(line_pattern, observation, re.IGNORECASE)
            if not line_match:
                continue
            positions_str = line_match.group(1)
            if "none" in positions_str.lower():
                continue

            for x_str, y_str in re.findall(r'\((\d+),\s*(\d+)(?:,\s*\d+)?\)', positions_str):
                dx = int(x_str) - player_x
                if dx < self.filter_behind or dx > self.filter_ahead:
                    continue
                direction = "ahead" if dx >= 0 else "behind"
                abs_dx = abs(dx)
                if abs_dx <= self.bins["near"]:
                    dist_label = "near"
                elif abs_dx <= self.bins["mid"]:
                    dist_label = "mid"
                else:
                    dist_label = "far"
                tokens.append(f"{label}_{direction}_{dist_label}")

        return tokens

    def extract_preconditions(self, context_key: str, observation: str) -> list[str]:
        preconditions = []
        if not context_key or context_key == "clear_run":
            return preconditions

        parts = context_key.split("_")
        i = 0
        while i < len(parts):
            entity_labels = [e["label"] for e in self.entities]
            if parts[i] in entity_labels:
                entity = parts[i]
                i += 1
                direction, distance = None, None
                if i < len(parts) and parts[i] in ("ahead", "behind"):
                    direction = parts[i]
                    i += 1
                if i < len(parts) and parts[i] in ("near", "mid", "far"):
                    distance = parts[i]
                    i += 1
                if direction and distance:
                    preconditions.append(f"{entity}_{direction}_{distance}")
                elif direction:
                    preconditions.append(f"{entity}_{direction}")
                else:
                    preconditions.append(entity)
            else:
                i += 1
        return preconditions


class DictFieldExtractor:
    """
    Extracts named fields via regex patterns into a dict context.
    Used for narrative games like Pokemon Red.

    Config shape:
        fields: list[dict]  # [{name: str, pattern: str, type: "str"|"int"|"tuple"}]
    """

    def __init__(self, config: dict[str, Any]):
        self.fields = config["fields"]

    def extract(self, observation: str) -> dict[str, Any]:
        if isinstance(observation, list):
            observation = "\n".join(str(item) for item in observation)
        context = {}
        for field_def in self.fields:
            name = field_def["name"]
            pattern = field_def["pattern"]
            field_type = field_def.get("type", "str")
            try:
                m = re.search(pattern, observation, re.IGNORECASE)
                if m:
                    if field_type == "int":
                        context[name] = int(m.group(1))
                    elif field_type == "tuple":
                        context[name] = (int(m.group(1)), int(m.group(2)))
                    else:
                        context[name] = m.group(1)
            except Exception:
                pass
        return context

    def extract_preconditions(self, context_key: str, observation: str) -> list[str]:
        context = self.extract(observation) if isinstance(observation, str) else {}
        preconditions = []
        for k, v in context.items():
            preconditions.append(f"{k}={v}")
        return preconditions


class GeometricExtractor:
    """
    Extracts grid-based geometric features.
    Used for tile games like 2048.

    Config shape:
        score_pattern: str
        grid_pattern: str | None   # regex to extract grid text
        tile_keywords: list[str]   # tile values to look for
    """

    def __init__(self, config: dict[str, Any]):
        self.score_pattern = config.get("score_pattern", r"Score:\s*(\d+)")
        self.grid_pattern = config.get("grid_pattern")

    def extract(self, observation: str) -> dict[str, Any]:
        if isinstance(observation, list):
            observation = "\n".join(str(item) for item in observation)
        context = {}
        try:
            score_match = re.search(self.score_pattern, observation)
            if score_match:
                context["score"] = int(score_match.group(1))
            if self.grid_pattern:
                grid_match = re.search(self.grid_pattern, observation)
                if grid_match:
                    context["grid"] = grid_match.group(1).strip()
        except Exception as e:
            logger.warning(f"GeometricExtractor failed: {e}")
        return context

    def extract_preconditions(self, context_key: str, observation: str) -> list[str]:
        if context_key:
            return [f"board_state={context_key}"]
        return []


def build_context_extractor(mode: str, config: dict[str, Any]) -> ContextExtractor:
    """Factory: build the right extractor from game config."""
    if mode == "regex_spatial":
        return RegexSpatialExtractor(config)
    elif mode == "dict_fields":
        return DictFieldExtractor(config)
    elif mode == "geometric":
        return GeometricExtractor(config)
    else:
        raise ValueError(f"Unknown context extraction mode: {mode}")
