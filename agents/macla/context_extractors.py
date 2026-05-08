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
        # Get player position (only `player_x` is consumed downstream;
        # `player_y` is parsed but unused — kept as a structural mirror of
        # the regex's capture groups).
        player_x, _player_y = 0, 0
        m = re.search(self.player_pattern, observation)
        if m:
            player_x, _player_y = int(m.group(1)), int(m.group(2))

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

            for x_str, _y_str in re.findall(r"\((\d+),\s*(\d+)(?:,\s*\d+)?\)", positions_str):
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


class StrategicGridExtractor:
    """
    Extracts strategic-feature signatures from grid-based games (2048).

    The previous `GeometricExtractor` keyed procedures on the literal grid
    text, so every board produced a unique context key — procedures rarely
    fired on subsequent boards because the keys never matched. This extractor
    instead emits a strategic feature signature that captures the *invariants*
    that 2048 strategy depends on:

      max_tile_value  — quantised to log2 buckets (4, 16, 64, 256, ...)
      max_corner      — TL / TR / BL / BR / edge / center / none
      empty_bucket    — many (>=8) / mid (4-7) / few (<=3)
      chain_dir       — strongest descending-row direction (N/S/E/W/none)
      merge_count     — adjacent same-value pairs (0 / 1 / 2+)

    Millions of literal boards collapse into a few hundred strategic clusters.
    Procedures keyed on this signature fire across many boards with the same
    *shape*, accumulating the alpha/beta updates needed to actually be
    selected by the Bayesian selector.
    """

    def __init__(self, config: dict[str, Any]):
        self.score_pattern = config.get("score_pattern", r"Score:\s*(\d+)")
        self.grid_pattern = config.get(
            "grid_pattern",
            r"\[\[\s*\d+(?:\s*,\s*\d+){3}\s*\][\s,]*"
            r"\[\s*\d+(?:\s*,\s*\d+){3}\s*\][\s,]*"
            r"\[\s*\d+(?:\s*,\s*\d+){3}\s*\][\s,]*"
            r"\[\s*\d+(?:\s*,\s*\d+){3}\s*\]",
        )
        self.grid_size = config.get("grid_size", 4)

    def extract(self, observation: str) -> dict[str, Any]:
        if isinstance(observation, list):
            observation = "\n".join(str(item) for item in observation)

        context: dict[str, Any] = {}
        try:
            score_match = re.search(self.score_pattern, observation)
            if score_match:
                context["score"] = int(score_match.group(1))

            grid = self._parse_grid(observation)
            if grid is None:
                return context

            sig = self._strategic_signature(grid)
            # The Bayesian selector matches on `context_key` (a string). We
            # stringify the signature deterministically so identical shapes
            # produce identical keys.
            key = (
                f"max={sig['max_bucket']}|corner={sig['max_corner']}"
                f"|empty={sig['empty_bucket']}|chain={sig['chain_dir']}"
                f"|merges={sig['merge_bucket']}"
            )
            context["context_key"] = key
            context["signature"] = sig
        except Exception as e:
            logger.warning(f"StrategicGridExtractor failed: {e}")
        return context

    def extract_preconditions(self, context_key: str, observation: str) -> list[str]:
        # The signature key already encodes preconditions. Split it back so
        # individual features can be matched as discriminative tokens.
        if not context_key:
            return []
        return [tok for tok in context_key.split("|") if tok]

    def _parse_grid(self, observation: str) -> list[list[int]] | None:
        m = re.search(self.grid_pattern, observation)
        if not m:
            return None
        # Pull the 16 cell numbers in row-major order
        cells = [int(x) for x in re.findall(r"\d+", m.group(0))]
        if len(cells) != self.grid_size * self.grid_size:
            return None
        return [cells[i * self.grid_size : (i + 1) * self.grid_size] for i in range(self.grid_size)]

    def _strategic_signature(self, grid: list[list[int]]) -> dict[str, Any]:
        n = len(grid)
        max_val = max(max(row) for row in grid)

        # max_tile bucket — log2-style so 256 / 257 collapse to the same bucket
        if max_val <= 0:
            max_bucket = "0"
        else:
            # Find largest power of 2 <= max_val
            bucket_val = 1
            while bucket_val * 2 <= max_val:
                bucket_val *= 2
            max_bucket = str(bucket_val)

        # Locate max-tile position
        max_corner = "none"
        if max_val > 0:
            for r in range(n):
                for c in range(n):
                    if grid[r][c] == max_val:
                        if (r, c) == (0, 0):
                            max_corner = "TL"
                        elif (r, c) == (0, n - 1):
                            max_corner = "TR"
                        elif (r, c) == (n - 1, 0):
                            max_corner = "BL"
                        elif (r, c) == (n - 1, n - 1):
                            max_corner = "BR"
                        elif r in (0, n - 1) or c in (0, n - 1):
                            max_corner = "edge"
                        else:
                            max_corner = "center"
                        break
                if max_corner != "none":
                    break

        # Empty cell bucket
        empty = sum(1 for row in grid for v in row if v == 0)
        if empty >= 8:
            empty_bucket = "many"
        elif empty >= 4:
            empty_bucket = "mid"
        else:
            empty_bucket = "few"

        # Strongest descending-row direction (where the largest values cluster).
        # Compares row sums and col sums to find the dominant edge.
        row_sums = [sum(row) for row in grid]
        col_sums = [sum(grid[r][c] for r in range(n)) for c in range(n)]
        edges = {
            "N": row_sums[0],
            "S": row_sums[-1],
            "W": col_sums[0],
            "E": col_sums[-1],
        }
        chain_dir = max(edges, key=edges.get) if max(edges.values()) > 0 else "none"

        # Merge opportunities: adjacent cells with same value (and non-zero)
        merges = 0
        for r in range(n):
            for c in range(n):
                v = grid[r][c]
                if v == 0:
                    continue
                if c + 1 < n and grid[r][c + 1] == v:
                    merges += 1
                if r + 1 < n and grid[r + 1][c] == v:
                    merges += 1
        merge_bucket = "0" if merges == 0 else ("1" if merges == 1 else "2+")

        return {
            "max_value": max_val,
            "max_bucket": max_bucket,
            "max_corner": max_corner,
            "empty_count": empty,
            "empty_bucket": empty_bucket,
            "chain_dir": chain_dir,
            "merge_count": merges,
            "merge_bucket": merge_bucket,
        }


def build_context_extractor(mode: str, config: dict[str, Any]) -> ContextExtractor:
    """Factory: build the right extractor from game config."""
    if mode == "regex_spatial":
        return RegexSpatialExtractor(config)
    elif mode == "dict_fields":
        return DictFieldExtractor(config)
    elif mode == "geometric":
        return GeometricExtractor(config)
    elif mode == "strategic_grid":
        return StrategicGridExtractor(config)
    else:
        raise ValueError(f"Unknown context extraction mode: {mode}")
