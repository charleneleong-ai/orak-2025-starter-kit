"""Shared 0-100 progress-to-win metric for 2048.

Mirrors `evaluation_utils.mcp_game_servers.twenty_fourty_eight.game.twenty_fourty_eight_env.normalize_2048_score`.
Both layers must use the same formula so the chart's per-step score and
the agent-side wandb dashboards agree.
"""

import math

# 2048 = 2^11, so log2(max_tile)/11 maps progress onto 0-1.
WIN_TILE_LOG2 = 11.0


def normalize_2048_score(max_tile: int) -> float:
    if max_tile <= 1:
        return 0.0
    return min((math.log2(max_tile) / WIN_TILE_LOG2) * 100.0, 100.0)
