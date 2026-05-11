#!/usr/bin/env bash
# Mario + 2048 self-reflection retest with AWQ-model guard.
#
# Original cross-game chain run (16:35Z–17:21Z) produced NO valid mario/2048
# data — between pokemon finishing and mario starting, the working tree was
# switched and the mario/2048 configs reverted to `unsloth/gemma-4-26b-a4b-it`
# (non-AWQ). vLLM only serves AWQ → every LLM call 404'd → fallback action
# on every step. self-reflection never fired (0/300 prompts had critique).
#
# This script:
# - Asserts each game's agent config has the AWQ model before launching.
# - Skips pokemon (the 16:35Z run was valid and tied Stage D 57.14% with
#   meaningful trajectory deepening — see PR #64 body).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"

run_game() {
    local game="$1" env_cfg_path="$2"
    local agent_cfg_path="configs/$game/agent/gemma_26b.yaml"

    # Config-drift guard: the agent config MUST point at the AWQ model
    # served by the running vLLM. Without this guard, an accidental branch
    # switch silently corrupts the run (see PR #64 retro).
    if ! grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$agent_cfg_path"; then
        echo "FATAL: $agent_cfg_path is NOT pointing at the AWQ model."
        echo "  Current: $(grep '^model:' $agent_cfg_path)"
        echo "  vLLM serves: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
        echo "  Aborting before silently producing fallback-action garbage."
        return 1
    fi
    grep -q "use_self_reflection: true" "$agent_cfg_path" || {
        echo "FATAL: $agent_cfg_path does not have self-reflection enabled"
        return 1
    }

    local run_id="cross_game_self_reflect_${game}_$(date -u +%Y%m%dT%H%M%SZ)_v2"
    local game_logs="$GAME_DATA_DIR/game_logs/$game/$run_id"
    local started; started=$(date +%s)

    sed -i 's/^max_steps: .*/max_steps: 300/' "$env_cfg_path"
    mkdir -p "$GAME_DATA_DIR/game_logs/$game"

    echo
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] GAME: $game (v2 with AWQ guard)"
    echo "  run_id=$run_id"
    echo "================================================================"

    if ! uv run python run.py \
        -c gemma_26b \
        --local --games "$game" \
        --run-id "$run_id" \
        -d "Stage D + self-reflection cross-game test on $game v2 (AWQ guard)"; then
        echo "[FAIL] $game run.py exited non-zero — continuing chain"
        return 1
    fi

    local actual_dir="$GAME_DATA_DIR/$game/$run_id"
    [ -d "$actual_dir" ] && ln -sfn "$actual_dir" "$game_logs"

    local elapsed=$(( ( $(date +%s) - started ) / 60 ))
    echo "[$game] runtime=${elapsed}min"

    uv run python experiments/cross_game_self_reflect/append.py \
        --game "$game" --game-logs "$game_logs" --runtime-min "$elapsed" \
        || echo "[WARN] $game append.py failed"
}

echo "[$(date -u +%H:%M:%SZ)] === mario + 2048 v2 chain start ==="
run_game super_mario configs/super_mario/env/default.yaml
run_game twenty_fourty_eight configs/twenty_fourty_eight/env/default.yaml
echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] MARIO + 2048 v2 CHAIN COMPLETE"
echo "================================================================"
