#!/usr/bin/env bash
# Cross-game retest using per-game adapter recommendations (commit 669b199).
#
# Adapter defaults (from cross-game retro):
#   pokemon_red:         use_self_reflection=True,  reflect_every=10
#   super_mario:         use_self_reflection=True,  reflect_every=30
#   twenty_fourty_eight: use_self_reflection=False  (-9pp regression at every-10)
#
# YAMLs no longer override → adapter decides. Verifies the new
# Optional[bool/int] LocalConfig path works end-to-end + measures whether
# the per-game tuning lifts mario / restores 2048 to its Stage D baseline.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"

run_game() {
    local game="$1" env_cfg_path="$2"
    local agent_cfg_path="configs/$game/agent/gemma_26b.yaml"

    grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$agent_cfg_path" || {
        echo "FATAL: $agent_cfg_path not pointing at AWQ. Current: $(grep '^model:' $agent_cfg_path)"; return 1; }
    # YAML must NOT explicitly set use_self_reflection — adapter should win.
    if grep -q "^use_self_reflection:" "$agent_cfg_path"; then
        echo "WARN: $agent_cfg_path has explicit use_self_reflection — adapter recommendation is bypassed"
    fi

    local run_id="cross_game_self_reflect_${game}_$(date -u +%Y%m%dT%H%M%SZ)_v3"
    local game_logs="$GAME_DATA_DIR/game_logs/$game/$run_id"
    local started; started=$(date +%s)

    sed -i 's/^max_steps: .*/max_steps: 300/' "$env_cfg_path"
    mkdir -p "$GAME_DATA_DIR/game_logs/$game"

    echo
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] GAME: $game (v3 — adapter recommendations)"
    echo "  run_id=$run_id"
    echo "================================================================"

    if ! uv run python run.py \
        -c gemma_26b \
        --local --games "$game" \
        --run-id "$run_id" \
        -d "Stage D + per-game adapter self-reflection defaults ($game) v3"; then
        echo "[FAIL] $game run.py exited non-zero"; return 1
    fi

    local actual_dir="$GAME_DATA_DIR/$game/$run_id"
    [ -d "$actual_dir" ] && ln -sfn "$actual_dir" "$game_logs"

    local elapsed=$(( ( $(date +%s) - started ) / 60 ))
    echo "[$game] runtime=${elapsed}min"

    uv run python experiments/cross_game_self_reflect/append.py \
        --game "$game" --game-logs "$game_logs" --runtime-min "$elapsed" \
        || echo "[WARN] $game append.py failed"
}

echo "[$(date -u +%H:%M:%SZ)] === cross-game v3 chain start (adapter recommendations) ==="
run_game pokemon_red         configs/pokemon_red/env/default.yaml
run_game super_mario         configs/super_mario/env/default.yaml
run_game twenty_fourty_eight configs/twenty_fourty_eight/env/default.yaml
echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] CROSS-GAME v3 CHAIN COMPLETE"
echo "================================================================"
