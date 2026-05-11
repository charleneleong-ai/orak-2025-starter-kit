#!/usr/bin/env bash
# Cross-game Stage D + self-reflection test.
#
# Runs the same Stage D config (vmem ON + planner ON + self-reflection ON,
# reflect_every=10) across pokemon_red → super_mario → twenty_fourty_eight
# sequentially. Each game's gemma_26b.yaml already has use_self_reflection=true
# and points at the AWQ-26B model that's served by the running vLLM.
#
# Compares against Stage D baselines (from PR #31's ablation):
#   pokemon_red:         57.14% (4/7)
#   super_mario:         35.21%
#   twenty_fourty_eight: 63.64% (max_tile=128)
#
# Launch as detached daemon (PPID=1):
#   setsid nohup ./experiments/cross_game_self_reflect/run_chain.sh \
#     </dev/null >>logs/cross_game_self_reflect_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"

run_game() {
    local game="$1" env_cfg_path="$2"
    local run_id="cross_game_self_reflect_${game}_$(date -u +%Y%m%dT%H%M%SZ)"
    local game_logs="$GAME_DATA_DIR/game_logs/$game/$run_id"
    local started; started=$(date +%s)

    # Force max_steps=300 across all games for a clean Stage D comparison
    sed -i 's/^max_steps: .*/max_steps: 300/' "$env_cfg_path"

    mkdir -p "$GAME_DATA_DIR/game_logs/$game"

    echo
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] GAME: $game"
    echo "  config: gemma_26b (vmem ON · planner ON · self_reflection every 10)"
    echo "  run_id=$run_id"
    echo "================================================================"

    if ! uv run python run.py \
        -c gemma_26b \
        --local --games "$game" \
        --run-id "$run_id" \
        -d "Stage D + self-reflection cross-game test on $game (vmem ON · planner ON · reflect_every=10)"; then
        echo "[FAIL] $game run.py exited non-zero — continuing chain"
        return 1
    fi

    local actual_dir="$GAME_DATA_DIR/$game/$run_id"
    if [ -d "$actual_dir" ]; then
        ln -sfn "$actual_dir" "$game_logs"
    fi

    local elapsed=$(( ( $(date +%s) - started ) / 60 ))
    echo "[$game] runtime=${elapsed}min"

    uv run python experiments/cross_game_self_reflect/append.py \
        --game "$game" --game-logs "$game_logs" --runtime-min "$elapsed" \
        || echo "[WARN] $game append.py failed"
}

echo "[$(date -u +%H:%M:%SZ)] === cross-game self-reflection chain start ==="
run_game pokemon_red configs/pokemon_red/env/default.yaml
run_game super_mario configs/super_mario/env/default.yaml
run_game twenty_fourty_eight configs/twenty_fourty_eight/env/default.yaml
echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] CROSS-GAME CHAIN COMPLETE"
echo "================================================================"
