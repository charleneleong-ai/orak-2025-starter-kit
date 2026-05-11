#!/usr/bin/env bash
# Mario rerun chain on AWQ-26B — mirrors the 2048 rerun for cross-game
# scoreboard symmetry. Runs Stage A → Stage B → Stage C → Stage D
# sequentially against the same running vLLM that pokemon/2048 use.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/super_mario/agent/gemma_26b.yaml"
ENV_CFG="configs/super_mario/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/super_mario"

ORIG_AGENT_MODEL=$(grep -E "^model:" "$AGENT_CFG" | head -1)
ORIG_AGENT_VMEM=$(grep -E "^use_vector_memory:" "$AGENT_CFG" | head -1)
ORIG_AGENT_PLANNER=$(grep -E "^use_subtask_planning:" "$AGENT_CFG" | head -1)
ORIG_ENV_STEPS=$(grep -E "^max_steps:" "$ENV_CFG" | head -1)

restore() {
    echo "[restore] resetting mario agent + env config to original"
    if [ -n "$ORIG_AGENT_MODEL" ]; then
        sed -i "s|^model: .*|${ORIG_AGENT_MODEL}|" "$AGENT_CFG"
    fi
    sed -i "s|^use_vector_memory: .*|${ORIG_AGENT_VMEM:-use_vector_memory: true}|" "$AGENT_CFG"
    sed -i "s|^use_subtask_planning: .*|${ORIG_AGENT_PLANNER:-use_subtask_planning: true}|" "$AGENT_CFG"
    sed -i "s|^max_steps: .*|${ORIG_ENV_STEPS:-max_steps: 300}|" "$ENV_CFG"
}
trap restore EXIT

# Reuse the running vLLM (AWQ-26B).
sed -i 's|^model: .*|model: "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"|' "$AGENT_CFG"

run_stage() {
    local stage="$1" vmem="$2" planner="$3" max_steps="$4" desc="$5"
    local run_id="pr31_rerun_mario_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
    local game_logs="$GAME_DATA_DIR/game_logs/super_mario/$run_id"
    local started; started=$(date +%s)

    echo
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] STAGE: $stage"
    echo "  vmem=$vmem  planner=$planner  max_steps=$max_steps"
    echo "  run_id=$run_id"
    echo "================================================================"

    sed -i "s|^use_vector_memory: .*|use_vector_memory: ${vmem}|" "$AGENT_CFG"
    sed -i "s|^use_subtask_planning: .*|use_subtask_planning: ${planner}|" "$AGENT_CFG"
    sed -i "s|^max_steps: .*|max_steps: ${max_steps}|" "$ENV_CFG"

    if ! uv run python run.py \
        -c gemma_26b \
        --local --games super_mario \
        --run-id "$run_id" \
        -d "$desc"; then
        echo "[FAIL] $stage run.py exited non-zero — skipping append"
        return 1
    fi

    local actual_dir="$GAME_DATA_DIR/super_mario/$run_id"
    if [ -d "$actual_dir" ]; then
        ln -sfn "$actual_dir" "$game_logs"
    else
        echo "[WARN] $stage actual run dir missing: $actual_dir"
    fi

    local elapsed=$(( ( $(date +%s) - started ) / 60 ))
    echo "[$stage] runtime=${elapsed}min"

    if ! uv run python experiments/pr31_mario_rerun/append.py \
        --variant "$stage" \
        --description "$desc" \
        --game-logs "$game_logs" \
        --runtime-min "$elapsed"; then
        echo "[FAIL] $stage append.py exited non-zero"
        return 1
    fi

    echo "[$stage] DONE"
}

run_stage stage_a_mario false false 300 \
    "Stage A mario 26B-AWQ: vmem OFF, planner OFF — model-only baseline"
run_stage stage_b_mario false true 300 \
    "Stage B mario 26B-AWQ: vmem OFF, planner ON — isolate planner contribution"
run_stage stage_c_mario true false 300 \
    "Stage C mario 26B-AWQ: vmem ON, planner OFF — isolate vmem contribution"
run_stage stage_d_mario true true 300 \
    "Stage D mario 26B-AWQ: vmem ON, planner ON — full stack"

echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] MARIO CHAIN COMPLETE"
echo "================================================================"
