#!/usr/bin/env bash
# 2048 rerun chain on current master + AWQ-26B (re-uses the running vLLM
# from the pokemon ablation chain). Reruns Stage A → Stage C → Stage D
# so the cross-game scoreboard has clean post-#46-units 2048 numbers.
#
# Mutates configs/twenty_fourty_eight/agent/gemma_26b.yaml to point at
# cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit (matches the running vLLM) and
# flips vmem/planner per stage; trap restores the original on exit.
#
# Launch as detached daemon (PPID=1):
#   setsid nohup ./experiments/pr31_2048_rerun/run_chain.sh \
#     </dev/null >>logs/pr31_2048_rerun_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/twenty_fourty_eight/agent/gemma_26b.yaml"
ENV_CFG="configs/twenty_fourty_eight/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/twenty_fourty_eight"

# Snapshot the original for trap restore.
ORIG_AGENT_MODEL=$(grep -E "^model:" "$AGENT_CFG" | head -1)
ORIG_AGENT_VMEM=$(grep -E "^use_vector_memory:" "$AGENT_CFG" | head -1)
ORIG_AGENT_PLANNER=$(grep -E "^use_subtask_planning:" "$AGENT_CFG" | head -1)
ORIG_ENV_STEPS=$(grep -E "^max_steps:" "$ENV_CFG" | head -1)

restore() {
    echo "[restore] resetting 2048 agent + env config to original"
    if [ -n "$ORIG_AGENT_MODEL" ]; then
        sed -i "s|^model: .*|${ORIG_AGENT_MODEL}|" "$AGENT_CFG"
    fi
    sed -i "s|^use_vector_memory: .*|${ORIG_AGENT_VMEM:-use_vector_memory: true}|" "$AGENT_CFG"
    sed -i "s|^use_subtask_planning: .*|${ORIG_AGENT_PLANNER:-use_subtask_planning: true}|" "$AGENT_CFG"
    sed -i "s|^max_steps: .*|${ORIG_ENV_STEPS:-max_steps: 300}|" "$ENV_CFG"
}
trap restore EXIT

# Always swap in the AWQ model so we reuse the running vLLM (no reload).
sed -i 's|^model: .*|model: "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"|' "$AGENT_CFG"

run_stage() {
    local stage="$1" vmem="$2" planner="$3" max_steps="$4" desc="$5"
    local run_id="pr31_rerun_2048_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
    local game_logs="$GAME_DATA_DIR/game_logs/twenty_fourty_eight/$run_id"
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
        --local --games twenty_fourty_eight \
        --run-id "$run_id" \
        -d "$desc"; then
        echo "[FAIL] $stage run.py exited non-zero — skipping append"
        return 1
    fi

    local actual_dir="$GAME_DATA_DIR/twenty_fourty_eight/$run_id"
    if [ -d "$actual_dir" ]; then
        ln -sfn "$actual_dir" "$game_logs"
    else
        echo "[WARN] $stage actual run dir missing: $actual_dir"
    fi

    local elapsed=$(( ( $(date +%s) - started ) / 60 ))
    echo "[$stage] runtime=${elapsed}min"

    if ! uv run python experiments/pr31_2048_rerun/append.py \
        --variant "$stage" \
        --description "$desc" \
        --game-logs "$game_logs" \
        --runtime-min "$elapsed"; then
        echo "[FAIL] $stage append.py exited non-zero"
        return 1
    fi

    echo "[$stage] DONE"
}

# ---- Stage A: model only (vmem OFF, planner OFF) ----
run_stage stage_a_2048 false false 300 \
    "Stage A 2048 26B-AWQ: vmem OFF, planner OFF — model-only baseline (post-#46 units)"

# ---- Stage C: vmem ON, planner OFF ----
run_stage stage_c_2048 true false 300 \
    "Stage C 2048 26B-AWQ: vmem ON, planner OFF — isolate vector memory contribution (post-#46 units)"

# ---- Stage D: vmem ON, planner ON ----
run_stage stage_d_2048 true true 300 \
    "Stage D 2048 26B-AWQ: vmem ON, planner ON — full stack (post-#46 units)"

echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] 2048 CHAIN COMPLETE"
echo "================================================================"
