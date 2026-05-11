#!/usr/bin/env bash
# Stage B for 2048 — vmem OFF, planner ON. Fills the missing slot in
# the pr31_2048_rerun chain (which previously did only A/C/D), so the
# cross-game scoreboard can show the planner-only contribution on 2048
# the same way it does on pokemon.
#
# Launch as detached daemon (PPID=1):
#   setsid nohup ./experiments/pr31_2048_rerun/run_stage_b.sh \
#     </dev/null >>logs/pr31_2048_rerun_stage_b_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/twenty_fourty_eight/agent/gemma_26b.yaml"
ENV_CFG="configs/twenty_fourty_eight/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/twenty_fourty_eight"

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

# Swap in AWQ to reuse the running vLLM, set Stage B switches.
sed -i 's|^model: .*|model: "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"|' "$AGENT_CFG"
sed -i 's|^use_vector_memory: .*|use_vector_memory: false|' "$AGENT_CFG"
sed -i 's|^use_subtask_planning: .*|use_subtask_planning: true|' "$AGENT_CFG"
sed -i 's|^max_steps: .*|max_steps: 300|' "$ENV_CFG"

stage="stage_b_2048"
desc="Stage B 2048 26B-AWQ: vmem OFF, planner ON — isolate planner contribution (post-#46 units)"
run_id="pr31_rerun_2048_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
game_logs="$GAME_DATA_DIR/game_logs/twenty_fourty_eight/$run_id"
started=$(date +%s)

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE: $stage"
echo "  vmem=false  planner=true  max_steps=300"
echo "  run_id=$run_id"
echo "================================================================"

if ! uv run python run.py \
    -c gemma_26b \
    --local --games twenty_fourty_eight \
    --run-id "$run_id" \
    -d "$desc"; then
    echo "[FAIL] $stage run.py exited non-zero"
    exit 1
fi

actual_dir="$GAME_DATA_DIR/twenty_fourty_eight/$run_id"
if [ -d "$actual_dir" ]; then
    ln -sfn "$actual_dir" "$game_logs"
else
    echo "[WARN] $stage actual run dir missing: $actual_dir"
fi

elapsed=$(( ( $(date +%s) - started ) / 60 ))
echo "[$stage] runtime=${elapsed}min"

if ! uv run python experiments/pr31_2048_rerun/append.py \
    --variant "$stage" \
    --description "$desc" \
    --game-logs "$game_logs" \
    --runtime-min "$elapsed"; then
    echo "[FAIL] $stage append.py exited non-zero"
    exit 1
fi

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE B 2048 DONE"
echo "================================================================"
