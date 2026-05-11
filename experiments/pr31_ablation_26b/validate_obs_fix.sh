#!/usr/bin/env bash
# Live validation for PR #61 (pokemon obs preprocessor).
#
# Compares against the Stage A baseline (vmem OFF, planner OFF):
#   stage_a_26b           = 2/7  (28.57%)  300 steps   [truncated obs — "lucky" escape per Stage C++ retrospective]
#
# Expected with the fix: agent's obs now contains the FULL explored map
# of RedsHouse1f (via accumulated map_memory + replace_map_on_screen_with_full_map),
# so the exit warp at (3,7)/(2,7) is visible from any 1F position — should
# reach Oak's Lab consistently rather than relying on n=1 luck.
#
# Same Stage A config (no vmem, no planner) so any score lift is attributable
# to the obs fix, not the MMR/decay or planner wiring.
#
# Launch as detached daemon (PPID=1):
#   setsid nohup ./experiments/pr31_ablation_26b/validate_obs_fix.sh \
#     </dev/null >>logs/validate_obs_fix_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

grep -q "AWQ" "$AGENT_CFG" || { echo "FATAL: $AGENT_CFG is not the AWQ variant"; exit 1; }

restore() {
    echo "[restore] resetting agent + env config to Stage D defaults"
    sed -i 's/^use_vector_memory: .*/use_vector_memory: true/' "$AGENT_CFG"
    sed -i 's/^use_subtask_planning: .*/use_subtask_planning: true/' "$AGENT_CFG"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT

stage="stage_a_obs_fix"
desc="Stage A 26B + obs preprocessor (PR #61): vmem OFF, planner OFF, 300 steps — validate that expanded obs lets the agent reliably escape RedsHouse"
run_id="pr31_rerun_pokemon_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
game_logs="$GAME_DATA_DIR/game_logs/pokemon_red/$run_id"
started=$(date +%s)

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE: $stage"
echo "  vmem=false  planner=false  max_steps=300"
echo "  obs preprocessor (PR #61): enabled via game_adapter"
echo "  run_id=$run_id"
echo "================================================================"

sed -i 's/^use_vector_memory: .*/use_vector_memory: false/' "$AGENT_CFG"
sed -i 's/^use_subtask_planning: .*/use_subtask_planning: false/' "$AGENT_CFG"
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

if ! uv run python run.py \
    -c gemma_26b \
    --local --games pokemon_red \
    --run-id "$run_id" \
    -d "$desc"; then
    echo "[FAIL] $stage run.py exited non-zero"
    exit 1
fi

actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
if [ -d "$actual_dir" ]; then
    ln -sfn "$actual_dir" "$game_logs"
else
    echo "[WARN] $stage actual run dir missing: $actual_dir"
fi

elapsed=$(( ( $(date +%s) - started ) / 60 ))
echo "[$stage] runtime=${elapsed}min"

if ! uv run python experiments/pr31_ablation_26b/append.py \
    --variant "$stage" \
    --description "$desc" \
    --game-logs "$game_logs" \
    --runtime-min "$elapsed"; then
    echo "[FAIL] $stage append.py exited non-zero"
    exit 1
fi

if ! uv run python experiments/pr31_ablation_26b/post_retro.py; then
    echo "[WARN] $stage post_retro.py failed — continuing"
fi

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] OBS-FIX VALIDATION DONE"
echo "================================================================"
