#!/usr/bin/env bash
# Stage C′++ pokemon retest with temperature=0.3 (down from 0.7).
#
# Hypothesis: at T=0.7 the LLM's RedsHouse exit-vs-staircase sampling is
# stochastic — Stage C′ escaped (lucky), Stage C′++ wedged (unlucky), same
# config. Lower temp should make the LLM more reliably act on the explicit
# pokemon_prompts.py hint that the exit is at (2,7)/(3,7) and the
# staircase at (7,1).
#
# Config: same as run_stage_c_plus_mmr_decay.sh (vmem ON + MMR + decay,
# planner OFF, 600 steps) + temperature swap. Obs preprocessor (PR #61)
# auto-engages via the merged adapter.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

grep -q "AWQ" "$AGENT_CFG" || { echo "FATAL: $AGENT_CFG is not the AWQ variant"; exit 1; }

ORIG_TEMP=$(grep -E "^temperature:" "$AGENT_CFG" | head -1)

restore() {
    echo "[restore] resetting agent + env config to Stage D defaults (T=0.7)"
    sed -i 's/^use_vector_memory: .*/use_vector_memory: true/' "$AGENT_CFG"
    sed -i 's/^use_subtask_planning: .*/use_subtask_planning: true/' "$AGENT_CFG"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
    sed -i "s|^temperature: .*|${ORIG_TEMP:-temperature: 0.7}|" "$AGENT_CFG"
}
trap restore EXIT

sed -i 's/^use_vector_memory: .*/use_vector_memory: true/' "$AGENT_CFG"
sed -i 's/^use_subtask_planning: .*/use_subtask_planning: false/' "$AGENT_CFG"
sed -i 's/^max_steps: .*/max_steps: 600/' "$ENV_CFG"
sed -i 's/^temperature: .*/temperature: 0.3/' "$AGENT_CFG"

stage="stage_c_plus_26b_mmr_decay_t03"
desc="Stage C′++ pokemon, vmem ON + MMR + decay, planner OFF, 600 steps, temperature=0.3 — test whether lower sampling temp lets the LLM act on the explicit exit-at-(2,7)/(3,7) prompt hint"
run_id="pr31_rerun_pokemon_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
game_logs="$GAME_DATA_DIR/game_logs/pokemon_red/$run_id"
started=$(date +%s)

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE: $stage"
echo "  vmem=true (MMR+decay)  planner=false  max_steps=600  temperature=0.3"
echo "  obs preprocessor (PR #61): auto-engaged via adapter"
echo "  run_id=$run_id"
echo "================================================================"

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

uv run python experiments/pr31_ablation_26b/post_retro.py || \
    echo "[WARN] $stage post_retro.py failed — continuing"

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE C′++ LOWTEMP DONE"
echo "================================================================"
