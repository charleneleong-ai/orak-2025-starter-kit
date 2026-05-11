#!/usr/bin/env bash
# Stage C retest with PR #60's MMR rerank + repetition decay enabled.
#
# Compares against:
#   stage_c_26b   = 0.00% (0/7) over 300 steps    [vmem ON, planner OFF, vanilla retrieval]
#
# Hypothesis: MMR diversity + per-memory repetition decay break the
# retrieval reinforcement loop that pinned the agent in Red's house
# re-reading SIGN_REDSHOUSE1F_TV for 300 steps.
#
# Config differences vs vanilla Stage C:
#   vector_memory_use_mmr: true
#   vector_memory_mmr_lambda: 0.5
#   vector_memory_decay_alpha: 0.5
#   vector_memory_decay_window: 20
#
# Launch as detached daemon (PPID=1):
#   setsid nohup ./experiments/pr31_ablation_26b/run_stage_c_mmr_decay.sh \
#     </dev/null >>logs/stage_c_mmr_decay_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Sanity: agent config must point at AWQ + have new MMR/decay knobs
grep -q "AWQ" "$AGENT_CFG" || { echo "FATAL: $AGENT_CFG is not the AWQ variant"; exit 1; }
grep -q "vector_memory_use_mmr: true" "$AGENT_CFG" || { echo "FATAL: $AGENT_CFG does not have MMR enabled"; exit 1; }
grep -q "vector_memory_decay_alpha: 0.5" "$AGENT_CFG" || { echo "FATAL: $AGENT_CFG does not have decay enabled"; exit 1; }

# Restore Stage D defaults on any exit (mirrors run_chain.sh)
restore() {
    echo "[restore] resetting agent + env config to Stage D defaults"
    sed -i 's/^use_vector_memory: .*/use_vector_memory: true/' "$AGENT_CFG"
    sed -i 's/^use_subtask_planning: .*/use_subtask_planning: true/' "$AGENT_CFG"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT

stage="stage_c_26b_mmr_decay"
desc="Stage C retest 26B: vmem ON + MMR + decay (alpha=0.5, lambda=0.5), planner OFF — validate PR #60 breaks the retrieval reinforcement loop"
run_id="pr31_rerun_pokemon_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
game_logs="$GAME_DATA_DIR/game_logs/pokemon_red/$run_id"
started=$(date +%s)

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE: $stage"
echo "  vmem=true (MMR+decay)  planner=false  max_steps=300"
echo "  run_id=$run_id"
echo "================================================================"

sed -i 's/^use_vector_memory: .*/use_vector_memory: true/' "$AGENT_CFG"
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
    echo "[WARN] $stage post_retro.py failed (likely gh auth) — continuing"
fi

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE C RETEST DONE"
echo "================================================================"
