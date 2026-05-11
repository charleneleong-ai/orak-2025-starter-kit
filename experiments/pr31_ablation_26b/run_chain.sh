#!/usr/bin/env bash
# Chain runner for the remaining PR #31 ablation stages on AWQ-26B.
#
# Runs Stage C -> Stage B -> Stage D++ sequentially, mutating
# configs/pokemon_red/agent/gemma_26b.yaml and configs/pokemon_red/env/default.yaml
# in-place between stages and restoring them on EXIT.
#
# Each stage:
#   1. Flips vmem/planner/max_steps via sed
#   2. Runs `python run.py -c gemma_26b --local --games pokemon_red --run-id pr31_rerun_pokemon_<stage>_<ts>`
#   3. On success: appends to results.jsonl, regenerates plot, PATCHes PR comment
#
# Launch as detached daemon (PPID=1):
#   setsid nohup ./experiments/pr31_ablation_26b/run_chain.sh \
#     </dev/null >>logs/pr31_chain_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
# evaluation_utils/runner.py writes per-run state to $GAME_DATA_DIR/<game>/<run_id>/
# but append.py + post_retro.py.find_run_dir look under $GAME_DATA_DIR/game_logs/<game>/.
# Symlink each run dir into the expected layout after run.py succeeds (see run_stage).
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Sanity: ensure agent config points at AWQ model
grep -q "AWQ" "$AGENT_CFG" || { echo "FATAL: $AGENT_CFG is not the AWQ variant"; exit 1; }

# Restore original Stage D defaults on any exit (success or failure)
restore() {
    echo "[restore] resetting agent + env config to Stage D defaults"
    sed -i 's/^use_vector_memory: .*/use_vector_memory: true/' "$AGENT_CFG"
    sed -i 's/^use_subtask_planning: .*/use_subtask_planning: true/' "$AGENT_CFG"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT

run_stage() {
    local stage="$1" vmem="$2" planner="$3" max_steps="$4" desc="$5"
    local run_id="pr31_rerun_pokemon_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
    local game_logs="$GAME_DATA_DIR/game_logs/pokemon_red/$run_id"
    local started; started=$(date +%s)

    echo
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] STAGE: $stage"
    echo "  vmem=$vmem  planner=$planner  max_steps=$max_steps"
    echo "  run_id=$run_id"
    echo "================================================================"

    sed -i "s/^use_vector_memory: .*/use_vector_memory: ${vmem}/" "$AGENT_CFG"
    sed -i "s/^use_subtask_planning: .*/use_subtask_planning: ${planner}/" "$AGENT_CFG"
    sed -i "s/^max_steps: .*/max_steps: ${max_steps}/" "$ENV_CFG"

    if ! uv run python run.py \
        -c gemma_26b \
        --local --games pokemon_red \
        --run-id "$run_id" \
        -d "$desc"; then
        echo "[FAIL] $stage run.py exited non-zero — skipping append/post"
        return 1
    fi

    # Symlink the actual run dir into the layout post_retro.py / append.py expect.
    local actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
    if [ -d "$actual_dir" ]; then
        ln -sfn "$actual_dir" "$game_logs"
    else
        echo "[WARN] $stage actual run dir missing: $actual_dir"
    fi

    local elapsed=$(( ( $(date +%s) - started ) / 60 ))
    echo "[$stage] runtime=${elapsed}min"

    if ! uv run python experiments/pr31_ablation_26b/append.py \
        --variant "$stage" \
        --description "$desc" \
        --game-logs "$game_logs" \
        --runtime-min "$elapsed"; then
        echo "[FAIL] $stage append.py exited non-zero"
        return 1
    fi

    if ! uv run python experiments/pr31_ablation_26b/post_retro.py; then
        echo "[WARN] $stage post_retro.py failed (likely gh auth) — continuing"
    fi

    echo "[$stage] DONE"
}

# ---- Stage C: vector memory ON, planner OFF ----
run_stage stage_c_26b true false 300 \
    "Stage C 26B: vmem ON, planner OFF — isolate vector memory contribution"

# ---- Stage B: vector memory OFF, planner ON ----
run_stage stage_b_26b false true 300 \
    "Stage B 26B: vmem OFF, planner ON — isolate subtask planner contribution"

# ---- Stage D++: both ON, 600 steps ----
run_stage stage_d_plus_26b true true 600 \
    "Stage D++ 26B: vmem ON, planner ON, 600 steps + grace=10 — extend Stage D's 57.14% with longer runtime"

echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] CHAIN COMPLETE"
echo "================================================================"
