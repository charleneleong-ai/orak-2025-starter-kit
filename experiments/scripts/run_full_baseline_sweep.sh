#!/usr/bin/env bash
# Cross-game baseline sweep: fills in the Stage A mario + pokemon gaps and
# retries the failed Stage D pokemon sweep, then regenerates the scoreboard
# plot. Produces data only — does NOT git push or post PR comments. Review
# the results, then commit / update the PR by hand.
#
# Manual invocation:
#   ./experiments/scripts/run_full_baseline_sweep.sh
#
# vLLM Gemma server must be up first (./serving/gemma_serve.sh).

set -euo pipefail

REPO=/workspace/orak-2025-starter-kit
PYTHON=$REPO/.venv/bin/python
LOG_DIR=$REPO/logs
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$LOG_DIR/full_baseline_sweep_${TS}.log

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "=== full_baseline_sweep starting at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
cd "$REPO"

# 1. vLLM health check
if ! curl -sf -m 5 http://localhost:8000/v1/models > /dev/null; then
    echo "ERROR: vLLM at :8000 not responding. Start it via ./serving/gemma_serve.sh and re-run."
    exit 1
fi
echo "vLLM healthy."

# 2. Sweep config — tag, config name, games, brief description
declare -a SWEEPS=(
    "harness_check_mario|gemma_stage_a|super_mario|Stage A mario baseline (PR #28 gap-fill)"
    "harness_check_pokemon|gemma_stage_a|pokemon_red|Stage A pokemon baseline (PR #28 gap-fill)"
    "pokemon_check_v4|gemma|pokemon_red|Stage D pokemon retry (v3 sweep crashed)"
)

for line in "${SWEEPS[@]}"; do
    IFS='|' read -r tag cfg games desc <<< "$line"
    echo
    echo "--- Sweep: $tag (config=$cfg, games=$games) ---"
    echo "    $desc"
    "$PYTHON" "$REPO/experiments/autoresearch.py" run \
        --config "$cfg" \
        --tag "$tag" \
        --games "$games" \
        --max-iterations 2 \
        --note "$desc" \
        --config-name gemma
    echo "--- Sweep $tag complete at $(date -u +%H:%MZ) ---"
done

# 3. Regenerate scoreboard plot — all 3 games × 3 stages
PLOT_OUT=$REPO/docs/experiments/gemma/plots/stage_d_cross_game.png
PYTHONPATH=$REPO "$PYTHON" "$REPO/experiments/plot_comparisons.py" scoreboard \
    --game twenty_fourty_eight \
        --tag harness_check          --label "Stage A" \
        --tag cognitive_check_v2     --label "Stage C (vmem)" \
        --tag stage_d_ablation_2048  --label "Stage D (vmem+planner)" \
        --sep 3 \
    --game super_mario \
        --tag harness_check_mario          --label "Stage A" \
        --tag mario_check                  --label "Stage C (vmem)" \
        --tag stage_d_ablation_mario_v3    --label "Stage D (vmem+planner)" \
        --sep 3 \
    --game pokemon_red \
        --tag harness_check_pokemon  --label "Stage A" \
        --tag pokemon_check          --label "Stage C (vmem)" \
        --tag pokemon_check_v4       --label "Stage D (vmem+planner)" \
        --sep 3 \
    --config-name gemma \
    --title "Stage D cross-game ablation — full A/C/D triple" \
    --out "$PLOT_OUT"

echo
echo "Plot regenerated: $PLOT_OUT"
echo "=== full_baseline_sweep done at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo
echo "Next steps (manual): review experiments/{harness_check_mario,harness_check_pokemon,pokemon_check_v4}/gemma/results.jsonl,"
echo "then commit + push + update PR #28 by hand."
