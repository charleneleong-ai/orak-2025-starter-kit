#!/usr/bin/env bash
# Stage G — procedure-layer escape live validation on pokemon Stage D.
#
# Post-Stage-E + 600-step + (queued) Stage F diagnosis: all three action-layer
# interventions plateau at score=4 (57.14%) for pokemon. Hypothesis: the
# bottleneck is the Bayesian procedure selector locking onto a failing
# procedure at the milestone-4 plateau. Two surgical fixes:
#
#   1. failure-streak retirement (K=5): retire procedures with 5 consecutive
#      failures; selector falls through to LLM fallback
#   2. stuck-state forced LLM (N=50): when 50 steps pass without ANY
#      step-success, _compute_adaptive_theta returns 1.01 → all candidates
#      rejected → LLM fallback for at least one step
#
# Comparison baselines:
#   Stage D pure (PR #31):                57.14% (300st, n=1)
#   Stage D + reflect v3 (PR #64):        57.14% (300st)
#   Stage E LangGraph + verify_action:    57.14% (300st)
#   Stage D + reflect (600 steps):        57.14% (600st)
#   This Stage G (procedure-escape K=5/N=50, 300st): target ≥ 71.43%
#
# Both knobs are configurable via YAML (LocalConfig.procedure_failure_streak_max,
# .force_llm_after_stuck_steps). Setting either to 0 disables that fix.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$AGENT_CFG" || {
    echo "FATAL: $AGENT_CFG not AWQ"; exit 1; }

restore() {
    echo "[restore] max_steps → 300"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

stage="stage_g_procedure_escape"
run_id="pr_procesc_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
game_logs="$GAME_DATA_DIR/game_logs/pokemon_red/$run_id"
started=$(date +%s)

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE G: $stage"
echo "  config: gemma_26b + LocalConfig defaults (K=5, N=50)"
echo "  baseline: Stage D = 57.14% (4/7), target ≥ 71.43% (5/7)"
echo "  run_id=$run_id"
echo "================================================================"

if ! uv run python run.py \
    -c gemma_26b \
    --local --games pokemon_red \
    --run-id "$run_id" \
    -d "Stage G: procedure-layer escape (failure-streak retire K=5 + stuck-state force-LLM N=50) on pokemon Stage D — does escaping the procedure loop bank milestone 5?"; then
    echo "[FAIL] $stage run.py exited non-zero"
    exit 1
fi

actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
[ -d "$actual_dir" ] && ln -sfn "$actual_dir" "$game_logs"

elapsed=$(( ( $(date +%s) - started ) / 60 ))
echo "[$stage] runtime=${elapsed}min"
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE G DONE"
echo "================================================================"
