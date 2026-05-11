#!/usr/bin/env bash
# Stage F: PR #67 plan-do-check live validation on pokemon.
#
# Tool gate + LLM plan check + bounded retry loop. Adapter recommends both
# ON by default. Same Stage D config (vmem + planner) — the only addition
# is the action-validator pipeline + retry-on-rejection.
#
# Comparison baselines:
#   Stage D (PR #31):                 57.14% (300st, no validator)
#   Stage D + reflect (PR #64):       57.14% (300st, periodic critique)
#   Stage E (PR #66):                 ???    (300st, LangGraph + Reflexion verify)
#   Stage F (this run, PR #67):       ???    (300st, tool gate + plan check + retry)
#   Stage D++ (PR #31):               71.43% (600st, no validator)
#
# The diagnosed failure mode (action LLM emits move_to(3,7) when the tile
# is a WarpPoint requiring warp_with_warp_point) — tool gate should catch
# the tile-type mismatch deterministically; plan check should catch the
# subgoal-divergence even when the rule doesn't fire.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Config-drift guards
grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$AGENT_CFG" || {
    echo "FATAL: $AGENT_CFG not AWQ. Current: $(grep '^model:' $AGENT_CFG)"; exit 1; }
# YAML should NOT explicitly set use_tool_gating / use_plan_check — adapter wins
if grep -qE "^(use_tool_gating|use_plan_check):" "$AGENT_CFG"; then
    echo "WARN: explicit validator key in $AGENT_CFG bypasses adapter recommendation"
fi
# Pokemon adapter recommends both validator legs ON
grep -q "RECOMMENDED_USE_TOOL_GATING = True" agents/pokemon_red/game_adapter.py || {
    echo "FATAL: pokemon adapter doesn't recommend tool gating ON"; exit 1; }
grep -q "RECOMMENDED_USE_PLAN_CHECK = True" agents/pokemon_red/game_adapter.py || {
    echo "FATAL: pokemon adapter doesn't recommend plan check ON"; exit 1; }

restore() {
    echo "[restore] max_steps → 300 (default)"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

stage="stage_f_plan_do_check"
run_id="pr67_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
game_logs="$GAME_DATA_DIR/game_logs/pokemon_red/$run_id"
started=$(date +%s)

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE F: $stage (PR #67 live validation)"
echo "  Stage D config + ToolGate + LLMPlanCheck + retry (max 2)"
echo "  baselines to beat: Stage D 57.14%, Stage D + reflect 57.14%"
echo "  run_id=$run_id"
echo "================================================================"

if ! uv run python run.py \
    -c gemma_26b \
    --local --games pokemon_red \
    --run-id "$run_id" \
    -d "Stage F (PR #67): plan-do-check on pokemon — tool gate + plan check + bounded retry"; then
    echo "[FAIL] $stage run.py exited non-zero"
    exit 1
fi

actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
[ -d "$actual_dir" ] && ln -sfn "$actual_dir" "$game_logs"

elapsed=$(( ( $(date +%s) - started ) / 60 ))
echo "[$stage] runtime=${elapsed}min"
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE F DONE"
echo "================================================================"
