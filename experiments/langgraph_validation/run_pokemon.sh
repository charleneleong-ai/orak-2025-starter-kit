#!/usr/bin/env bash
# Live validation of PR #66 (feat/langgraph-react) on pokemon Stage D.
# Reflexion-style self-verification — a second LLM pass that re-reads obs +
# critique and verifies the proposed action before committing. Compared
# against PR #64's Stage D + self-reflection baseline at 57.14%.
#
# Same Stage D config (vmem ON + planner ON), but the agent is the
# LangGraph subclass with use_verify_action=true.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b_langgraph.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"
export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Guards: AWQ model + verify_action enabled (config-drift insurance per PR #64 retro)
grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$AGENT_CFG" || { echo "FATAL: $AGENT_CFG not AWQ"; exit 1; }
grep -q "use_verify_action: true" "$AGENT_CFG" || { echo "FATAL: verify_action not enabled"; exit 1; }
grep -q "LangGraphMaclaAgent" "$AGENT_CFG" || { echo "FATAL: not pointing at LangGraphMaclaAgent"; exit 1; }

restore() {
    echo "[restore] max_steps → 300"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

stage="stage_e_langgraph_verify"
run_id="pr66_${stage}_$(date -u +%Y%m%dT%H%M%SZ)"
game_logs="$GAME_DATA_DIR/game_logs/pokemon_red/$run_id"
started=$(date +%s)

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE E: $stage (PR #66 live validation)"
echo "  config: gemma_26b_langgraph (LangGraph + verify_action=true)"
echo "  baseline: Stage D + self-reflect (PR #64) = 57.14% (4/7)"
echo "  run_id=$run_id"
echo "================================================================"

if ! uv run python run.py \
    -c gemma_26b_langgraph \
    --local --games pokemon_red \
    --run-id "$run_id" \
    -d "Stage E (PR #66): LangGraph + verify_action on pokemon — does Reflexion-style self-verification lift Stage D's 57.14%?"; then
    echo "[FAIL] $stage run.py exited non-zero"
    exit 1
fi

actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
[ -d "$actual_dir" ] && ln -sfn "$actual_dir" "$game_logs"

elapsed=$(( ( $(date +%s) - started ) / 60 ))
echo "[$stage] runtime=${elapsed}min"
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE E DONE"
echo "================================================================"
