#!/usr/bin/env bash
# Stage N smoke run — single iter, max_steps=50, no checkpoint inheritance.
# Validates two things before committing to the n=5 sweep:
#   1. Novelty hint reaches the LLM subtask planner (string "### Novelty"
#      appears in raw_requests.jsonl).
#   2. Selector log line no longer carries `(new_map=...)` — confirms the
#      novelty-bump cleanup landed in BayesianProcedureSelector.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
N_STEPS=50
TAG="stage_n_smoke"

export GAME_DATA_DIR="/tmp/orak-stage-n-smoke"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Pre-flight: code present
grep -q "_SDC_BOOTSTRAP_N = 3" agents/macla/macla_lib.py \
    || { echo "FATAL: _SDC_BOOTSTRAP_N missing"; exit 1; }
grep -q "def map_visit_status" agents/macla/macla_lib.py \
    || { echo "FATAL: map_visit_status missing"; exit 1; }
grep -q "### Novelty" agents/macla/unified.py \
    || { echo "FATAL: novelty hint wiring missing in unified.py"; exit 1; }
grep -q "_NEW_MAP_THETA" agents/macla/macla_lib.py \
    && { echo "FATAL: _NEW_MAP_THETA should be removed"; exit 1; } || true
echo "[preflight] Stage N code present"

# Pre-flight: vLLM
served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
    | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
declared=$(grep '^model:' "configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml" \
    | head -1 | sed 's/model: *"//;s/" *$//')
[[ "$served" == "$declared" ]] \
    || { echo "FATAL: vLLM mismatch. declared=$declared served=$served"; exit 1; }
echo "[preflight] vLLM serving $served"

restore() { sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"; }
trap restore EXIT
sed -i "s/^max_steps: .*/max_steps: $N_STEPS/" "$ENV_CFG"

run_id="${TAG}_$(date -u +%Y%m%dT%H%M%SZ)"
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] $TAG  max_steps=$N_STEPS  run_id=$run_id"
echo "================================================================"

uv run python run.py \
    -c "$AGENT_CFG_NAME" \
    --local --games pokemon_red \
    --run-id "$run_id" \
    -d "Stage N smoke (max_steps=$N_STEPS, no checkpoint)"
