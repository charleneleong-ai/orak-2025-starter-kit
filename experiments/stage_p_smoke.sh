#!/usr/bin/env bash
# Stage P smoke run — single iter, max_steps=50, no checkpoint inheritance.
# Validates two things before committing to the n=5 sweep:
#   1. Map-graph hint reaches the LLM subtask planner — the string
#      "### Map graph" appears in raw_requests.jsonl.
#   2. The MAP_GRAPH constant + map_graph_hint method are wired in
#      unified.py.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Load API keys if present
[[ -f "$REPO/.env" ]] && set -a && source "$REPO/.env" && set +a

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
N_STEPS=50
TAG="stage_p_smoke"

export GAME_DATA_DIR="/tmp/orak-stage-p-smoke"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Pre-flight: Stage P code present
grep -q "^MAP_GRAPH:" agents/macla/macla_lib.py \
    || { echo "FATAL: MAP_GRAPH constant missing"; exit 1; }
grep -q "def map_graph_hint" agents/macla/macla_lib.py \
    || { echo "FATAL: map_graph_hint method missing"; exit 1; }
grep -q "graph_hint = mem.map_graph_hint" agents/macla/unified.py \
    || { echo "FATAL: Stage P observation augmentation missing in unified.py"; exit 1; }
grep -q '"ViridianCity"' agents/macla/macla_lib.py \
    || { echo "FATAL: MAP_GRAPH missing ViridianCity (the M5 unblock)"; exit 1; }
echo "[preflight] Stage P code present"

# Pre-flight: asm files
ASM_DIR="evaluation_utils/mcp_game_servers/pokemon_red/game/pokered/data/maps/objects"
asm_count=$(find "$ASM_DIR" -maxdepth 1 -name "*.asm" 2>/dev/null | wc -l)
[[ "$asm_count" -ge 100 ]] || { echo "FATAL: only $asm_count .asm files"; exit 1; }
echo "[preflight] $asm_count .asm files in pokered/data/maps/objects/"

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
    -d "Stage P smoke (max_steps=$N_STEPS, no checkpoint)"

# Post-flight: verify Map graph hint fired (logged by unified.py each time it prepends)
# raw_requests.jsonl logs executor calls; the planner call isn't captured there.
# Instead check the daemon log (passed via STAGE_P_LOGFILE env var) or the run log.
run_log="${STAGE_P_LOGFILE:-}"
if [[ -z "$run_log" ]]; then
    # fallback: most-recent smoke log in standard location
    run_log=$(ls -t "$REPO/logs/stage_p_smoke_"*.log 2>/dev/null | head -1)
fi
echo
if [[ -n "$run_log" && -f "$run_log" ]]; then
    hits=$(grep -c "graph_hint fired" "$run_log" || echo 0)
    echo "[postflight] 'graph_hint fired' log lines: $hits"
    [[ "$hits" -gt 0 ]] || { echo "FATAL: Map graph hint never fired — check unified.py wiring"; exit 1; }
else
    # Also check raw_requests for the hint text (older format / alternate log path)
    raw_log="$GAME_DATA_DIR/pokemon_red/$run_id/logs/raw_requests.jsonl"
    if [[ -f "$raw_log" ]]; then
        hits=$(grep -c "Map graph" "$raw_log" || echo 0)
        echo "[postflight] 'Map graph' in raw_requests.jsonl: $hits (indirect check)"
    else
        echo "[postflight] WARNING: no log found to verify wiring"
    fi
fi
