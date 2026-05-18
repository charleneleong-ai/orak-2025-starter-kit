#!/usr/bin/env bash
# Stage R v2 spike test — single iter, reduced max_steps.
#
# Purpose: verify that the gemma_26b max_tokens=11000 bump + Stage R v2
# wiring (subgoal stack + Reflexion + planner HARD CONSTRAINT) all
# initialise and run end-to-end before committing to the 5h full sweep.
#
# Pass criteria:
#   - vLLM call succeeds (no LengthFinishReasonError on the bump)
#   - "[MACLA] seeded initial subgoal stack" appears in log
#   - "active_subgoal" / "HARD CONSTRAINT" reaches the prompt
#   - process exits cleanly, evaluation_summary.json written
#
# Runtime: ~5-10 min vs 60 min/iter for the full launcher.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
[[ -f "$REPO/.env" ]] && set -a && source "$REPO/.env" && set +a

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
AGENT_CFG="configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml"
TAG="stage_r_v2_spike"
RESULTS_DIR="experiments/stage_r_subgoals"
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/tmp/orak-stage-r-spike"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Preflights mirror full sweep
ASM_DIR="evaluation_utils/mcp_game_servers/pokemon_red/game/pokered/data/maps/objects"
asm_count=$(find "$ASM_DIR" -maxdepth 1 -name "*.asm" 2>/dev/null | wc -l)
[[ "$asm_count" -ge 100 ]] || { echo "FATAL: only $asm_count .asm files"; exit 1; }
[[ -s "executables/pokemon_red/pyboy/pokered.gbc" ]] || { echo "FATAL: ROM missing"; exit 1; }

served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
    | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
declared=$(grep '^model:' "$AGENT_CFG" | head -1 | sed 's/model: *"//;s/" *$//')
[[ "$served" == "$declared" ]] || { echo "FATAL: vLLM mismatch $served vs $declared"; exit 1; }

# Confirm the max_tokens bump landed in the config the spike will load
mt=$(grep -E '^max_tokens:' "$AGENT_CFG" | awk '{print $2}')
[[ "$mt" == "11000" ]] || { echo "FATAL: max_tokens not bumped (got $mt, expected 11000)"; exit 1; }
echo "[preflight] vLLM=$served, max_tokens=$mt, asm=$asm_count files"

# Short episode for spike — restore on exit
restore() { sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"; }
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 50/' "$ENV_CFG"
echo "[preflight] env max_steps temporarily set to 50 for spike"

run_id="${TAG}_$(date -u +%Y%m%dT%H%M%SZ)"
started=$(date +%s)
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] Stage R v2 spike — single iter, fresh"
echo "  run_id: $run_id"
echo "================================================================"

uv run python run.py \
    -c "$AGENT_CFG_NAME" \
    --local --games pokemon_red \
    --run-id "$run_id" \
    -d "Stage R v2 spike: max_tokens=11000 + subgoals+Reflexion smoke test"

rc=$?
elapsed=$(( ($(date +%s) - started) / 60 ))
actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
summary="$actual_dir/evaluation_summary.json"

echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] Stage R v2 SPIKE RESULT"
echo "================================================================"
echo "  exit_code:      $rc"
echo "  runtime_min:    $elapsed"
echo "  summary_exists: $([[ -s "$summary" ]] && echo YES || echo NO)"

# Wiring assertions — what we actually care about for the spike
if [[ -s "$summary" ]]; then
    score=$(python3 -c "
import json
d = json.load(open('$summary'))
eps = d.get('episodes', [])
print(max((float(e.get('final_score', 0.0)) for e in eps), default=0.0))
" 2>/dev/null)
    echo "  raw_score:      $score / 7.0"
fi
echo "================================================================"
exit $rc
