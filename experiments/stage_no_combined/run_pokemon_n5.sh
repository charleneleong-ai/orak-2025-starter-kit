#!/usr/bin/env bash
# Stage N + O — bootstrap-neutral signals, planner-side novelty,
# broaden acquisition by state-delta. n=5 cumulative-memory sweep.
#
# Baselines:
#   Stage L (PR #85) NEUTRAL+: [57.14, 57.14, 57.14, 28.57, 57.14],
#     mean 51.43%, M4 banking 259→229→172→140, 4 procs at end.
#   Stage M (PR #86) FLAT:     [57.14, 57.14, 28.57, 57.14, 57.14],
#     mean 51.43%, M4 banking 92→140→122→191 (no compounding),
#     4 procs at end, 1 successful_execution, 13 procedures_refined.
#
# Stage N+O bars:
#   Minimum: procedures_learned >= 50 by iter 5 (vs Stage M's 4).
#   Lift:    any iter past 57.14% OR Viridian (M5) entered.
#
# Smoke confirmed: 3 procs at step 50 / 0 successful_executions
# (pure state-delta acquisition); 4 novelty hint fires across map
# transitions; reached OaksLab by step 29 vs Stage L iter-1 baseline ~45.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
AGENT_CFG="configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml"
N=5
TAG="stage_no_combined"
RESULTS_DIR="experiments/stage_no_combined"
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/tmp/orak-stage-no-combined"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Pre-flights
ASM_DIR="evaluation_utils/mcp_game_servers/pokemon_red/game/pokered/data/maps/objects"
asm_count=$(find "$ASM_DIR" -maxdepth 1 -name "*.asm" 2>/dev/null | wc -l)
[[ "$asm_count" -ge 100 ]] || { echo "FATAL: only $asm_count .asm files"; exit 1; }
echo "[preflight] $asm_count .asm files in pokered/data/maps/objects/"
[[ -s "executables/pokemon_red/pyboy/pokered.gbc" ]] || { echo "FATAL: ROM missing"; exit 1; }

served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
    | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
declared=$(grep '^model:' "$AGENT_CFG" | head -1 | sed 's/model: *"//;s/" *$//')
[[ "$served" == "$declared" ]] || { echo "FATAL: vLLM mismatch $served vs $declared"; exit 1; }
echo "[preflight] vLLM serving $served"

# Pre-flight: Stage N + O code present
grep -q "_SDC_BOOTSTRAP_N = 3" agents/macla/macla_lib.py \
    || { echo "FATAL: Stage N _SDC_BOOTSTRAP_N missing"; exit 1; }
grep -q "def map_visit_status" agents/macla/macla_lib.py \
    || { echo "FATAL: Stage N map_visit_status missing"; exit 1; }
grep -q "### Novelty" agents/macla/unified.py \
    || { echo "FATAL: Stage N novelty wiring missing in unified.py"; exit 1; }
grep -q "_NEW_MAP_THETA" agents/macla/macla_lib.py \
    && { echo "FATAL: Stage M _NEW_MAP_THETA should be removed (Stage N cleanup)"; exit 1; } || true
grep -q "Stage O: broadens acquisition" agents/macla/macla_lib.py \
    || { echo "FATAL: Stage O acquisition broadening missing"; exit 1; }
echo "[preflight] Stage N + Stage O code present"

restore() { sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"; }
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

scores=()
prev_run_id=""

for iter in $(seq 1 $N); do
    run_id="${TAG}_iter${iter}_$(date -u +%Y%m%dT%H%M%SZ)"
    started=$(date +%s)
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] $TAG iter $iter/$N"
    echo "  inherit from: ${prev_run_id:-NONE (fresh)}"
    echo "  run_id:       $run_id"
    echo "================================================================"

    cmd=(uv run python run.py
        -c "$AGENT_CFG_NAME"
        --local --games pokemon_red
        --run-id "$run_id"
        -d "Stage N+O combined: iter $iter (inherit from ${prev_run_id:-NONE})"
    )
    [[ -n "$prev_run_id" ]] && cmd+=(--load-checkpoint --prev-run-id "$prev_run_id")

    if ! "${cmd[@]}"; then
        echo "[FAIL] iter $iter exited non-zero"
        scores+=("0.0")
        continue
    fi

    actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
    elapsed=$(( ($(date +%s) - started) / 60 ))
    summary="$actual_dir/evaluation_summary.json"
    score=$(python3 -c "
import json
d = json.load(open('$summary'))
eps = d.get('episodes', [])
raw = max((float(e.get('final_score', 0.0)) for e in eps), default=0.0)
print(f'{(raw/7.0)*100:.2f}')
" 2>/dev/null || echo "0.0")
    scores+=("$score")
    echo "[iter $iter] eval=${score}%, runtime=${elapsed}min, inherited=${prev_run_id:-NONE}"
    prev_run_id="$run_id"
done

echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] $TAG SUMMARY"
echo "================================================================"
python3 <<PYEOF
import json, statistics, datetime as dt
from pathlib import Path

scores = [$(IFS=,; echo "${scores[*]}")]
mean = statistics.mean(scores) if scores else 0
std = statistics.stdev(scores) if len(scores) > 1 else 0.0
early = statistics.mean(scores[:2]) if len(scores) >= 2 else scores[0]
late = statistics.mean(scores[-2:]) if len(scores) >= 2 else scores[-1]
delta = late - early
fmt = ", ".join(f"{s:.2f}%" for s in scores)
print(f"  Per-iter scores: {fmt}")
print(f"  Mean +/- std:    {mean:.2f}% +/- {std:.2f}pp")
print(f"  Early (1-2):     {early:.2f}%")
print(f"  Late (4-5):      {late:.2f}%")
print(f"  Learning delta:  {delta:+.2f}pp ({'LIFT' if delta>7 else 'FLAT' if abs(delta)<=7 else 'REGRESS'})")

out = Path("$RESULTS_DIR/results.jsonl")
row = {
    "experiment": 1,
    "variant": "$TAG",
    "game": "pokemon_red",
    "agent_config": "$AGENT_CFG_NAME",
    "evaluation_score": mean,
    "evaluation_score_std": std,
    "evaluation_score_min": min(scores),
    "evaluation_score_max": max(scores),
    "early_mean": early,
    "late_mean": late,
    "learning_delta": delta,
    "n_episodes": len(scores),
    "scores": scores,
    "steps": 300,
    "status": "KEEP",
    "description": f"Stage N+O combined: bootstrap-neutral signals + planner novelty + state-delta acquisition; {len(scores)}x pokemon cumulative memory",
    "notes": f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores=[{fmt}] learning_delta={delta:+.2f}pp",
    "tags": ["stage_no_combined", "cumulative_memory", "pokemon_red"],
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
