#!/usr/bin/env bash
# Stage L — map-aware procedure keys + iter-based TTL/decay (n=5).
#
# Tests whether the MACLA procedure-cache redesign (commit on
# feat/macla-map-aware-procedures branch) turns Stage K's negative transfer
# (iter 2 took +91 steps vs iter 1 to bank M4 in the post-asm-fix run) into
# neutral or positive transfer.
#
# Baseline = Stage K (post-asm-fix, master): n=5 cumulative memory on
# pokemon_red with the context-blind procedure cache. Expected pattern:
# 57.14% × 5 floor, but iter 2 slower than iter 1 to bank M4, iter 3 the
# most thrashy, no Viridian visits.
#
# Treatment = this Stage L: same n=5 cumulative memory but
#   - procedures keyed on (map_name, hash(steps))
#   - retrieval filters out wrong-map procedures
#   - prune_stale_procedures(max_age=2) on each checkpoint load
# Minimum bar: late_mean >= early_mean (no negative transfer).
# Lift bar: iter-over-iter steps-to-M4 decreasing, OR any iter reaching M5+.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
AGENT_CFG="configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml"
N=5
TAG="stage_l_map_aware"
RESULTS_DIR="experiments/stage_l_map_aware"
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/tmp/orak-stage-l-map-aware"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Pre-flights — same shape as run_pokemon_cumulative_rerun.sh
ASM_DIR="evaluation_utils/mcp_game_servers/pokemon_red/game/pokered/data/maps/objects"
asm_count=$(find "$ASM_DIR" -maxdepth 1 -name "*.asm" 2>/dev/null | wc -l)
[[ "$asm_count" -ge 100 ]] || { echo "FATAL: only $asm_count .asm files"; exit 1; }
echo "[preflight] $asm_count .asm files in pokered/data/maps/objects/"

[[ -s "executables/pokemon_red/pyboy/pokered.gbc" ]] || { echo "FATAL: ROM missing"; exit 1; }

# Pre-flight: vLLM serving Gemma 4-26B-A4B-AWQ-4bit
served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
    | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
declared=$(grep '^model:' "$AGENT_CFG" | head -1 | sed 's/model: *"//;s/" *$//')
if [[ "$served" != "$declared" ]]; then
    echo "FATAL: vLLM mismatch. declared=$declared served=$served"
    exit 1
fi
echo "[preflight] vLLM serving $served"

# Pre-flight: confirm we are running with the Stage L code (map-aware patch
# is in agents/macla/macla_lib.py). Cheap sanity check.
if ! grep -q "Stage L: map-aware procedure key" agents/macla/macla_lib.py; then
    echo "FATAL: agents/macla/macla_lib.py is missing the Stage L map-aware patch"
    echo "  Are you on the feat/macla-map-aware-procedures branch?"
    exit 1
fi
echo "[preflight] Stage L map-aware code present"

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
        -d "Stage L (map-aware procedures): iter $iter (inherit from ${prev_run_id:-NONE})"
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
    "description": f"Stage L: map-aware procedure keys + iter-based TTL/decay; {len(scores)}x pokemon cumulative memory",
    "notes": f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores=[{fmt}] learning_delta={delta:+.2f}pp",
    "tags": ["stage_l_map_aware", "cumulative_memory", "pokemon_red"],
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
