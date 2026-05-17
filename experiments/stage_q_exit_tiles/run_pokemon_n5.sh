#!/usr/bin/env bash
# Stage Q — every-step map-graph hint + exit-tile coordinates.
# n=5 cumulative-memory sweep against the post-asm-fix M5 ceiling.
#
# Baselines (post-asm-fix, 300 steps, n=5 cumulative):
#   Stage L  (PR #85) NEUTRAL+: [57.14, 57.14, 57.14, 28.57, 57.14], 51.43% mean
#   Stage M  (PR #86) FLAT:     [57.14, 57.14, 28.57, 57.14, 57.14], 51.43% mean
#   Stage N+O (PR #87) NEUTRAL+:[28.57, 57.14, 57.14, 57.14, 57.14], 51.43% mean
#   Stage P   (PR #90) FLAT:    [57.14, 57.14, 57.14, 57.14, 57.14], 57.14% mean, sigma=0
#       — 1,406 hint fires across 5 iters proved the planner consumed
#         the map-name hint, but agent stalled at Route1 (10, 35) every
#         iter — never found the exit tile. Bottleneck: planner has
#         destination name but no transition-tile coord.
#
# Stage Q intervention: render the exit tile directly into the hint —
#   "→ Route1: walk off the north edge"  (outdoor connection)
#   "→ OaksLab: walk to (12, 11)"        (indoor warp)
#
# Stage Q bars:
#   Minimum: any iter Viridian dwell > 0 OR final_map contains 'Viridian'.
#   Lift:    any iter past 57.14%.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# Load API keys if present
[[ -f "$REPO/.env" ]] && set -a && source "$REPO/.env" && set +a

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
AGENT_CFG="configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml"
N=5
TAG="stage_q_exit_tiles"
RESULTS_DIR="experiments/stage_q_exit_tiles"
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/tmp/orak-stage-q"
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

# Pre-flight: Stage Q code present
grep -q "def graph_hint" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: pokemon_red.game_adapter.graph_hint missing"; exit 1; }
grep -q "Exit tiles" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: Stage Q exit-tile section missing in adapter"; exit 1; }
grep -q "self._adapter, \"graph_hint\"" agents/macla/unified.py \
    || { echo "FATAL: unified.py not routing through adapter.graph_hint"; exit 1; }
grep -q "build_exit_tiles" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: adapter not using build_exit_tiles"; exit 1; }
echo "[preflight] Stage Q code present (adapter graph_hint + exit-tile rendering)"

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
        -d "Stage Q exit-tiles: iter $iter (inherit from ${prev_run_id:-NONE})"
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
    "description": f"Stage Q: exit-tile coordinates in graph_hint; {len(scores)}x pokemon cumulative memory",
    "notes": f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores=[{fmt}] learning_delta={delta:+.2f}pp",
    "tags": ["stage_q_exit_tiles", "cumulative_memory", "pokemon_red"],
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
