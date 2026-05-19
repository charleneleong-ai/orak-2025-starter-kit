#!/usr/bin/env bash
# Stage R v4 — five-lever stack on top of v3.
#
# v3 introspection (docs/experiments/stage_r_subgoals/v3_n5_introspection.md)
# showed cumulative memory works (iter 5 hit milestones 13-44% faster than
# iter 1, +33 procedures, +1142 refinements) but every iter pinned at
# 57.14% (M5/Viridian). Five concurrent levers in v4:
#
#   0) Adapter graph_hint wiring — unified.py was still calling the
#      hand-authored ~30-map MAP_GRAPH in macla_lib. Swapped to the
#      adapter's auto-extracted 221-map + 404-exit-tile path. Stage Q's
#      exit-tile coords ("walk to (X, Y)") now actually reach the planner.
#   1) Anti-perseveration position hint — track (map, x, y) visit count;
#      surface "### Recently looped" when threshold=5 crossed. v3 had
#      single tiles revisited 44× without the planner noticing.
#   3) Step budget 300 → 600 — iter 5 hit score-4.0 at step 198/300, with
#      only ~100 steps left to chase any M6 milestone. Doubled budget.
#   4) Per-iter reset of stagnation + position counters via __setstate__.
#      v3 iter 2 started at stagnation=440 from iter 1's tail; same
#      shape now precluded for position visits.
#   5) Perf-prune write site — last_iter_score wasn't being written, so
#      prune_low_score_iter has been a no-op forever. Wired into
#      record_episode_end.
#
# F4 (move_to boundary detection) is deliberately deferred to Stage S
# for clean attribution.
#
# Stage R v4 bars:
#   Minimum: no iter < 50% (don't regress v3).
#   Lift:    >=2 iters past 57.14% OR mean > 57.14% (break the ceiling).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# Load API keys if present
[[ -f "$REPO/.env" ]] && set -a && source "$REPO/.env" && set +a

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
AGENT_CFG="configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml"
N=5
TAG="stage_r_subgoals_v4"
RESULTS_DIR="experiments/stage_r_subgoals_v4"
MAX_STEPS=600
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/tmp/orak-stage-r-subgoals-v4"
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

# Pre-flight: Stage R v4 code present
# (0) Adapter wiring + hand-authored MAP_GRAPH removed
grep -q "self._adapter" agents/macla/unified.py \
    || { echo "FATAL: unified.py doesn't reach adapter"; exit 1; }
grep -q 'getattr(self._adapter, "graph_hint"' agents/macla/unified.py \
    || { echo "FATAL: v4(0) adapter graph_hint dispatch missing"; exit 1; }
grep -q "mem.visited_maps" agents/macla/unified.py \
    || { echo "FATAL: v4(0) graph_hint not passing visited_maps"; exit 1; }
grep -q "^MAP_GRAPH" agents/macla/macla_lib.py \
    && { echo "FATAL: v4(0) hand-authored MAP_GRAPH must be deleted from macla_lib"; exit 1; }
grep -q "def map_graph_hint" agents/macla/macla_lib.py \
    && { echo "FATAL: v4(0) map_graph_hint method must be deleted from macla_lib"; exit 1; }
echo "[preflight] v4(0) adapter graph_hint wired, hand-authored path removed"

# (1) Anti-perseveration
grep -q "def record_position" agents/macla/macla_lib.py \
    || { echo "FATAL: v4(1) record_position missing"; exit 1; }
grep -q "def looped_positions_hint" agents/macla/macla_lib.py \
    || { echo "FATAL: v4(1) looped_positions_hint missing"; exit 1; }
grep -q "looped_positions_hint" agents/macla/unified.py \
    || { echo "FATAL: v4(1) unified.py not calling looped_positions_hint"; exit 1; }
echo "[preflight] v4(1) anti-perseveration counter + hint wired"

# (4) __setstate__ reset
grep -q "def __setstate__" agents/macla/macla_lib.py \
    || { echo "FATAL: v4(4) __setstate__ reset missing"; exit 1; }
echo "[preflight] v4(4) per-iter reset on checkpoint load"

# (5) Perf-prune write
grep -q "last_iter_score = float(score)" agents/macla/unified.py \
    || { echo "FATAL: v4(5) record_episode_end not writing last_iter_score"; exit 1; }
echo "[preflight] v4(5) perf-prune write site wired"

# Carry-overs from v3 (still expected to be present)
grep -q "SUBGOAL_STAGNATION_THRESHOLD" agents/macla/unified.py \
    || { echo "FATAL: v3 escape valve threshold missing"; exit 1; }
grep -q "### Currently pursuing" agents/_cognitive/subtask_planner.py \
    || { echo "FATAL: v3 soft phrasing missing"; exit 1; }
echo "[preflight] v3 carry-overs (escape valve, soft phrasing) present"

# (6) Subgoal stack extended to full M5-M7 ladder via generic
# build_score_milestone_stack — pokemon adapter is data-only.
grep -q "def build_score_milestone_stack" agents/macla/macla_lib.py \
    || { echo "FATAL: v4(6) generic helper missing in macla_lib"; exit 1; }
grep -q "_POKEMON_MILESTONE_LIBRARY" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: v4(6) pokemon milestone library missing"; exit 1; }
grep -q "build_score_milestone_stack" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: v4(6) pokemon adapter not using generic helper"; exit 1; }
echo "[preflight] v4(6) M5-M7 subgoal ladder + generic framework helper wired"

# (3) Step budget bump 300 → 600 — applied to env config with restore trap
restore() { sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"; }
trap restore EXIT
sed -i "s/^max_steps: .*/max_steps: $MAX_STEPS/" "$ENV_CFG"
grep -q "^max_steps: $MAX_STEPS" "$ENV_CFG" \
    || { echo "FATAL: v4(3) max_steps bump to $MAX_STEPS didn't take"; exit 1; }
echo "[preflight] v4(3) max_steps=$MAX_STEPS"

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
        -d "Stage R v4 (adapter graph_hint + anti-perseveration + budget600 + setstate-reset + perf-prune-fix): iter $iter (inherit from ${prev_run_id:-NONE})"
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
    "steps": $MAX_STEPS,
    "status": "KEEP",
    "description": f"Stage R v4: adapter graph_hint + anti-perseveration + budget$MAX_STEPS + setstate-reset + perf-prune-fix; {len(scores)}x pokemon cumulative memory",
    "notes": f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores=[{fmt}] learning_delta={delta:+.2f}pp",
    "tags": ["stage_r_subgoals_v4", "cumulative_memory", "pokemon_red"],
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
