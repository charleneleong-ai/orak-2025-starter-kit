#!/usr/bin/env bash
# Stage R — hierarchical subgoal stack + Reflexion (off master).
# n=5 cumulative-memory sweep against the M4/M5 ceilings.
#
# Diagnosis (Stage Q v1, PR #92): exit-tile hint reached the planner
# 1,406 times across n=5 but the executor never committed to the named
# subtask. iter 1 lifted to 71.43% (M5) then iters 3-5 collapsed to
# PalletTown — proc cache + flat subtask hint together couldn't break
# the long-horizon planning bottleneck.
#
# Stage R intervention (two combined architectural lifts):
#   1) Hierarchical subgoal stack with explicit completion predicates.
#      Adapter ships SUBGOAL_TEMPLATES (NavigateToMap, TalkTo,
#      DefeatTrainer) + initial_subgoal_stack() — pokemon defaults to
#      [ViridianCity, Route1] (top of stack = Route1, the immediate
#      next step). Per step the executor checks the top subgoal's
#      completion(obs) and pops on fire. The active subgoal renders
#      directly into the action LLM's prompt as "[Active subgoal —
#      pursue this until its completion predicate fires]".
#   2) Reflexion summary built from prev iter's game_states.jsonl via
#      autoresearch.trajectory.extract_iter_metrics, prepended to the
#      subtask planner's history at episode start. Tells the planner
#      "Iter N-1 you hit M2@49 then stalled in PalletTown for 250 steps
#      — hypothesise why and try differently."
#
# Stage R bars:
#   Minimum: sigma(scores) lower than Stage Q v1's 20.20pp variance
#            AND no iter scores at 0.0.
#   Lift:    >=2 iters past 57.14% OR mean > 57.14%.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

# Load API keys if present
[[ -f "$REPO/.env" ]] && set -a && source "$REPO/.env" && set +a

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
AGENT_CFG="configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml"
N=5
TAG="stage_r_subgoals"
RESULTS_DIR="experiments/stage_r_subgoals"
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/tmp/orak-stage-r-subgoals"
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

# Pre-flight: Stage R code present
grep -q "class Subgoal" agents/macla/macla_lib.py \
    || { echo "FATAL: Subgoal dataclass missing"; exit 1; }
grep -q "def push_subgoal" agents/macla/macla_lib.py \
    || { echo "FATAL: subgoal_stack methods missing"; exit 1; }
grep -q "def check_active_subgoal_completion" agents/macla/macla_lib.py \
    || { echo "FATAL: subgoal completion check missing"; exit 1; }
grep -q "SUBGOAL_TEMPLATES" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: pokemon SUBGOAL_TEMPLATES missing"; exit 1; }
grep -q "def initial_subgoal_stack" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: pokemon initial_subgoal_stack missing"; exit 1; }
grep -q "build_reflexion_summary" agents/macla/unified.py \
    || { echo "FATAL: unified.py not importing Reflexion helper"; exit 1; }
grep -q "_init_episode_subgoals" agents/macla/unified.py \
    || { echo "FATAL: unified.py episode-init hook missing"; exit 1; }
grep -q "check_active_subgoal_completion" agents/macla/unified.py \
    || { echo "FATAL: unified.py per-step completion check missing"; exit 1; }
grep -q "active_subgoal=" agents/macla/unified.py \
    || { echo "FATAL: unified.py not threading active_subgoal into planner"; exit 1; }
echo "[preflight] subgoal + Reflexion + planner-constraint code present"

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
        -d "Stage R subgoals+Reflexion: iter $iter (inherit from ${prev_run_id:-NONE})"
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
    "description": f"Stage R: hierarchical subgoals + Reflexion; {len(scores)}x pokemon cumulative memory",
    "notes": f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores=[{fmt}] learning_delta={delta:+.2f}pp",
    "tags": ["stage_r_subgoals", "cumulative_memory", "pokemon_red"],
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
