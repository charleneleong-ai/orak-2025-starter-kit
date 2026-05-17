#!/usr/bin/env bash
# Stage R — performance-gated proc-cache prune (exit-tile hint + Stage R prune).
# n=5 cumulative-memory sweep against the Stage Q regression.
#
# Baselines (post-asm-fix, 300 steps, n=5 cumulative):
#   Stage P   (PR #90) FLAT:    [57.14, 57.14, 57.14, 57.14, 57.14], 57.14% mean, sigma=0
#   Stage Q   (PR #92) REGRESS: [71.43, 57.14, 28.57, 28.57, 28.57(?)] — iter 1
#       lifted past M5 (71.43%, first time across all stages) but iters
#       3-5 collapsed back to PalletTown and never escaped. Diagnosis
#       via autoresearch.trajectory introspect: each iter inherits the
#       prior iter's proc cache; bad iters add PalletTown-loiter procs
#       that survive Stage L's age-based prune and trap late iters.
#
# Stage R intervention: on checkpoint load, BEFORE bumping iter, drop
# every proc whose origin_iter == prev_iter if that iter scored below
# the per-game M4 threshold (4/7 for pokemon).
#
# Stage R bars:
#   Minimum: sigma(scores) lower than Stage Q's ~21pp variance.
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
TAG="stage_r_perf_prune"
RESULTS_DIR="experiments/stage_r_perf_prune"
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/tmp/orak-stage-r"
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

# Pre-flight: Stage Q + R code present
grep -q "def graph_hint" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: pokemon_red.game_adapter.graph_hint missing"; exit 1; }
grep -q "Exit tiles" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: Stage Q exit-tile section missing in adapter"; exit 1; }
grep -q "self._adapter, \"graph_hint\"" agents/macla/unified.py \
    || { echo "FATAL: unified.py not routing through adapter.graph_hint"; exit 1; }
grep -q "build_exit_tiles" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: adapter not using build_exit_tiles"; exit 1; }
echo "[preflight] Stage Q code present (adapter graph_hint + exit-tile rendering)"

# Pre-flight: Stage R code present
grep -q "def prune_low_score_iter" agents/macla/macla_lib.py \
    || { echo "FATAL: Stage R prune_low_score_iter missing"; exit 1; }
grep -q "PROC_CACHE_MIN_ITER_SCORE" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: pokemon adapter PROC_CACHE_MIN_ITER_SCORE missing"; exit 1; }
grep -q "origin_iter" agents/macla/macla_lib.py \
    || { echo "FATAL: ProceduralMemoryEntry origin_iter field missing"; exit 1; }
grep -q "prune_low_score_iter" agents/macla/base.py \
    || { echo "FATAL: base.py not wired to call prune_low_score_iter on load"; exit 1; }
grep -q "mem.last_iter_score = float(score)" agents/macla/base.py \
    || { echo "FATAL: base.py not recording per-iter score at episode end"; exit 1; }
echo "[preflight] Stage R code present (origin_iter tagging + prune_low_score_iter wiring)"

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
        -d "Stage R perf-prune: iter $iter (inherit from ${prev_run_id:-NONE})"
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
    "description": f"Stage R: exit-tile hint + performance-gated proc-cache prune; {len(scores)}x pokemon cumulative memory",
    "notes": f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores=[{fmt}] learning_delta={delta:+.2f}pp",
    "tags": ["stage_r_perf_prune", "cumulative_memory", "pokemon_red"],
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
