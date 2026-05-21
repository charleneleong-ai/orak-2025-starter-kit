#!/usr/bin/env bash
# Stage S Step 1 — no-inherit baseline. HYPOTHESIS FALSIFIED — KEPT FOR REPLICATION.
#
# Result: 57.14% × 5 (FLAT, std 0.0). Cache inheritance was never the wall.
# Full diagnosis: docs/experiments/stage_s_cache_veto/step1_n5_introspection.md
# Reverts on this branch: ef1ab2f (cache veto impl) + e5c3103 (Step 2 launcher).
# The actual wall (Pallet → Viridian transition, score-based EnterViridian
# with no spatial pull) was diagnosed from this sweep's per-iter logs and
# fixed in v1 (commit 1811154): NavigateToMap("ViridianCity") subgoal bridge.
#
# Originally intended as the upper-bound check for the cache-veto hypothesis:
# every iter --load-checkpoint disabled, each iter a fresh slate. The
# falsification came from 5 iters all bouncing Route1 → PalletTown without
# ever entering Viridian — diagnostic signal, not a fix.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

# Load API keys if present
[[ -f "$REPO/.env" ]] && set -a && source "$REPO/.env" && set +a

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
AGENT_CFG="configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml"
N=5
TAG="stage_s_cache_veto_step1_no_inherit"
RESULTS_DIR="experiments/stage_s_cache_veto_step1_no_inherit"
MAX_STEPS=600
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/workspace/orak-stage-s-step1"
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

# Pre-flight: v5 lever stack still intact (Stage S = v5 levers; the cache-veto
# code that was here originally was reverted on this branch — ef1ab2f / e5c3103 —
# after Step 1's FLAT result falsified the hypothesis).
grep -q "^PROC_CACHE_MIN_ITER_SCORE = 5.0" agents/pokemon_red/game_adapter.py \
    || { echo "FATAL: PROC_CACHE_MIN_ITER_SCORE not 5.0"; exit 1; }
echo "[preflight] PROC_CACHE_MIN_ITER_SCORE=5.0 (M5 / EnterViridian gate)"

# v4 levers still required
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
    # autoresearch-current-run "default" log format: [ts] Iter N/M: rest
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Iter $iter/$N: $TAG inherit_from=${prev_run_id:-NONE} run_id=$run_id"
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] $TAG iter $iter/$N"
    echo "  inherit from: ${prev_run_id:-NONE (fresh)}"
    echo "  run_id:       $run_id"
    echo "================================================================"

    cmd=(uv run python run.py
        -c "$AGENT_CFG_NAME"
        --local --games pokemon_red
        --run-id "$run_id"
        -d "Stage S Step 1 no-inherit baseline (every iter fresh): iter $iter"
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
    # autoresearch-current-run "default" iter-done marker
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Iter $iter/$N finished"

    # Per-iter results.jsonl row via autoresearch.log_experiment — the
    # canonical schema (matches gemma4-rlvr and any future sweep). Mid-sweep
    # cancellation now leaves a record because the row writes as soon as
    # each iter completes (v2 symptom was only the stale v1 row surviving).
    ITER=$iter SCORE=$score ELAPSED=$elapsed RUN_ID="$run_id" \
    PREV_RUN_ID="$prev_run_id" TAG="$TAG" N_ITERS=$N STEPS=$MAX_STEPS \
    .venv/bin/python - <<'PYEOF'
import os
from autoresearch import log_experiment

log_experiment(
    experiments_dir="experiments",
    tag=os.environ["TAG"],
    game="pokemon_red",
    score=float(os.environ["SCORE"]),
    steps=int(os.environ["STEPS"]),
    status="IN_PROGRESS",
    description=f"Stage S Step 1 no-inherit iter {os.environ['ITER']}/{os.environ['N_ITERS']} (fresh)",
    runtime_min=float(os.environ["ELAPSED"]),
    extra={
        "record_type": "per_iter",
        "iter": int(os.environ["ITER"]),
        "n_iters": int(os.environ["N_ITERS"]),
        "run_id": os.environ["RUN_ID"],
        "inherited_from": os.environ["PREV_RUN_ID"] or None,
    },
)
PYEOF
    # Stage S Step 1: deliberately DO NOT propagate prev_run_id — every
    # iter is a fresh slate. This is the diagnostic upper bound.
done

echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] $TAG SUMMARY"
echo "================================================================"
SCORES_CSV=$(IFS=,; echo "${scores[*]}") \
TAG="$TAG" AGENT_CFG_NAME="$AGENT_CFG_NAME" STEPS=$MAX_STEPS \
.venv/bin/python <<'PYEOF'
import os, statistics
from autoresearch import log_experiment

scores = [float(s) for s in os.environ["SCORES_CSV"].split(",") if s]
mean = statistics.mean(scores) if scores else 0.0
std = statistics.stdev(scores) if len(scores) > 1 else 0.0
early = statistics.mean(scores[:2]) if len(scores) >= 2 else (scores[0] if scores else 0.0)
late = statistics.mean(scores[-2:]) if len(scores) >= 2 else (scores[-1] if scores else 0.0)
delta = late - early
fmt = ", ".join(f"{s:.2f}%" for s in scores)
print(f"  Per-iter scores: {fmt}")
print(f"  Mean +/- std:    {mean:.2f}% +/- {std:.2f}pp")
print(f"  Early (1-2):     {early:.2f}%")
print(f"  Late (4-5):      {late:.2f}%")
print(f"  Learning delta:  {delta:+.2f}pp ({'LIFT' if delta>7 else 'FLAT' if abs(delta)<=7 else 'REGRESS'})")

tag = os.environ["TAG"]
steps = int(os.environ["STEPS"])
results_path = log_experiment(
    experiments_dir="experiments",
    tag=tag,
    game="pokemon_red",
    score=mean,
    steps=steps,
    status="KEEP",
    description=(
        f"Stage S Step 1 no-inherit baseline: v5 lever stack, budget{steps}, "
        f"every iter fresh (diagnostic upper bound for cache-veto comparison)"
    ),
    notes=f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores=[{fmt}] learning_delta={delta:+.2f}pp",
    extra={
        "record_type": "sweep_summary",
        "agent_config": os.environ["AGENT_CFG_NAME"],
        "evaluation_score_std": std,
        "evaluation_score_min": min(scores) if scores else 0.0,
        "evaluation_score_max": max(scores) if scores else 0.0,
        "early_mean": early,
        "late_mean": late,
        "learning_delta": delta,
        "n_episodes": len(scores),
        "scores": scores,
        "tags": [tag, "cumulative_memory", "pokemon_red"],
    },
)
print(f"  Appended sweep_summary row to {results_path}")
PYEOF
echo "================================================================"
