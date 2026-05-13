#!/usr/bin/env bash
# Stage K — cumulative cross-episode memory on pokemon Stage D, n=5.
#
# Hypothesis: Stages A→H all show convergence at 57.14% (4/7 milestones)
# on pokemon. The 4/7 milestones bank from in-town actions (starter, Pokedex,
# Mom, etc.); milestone 5+ requires leaving Pallet Town and navigating
# Viridian + Forest + Pewter Gym. Iter 2 of Stage H literally never left
# OaksLab but still scored 4/7.
#
# Each iter today starts FRESH — no memory of previous attempts. The
# checkpoint system *can* save/load the EnhancedHierarchicalMemorySystem
# (procedures + atomic memory) via --load-checkpoint + --prev-run-id, but
# no launcher uses this. Stage K wires that in.
#
# Mechanism:
#   iter 1: fresh start, saves checkpoints every 10 steps
#   iter 2: --load-checkpoint --prev-run-id <iter1>  → inherits iter1's memory
#   iter 3: --load-checkpoint --prev-run-id <iter2>  → inherits iter2's memory (which had iter1's)
#   ... (procedures + atomic memory compound)
#
# If the memory system captures useful generalisation (e.g., "warp pattern X
# escapes Pallet Town"), late iters should bank more milestones than early
# ones — a *learning curve* across iters rather than i.i.d. samples.
#
# Comparison baselines (all gemma-4-26B-A4B-it-AWQ, 300 steps, pokemon Stage D):
#   Stage D pure (PR #31, n=1):            57.14%
#   Stage B' no procedures (PR #69, n=3):  42.86% ± 14.29pp  (memory turned OFF)
#   Stage G procedure-escape (PR #70, n=3): 47.62% ± 16.49pp
#   Stage H Qwen 3.5 A3B-Int4 (n=3):       57.14% ± 0pp     (different model, same ceiling)
#   Stage K cumulative n=5 (here):         target: monotonic lift across iters
#
# Falsification criteria:
#   Flat curve [57, 57, 57, 57, 57] → memory doesn't capture generalisable knowledge
#   Rising curve [57, 57, 71, 71, 86] → memory IS the missing piece; late iters break ceiling
#   Crashing curve [57, 28, 14, 0, 0]  → memory captures BAD habits; carryover is anti-learning
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$AGENT_CFG" || {
    echo "FATAL: $AGENT_CFG not AWQ"; exit 1; }

# Pre-flight: vLLM should be serving gemma 26B AWQ on :8000
served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -oE '"id"[^,]+' | head -1 || true)
if [[ "$served" != *"gemma-4-26B-A4B-it-AWQ-4bit"* ]]; then
    echo "FATAL: vLLM not serving gemma-4-26B-A4B-it-AWQ-4bit"
    echo "  Currently serving: $served"
    echo "  Restart with:  ./serving/gemma_serve.sh cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
    exit 1
fi

restore() {
    echo "[restore] max_steps → 300"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

n=5
scores=()
prev_run_id=""

for iter in $(seq 1 $n); do
    run_id="pr_stage_j_cumulative_pokemon_iter${iter}_$(date -u +%Y%m%dT%H%M%SZ)"
    started=$(date +%s)
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] STAGE K n=$n iter $iter/$n"
    echo "  inherit from: ${prev_run_id:-NONE (fresh start)}"
    echo "  config:       gemma_26b + Stage D + cumulative memory"
    echo "  target:       monotonic lift across iters (n=$n)"
    echo "  run_id:       $run_id"
    echo "================================================================"

    # Build the run.py invocation; add --load-checkpoint --prev-run-id only for iters 2+
    cmd=(uv run python run.py
        -c gemma_26b
        --local --games pokemon_red
        --run-id "$run_id"
        -d "Stage K iter $iter: cumulative cross-episode memory (inherit from ${prev_run_id:-NONE})"
    )
    if [[ -n "$prev_run_id" ]]; then
        cmd+=(--load-checkpoint --prev-run-id "$prev_run_id")
    fi

    if ! "${cmd[@]}"; then
        echo "[FAIL] iter $iter exited non-zero"
        scores+=("0.0")
        # Carry the prev_run_id forward anyway — next iter still inherits last *successful* iter
        # (Don't update prev_run_id on failure)
        continue
    fi

    actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
    elapsed=$(( ( $(date +%s) - started ) / 60 ))
    summary="$actual_dir/evaluation_summary.json"
    score=$(python3 -c "
import json
d = json.load(open('$summary'))
eps = d.get('episodes', [])
raw = max((float(e.get('final_score', 0.0)) for e in eps), default=0.0)
print(f'{(raw/7.0)*100:.2f}')
" 2>/dev/null || echo "0.0")
    scores+=("$score")
    echo "[iter $iter] eval=${score}%, runtime=${elapsed}min, inherited_from=${prev_run_id:-NONE}"

    # Update prev_run_id for next iter
    prev_run_id="$run_id"
done

echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE K n=$n CUMULATIVE SUMMARY"
echo "================================================================"
python3 <<PYEOF
import json, statistics, datetime as dt
from pathlib import Path

scores = [$(IFS=,; echo "${scores[*]}")]
fmt = ", ".join(f"{s:.2f}%" for s in scores)
mean = statistics.mean(scores) if scores else 0
std = statistics.stdev(scores) if len(scores) > 1 else 0.0
# Learning-curve metric: did late iters score higher than early iters?
early_mean = statistics.mean(scores[:2]) if len(scores) >= 2 else scores[0]
late_mean = statistics.mean(scores[-2:]) if len(scores) >= 2 else scores[-1]
delta_curve = late_mean - early_mean

print(f"  Per-iter scores:  {fmt}")
print(f"  Mean +/- std:     {mean:.2f}% +/- {std:.2f}pp")
print(f"  Min/Max:          {min(scores):.2f}% / {max(scores):.2f}%")
print(f"  Early mean (1-2): {early_mean:.2f}%")
print(f"  Late mean (4-5):  {late_mean:.2f}%")
print(f"  Learning delta:   {delta_curve:+.2f} pp  ({'LIFT' if delta_curve > 7 else 'FLAT' if abs(delta_curve) <= 7 else 'REGRESS'})")
print(f"  vs Stage D 57.14%: {mean - 57.14:+.2f} pp")
print(f"  vs Stage H 57.14%: {mean - 57.14:+.2f} pp")

out = Path("experiments/stage_k_cumulative_memory/gemma_26b/results.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
existing = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
row = {
    "experiment": len(existing) + 1,
    "variant": f"stage_k_cumulative_memory_pokemon_n{len(scores)}",
    "game": "pokemon_red",
    "model": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
    "evaluation_score": mean,
    "evaluation_score_std": std,
    "evaluation_score_min": min(scores),
    "evaluation_score_max": max(scores),
    "early_mean": early_mean,
    "late_mean": late_mean,
    "learning_delta": delta_curve,
    "n_episodes": len(scores),
    "scores": scores,
    "steps": 300,
    "status": "KEEP",
    "description": f"Stage K cumulative memory n={len(scores)} pokemon Stage D (procedures + vmem inherited iter-to-iter via --load-checkpoint --prev-run-id)",
    "notes": f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores={fmt}. learning_delta={delta_curve:+.2f}pp.",
    "tags": ["cumulative_memory", "stage_k_cumulative_memory", f"stage_k_cumulative_memory_pokemon_n{len(scores)}"],
    "wandb_url": "",
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    "config_name": "gemma_26b",
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
