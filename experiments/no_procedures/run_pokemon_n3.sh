#!/usr/bin/env bash
# Stage B' pokemon n=3 — raw-LLM-with-reflection baseline (no procedure cache).
#
# Tests competing hypothesis to PR #68: maybe the 57.14% pokemon ceiling is
# upstream of the procedure layer (LLM/planner combo capped at milestone 4
# regardless of architecture). Disable ONLY the procedure cache; keep
# self-reflection + planner + vmem ON. n=3 to settle variance vs n=1 ties.
#
# Read of outcomes:
#   mean ≈ 57.14% with low std       → LLM/planner ceiling is real;
#                                       procedure-layer fixes won't help
#   mean > 57.14% OR high variance   → procedure cache was suppressing
#                                       exploration; dropping it is the play
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b_no_procedures.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$AGENT_CFG" || {
    echo "FATAL: $AGENT_CFG not AWQ"; exit 1; }
grep -q "use_procedure_layer: false" "$AGENT_CFG" || {
    echo "FATAL: $AGENT_CFG missing use_procedure_layer: false"; exit 1; }

restore() {
    echo "[restore] max_steps → 300"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

n=3
scores=()
for iter in $(seq 1 $n); do
    run_id="pr_nopr_stage_b_prime_pokemon_iter${iter}_$(date -u +%Y%m%dT%H%M%SZ)"
    started=$(date +%s)
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] STAGE B' n=3 iter $iter/$n"
    echo "  run_id=$run_id"
    echo "================================================================"

    if ! uv run python run.py \
        -c gemma_26b_no_procedures \
        --local --games pokemon_red \
        --run-id "$run_id" \
        -d "Stage B' iter $iter: pokemon Stage D - procedure cache (reflection/planner/vmem ON)"; then
        echo "[FAIL] iter $iter exited non-zero"
        scores+=("0.0")
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
    echo "[iter $iter] eval=${score}%, runtime=${elapsed}min"
done

echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE B' n=3 SUMMARY"
echo "================================================================"
python3 <<PYEOF
import json, statistics, datetime as dt
from pathlib import Path

scores = [${scores[@]}]
fmt = ", ".join(f"{s:.2f}%" for s in scores)
mean = statistics.mean(scores) if scores else 0
std = statistics.stdev(scores) if len(scores) > 1 else 0.0
print(f"  Scores:  {fmt}")
print(f"  Mean +/- std: {mean:.2f}% +/- {std:.2f}")
print(f"  Min/Max: {min(scores):.2f}% / {max(scores):.2f}%")
print(f"  Stage D baseline (n=1):    57.14%")
print(f"  Delta vs baseline: {mean - 57.14:+.2f} pp (with {std:.2f}pp std)")

out = Path("experiments/no_procedures/gemma_26b/results.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
existing = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
row = {
    "experiment": len(existing) + 1,
    "variant": "stage_b_prime_no_procedures_pokemon_n3",
    "game": "pokemon_red",
    "evaluation_score": mean,
    "evaluation_score_std": std,
    "evaluation_score_min": min(scores),
    "evaluation_score_max": max(scores),
    "n_episodes": len(scores),
    "scores": scores,
    "steps": 300,
    "status": "KEEP",
    "description": "Stage B' n=3 pokemon: Stage D minus procedure cache (reflection/planner/vmem ON)",
    "notes": f"n=3: mean={mean:.2f}% std={std:.2f}pp scores={fmt}",
    "tags": ["no_procedures", "stage_b_prime_no_procedures_pokemon_n3"],
    "wandb_url": "",
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    "config_name": "gemma_26b_no_procedures",
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
