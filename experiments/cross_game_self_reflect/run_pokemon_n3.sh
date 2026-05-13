#!/usr/bin/env bash
# n=3 variance retest of pokemon Stage D + self-reflection (PR #64 follow-up #4).
#
# All pokemon Stage D-class runs to date have been n=1 / 300 steps. Single
# episode variance on pokemon is ~14pp (one milestone). The "tied" headline
# scores (Stage D=Stage D+reflect at 57.14%) aren't robust at n=1.
#
# This launcher runs the same Stage D + self-reflection config three times
# with different timestamps (effectively different LLM-sampling seeds at
# T=0.7), aggregates into a per-iteration row in results.jsonl, and prints
# mean ± std + min/max for the comparison report.
#
# Adapter recommendation: pokemon's adapter says use_self_reflection=True,
# reflect_every=10. YAML doesn't override, so the adapter wins on every iter.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/gemma_26b.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$AGENT_CFG" || {
    echo "FATAL: $AGENT_CFG not AWQ"; exit 1; }

restore() {
    echo "[restore] max_steps → 300"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

n=3
scores=()
for iter in $(seq 1 $n); do
    run_id="pr64_pokemon_d_reflect_n3_iter${iter}_$(date -u +%Y%m%dT%H%M%SZ)"
    game_logs="$GAME_DATA_DIR/game_logs/pokemon_red/$run_id"
    started=$(date +%s)

    echo
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] N=3 VARIANCE iter $iter/$n"
    echo "  config: Stage D + self-reflect (adapter default)"
    echo "  run_id=$run_id"
    echo "================================================================"

    if ! uv run python run.py \
        -c gemma_26b \
        --local --games pokemon_red \
        --run-id "$run_id" \
        -d "Pokemon Stage D + reflect, n=3 variance iter $iter — PR #64 follow-up #4"; then
        echo "[FAIL] iter $iter exited non-zero — recording 0 and continuing"
        scores+=("0.0")
        continue
    fi

    actual_dir="$GAME_DATA_DIR/pokemon_red/$run_id"
    [ -d "$actual_dir" ] && ln -sfn "$actual_dir" "$game_logs"

    elapsed=$(( ( $(date +%s) - started ) / 60 ))
    summary="$actual_dir/evaluation_summary.json"
    score=$(python3 -c "
import json
d = json.load(open('$summary'))
eps = d.get('episodes', [])
scores = [float(e.get('final_score', 0.0)) for e in eps] if eps else [0]
raw = max(scores)
# Pokemon final_score is raw 0..7, convert to %
print(f'{(raw/7.0)*100:.2f}')
" 2>/dev/null || echo "0.0")
    scores+=("$score")
    echo "[iter $iter] eval=${score}%, runtime=${elapsed}min"
done

echo
echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] N=3 VARIANCE SUMMARY"
echo "================================================================"
python3 <<PYEOF
import json
import statistics
from pathlib import Path
import datetime as dt

scores = [${scores[@]}]
fmt_scores = ", ".join(f"{s:.2f}%" for s in scores)
mean = statistics.mean(scores) if scores else 0
std = statistics.stdev(scores) if len(scores) > 1 else 0.0
print(f"  Scores:  {fmt_scores}")
print(f"  Mean ± std: {mean:.2f}% ± {std:.2f}")
print(f"  Min/Max: {min(scores):.2f}% / {max(scores):.2f}%")
print(f"  Stage D baseline (n=1): 57.14%")
print(f"  Delta vs baseline: {mean - 57.14:+.2f} pp (with {std:.2f}pp std at n={len(scores)})")

# Persist to results.jsonl
out = Path("experiments/cross_game_self_reflect/gemma_26b/results.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
existing = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
row = {
    "experiment": len(existing) + 1,
    "variant": "stage_d_self_reflect_pokemon_n3",
    "game": "pokemon_red",
    "evaluation_score": mean,
    "evaluation_score_std": std,
    "evaluation_score_min": min(scores),
    "evaluation_score_max": max(scores),
    "n_episodes": len(scores),
    "scores": scores,
    "steps": 300,
    "runtime_min": 0,  # cumulative not tracked here
    "status": "KEEP",
    "description": "Pokemon Stage D + self-reflect, n=3 variance retest (PR #64 follow-up #4)",
    "notes": f"n=3: mean={mean:.2f}% std={std:.2f}pp scores={fmt_scores}",
    "tags": ["cross_game_self_reflect", "stage_d_self_reflect_pokemon_n3"],
    "wandb_url": "",
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    "config_name": "gemma_26b",
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] N=3 VARIANCE DONE"
echo "================================================================"
