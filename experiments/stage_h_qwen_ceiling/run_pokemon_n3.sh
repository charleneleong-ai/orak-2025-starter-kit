#!/usr/bin/env bash
# Stage H — Qwen 3.6 27B FP8 ceiling check on pokemon Stage D, n=3.
#
# Same MACLA stack as gemma_26b (Stage D: vmem + planner + procedures ON,
# self-reflection per-game adapter default). Single variable: model swap from
# Gemma 4-26B-A4B-it-AWQ-4bit to Qwen3.6-27B-FP8.
#
# PRE-FLIGHT (manual): stop the gemma vLLM and start the qwen vLLM:
#   pkill -f 'vllm.entrypoints.openai.api_server'
#   nohup ./serving/qwen_serve.sh >/tmp/qwen_serve.log 2>&1 &
#   # wait until http://localhost:8000/v1/models responds
#
# Comparison baselines (all 300 steps, n=3 where shown):
#   Stage D (Gemma 26B AWQ, n=1):                 57.14%
#   Stage B' no procedures (Gemma, n=3):          42.86% ± 14.29pp
#   Stage G procedure-escape (Gemma, n=3):        47.62% ± 16.49pp
#   Stage H Qwen 3.6 27B (this run, n=3):         target ≥ 71.43% (5/7)
#
# Decision criteria:
#   mean >= 71.43%  → Stage D ceiling was model capacity; Qwen 3.6 breaks it
#   mean ≈ 57.14%   → ceiling is prompt/scaffold; pivot to planner-prompt overhaul
#   mean <  42%     → Qwen 3.6 underperforms on this task at fp8; check tool-call parser
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/qwen35_a3b_int4.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

grep -q "Qwen/Qwen3" "$AGENT_CFG" || {
    echo "FATAL: $AGENT_CFG not Qwen 3.x"; exit 1; }

# Pre-flight: is qwen vLLM serving on :8000?
if ! curl -s --max-time 3 http://localhost:8000/v1/models | grep -qi "qwen3"; then
    echo "FATAL: vLLM at :8000 is not serving Qwen 3.x"
    echo "  Stop gemma:  pkill -f 'vllm.entrypoints.openai.api_server'"
    echo "  Start qwen:  nohup ./serving/qwen_serve.sh >/tmp/qwen_serve.log 2>&1 &"
    echo "  Wait until:  curl -s http://localhost:8000/v1/models | grep -i qwen3"
    exit 1
fi

restore() {
    echo "[restore] max_steps → 300"
    sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"
}
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

n=3
scores=()
for iter in $(seq 1 $n); do
    run_id="pr_stage_h_qwen35_a3b_pokemon_iter${iter}_$(date -u +%Y%m%dT%H%M%SZ)"
    started=$(date +%s)
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] STAGE H n=3 iter $iter/$n"
    echo "  config:  qwen36_27b + Stage D (vmem + planner + procedures ON)"
    echo "  target:  >= 71.43% (5/7)"
    echo "  run_id:  $run_id"
    echo "================================================================"

    if ! uv run python run.py \
        -c qwen35_a3b_int4 \
        --local --games pokemon_red \
        --run-id "$run_id" \
        -d "Stage H iter $iter: Qwen 3.5 35B-A3B-Int4 ceiling check on pokemon Stage D"; then
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
echo "[$(date -u +%H:%M:%SZ)] STAGE H n=3 SUMMARY"
echo "================================================================"
python3 <<PYEOF
import json, statistics, datetime as dt
from pathlib import Path

scores = [$(IFS=,; echo "${scores[*]}")]
fmt = ", ".join(f"{s:.2f}%" for s in scores)
mean = statistics.mean(scores) if scores else 0
std = statistics.stdev(scores) if len(scores) > 1 else 0.0
print(f"  Scores:                          {fmt}")
print(f"  Mean +/- std:                    {mean:.2f}% +/- {std:.2f}pp")
print(f"  Min/Max:                         {min(scores):.2f}% / {max(scores):.2f}%")
print(f"  Gemma Stage D baseline (n=1):    57.14%")
print(f"  Gemma Stage B' (no procs) n=3:   42.86% +/- 14.29pp")
print(f"  Gemma Stage G (proc-escape) n=3: 47.62% +/- 16.49pp")
print(f"  Delta vs Stage D:                {mean - 57.14:+.2f} pp")
print(f"  Target (>= 71.43%):              {'HIT' if mean >= 71.43 else 'MISS'}")

out = Path("experiments/stage_h_qwen_ceiling/qwen35_a3b_int4/results.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
existing = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
row = {
    "experiment": len(existing) + 1,
    "variant": "stage_h_qwen35_a3b_int4_pokemon_n3",
    "game": "pokemon_red",
    "model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    "evaluation_score": mean,
    "evaluation_score_std": std,
    "evaluation_score_min": min(scores),
    "evaluation_score_max": max(scores),
    "n_episodes": len(scores),
    "scores": scores,
    "steps": 300,
    "status": "KEEP",
    "description": "Stage H ceiling check: Qwen 3.5 35B-A3B-GPTQ-Int4 on pokemon Stage D (procedures ON, planner ON, vmem ON)",
    "notes": f"n=3: mean={mean:.2f}% std={std:.2f}pp scores={fmt}. delta_vs_gemma_stage_d={mean-57.14:+.2f}pp",
    "tags": ["qwen", "qwen3.5-35b-a3b-int4", "stage_h_ceiling_check", "stage_h_qwen35_a3b_int4_pokemon_n3"],
    "wandb_url": "",
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    "config_name": "qwen35_a3b_int4",
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
