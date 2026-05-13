#!/usr/bin/env bash
# Stage K — Qwen3-30B-A3B-Thinking-2507-AWQ-4bit n=3 on pokemon Stage D.
#
# Tests whether explicit thinking-mode reasoning at decision time breaks the
# 57.14% milestone-4 ceiling that Stage H confirmed across model lineages.
#
# Direct comparison to Stage H:
#   Stage H model: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4         (no thinking mode)
#                  → 57.14% × n=3, σ=0
#   Stage K model: cyankiwi/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit
#                  (always-thinking variant; vLLM strips <think> via
#                   --reasoning-parser qwen3, agent harness unchanged)
#
# Same MACLA agent stack, same tool-call format, same prompts. The single
# variable is: does the LLM use an extended-reasoning budget before each
# action emission? Same active param count (3B), similar total params (30B
# vs 35B), same quant (AWQ-Int4).
#
# Decision criteria:
#   mean >= 71.43%  → thinking budget lifts the ceiling; pursue thinking-mode
#                     scaffold additions (e.g., port to Gemma via fine-tune)
#   mean ~= 57.14%  → thinking doesn't lift; either it's hitting same plateau
#                     for the same reason OR the gain is from the wrong
#                     reasoning style. Stage J cumulative memory becomes the
#                     next move.
#   mean <  42%     → thinking-mode interferes with tool-call emission or the
#                     30B/3B-active is meaningfully weaker than 35B/3B-active.
#                     Check truncation rate.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

AGENT_CFG="configs/pokemon_red/agent/qwen3_thinking.yaml"
ENV_CFG="configs/pokemon_red/env/default.yaml"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

grep -q "Qwen3-30B-A3B-Thinking" "$AGENT_CFG" || {
    echo "FATAL: $AGENT_CFG not thinking variant"; exit 1; }

# Pre-flight: is vLLM serving the thinking model?
if ! curl -s --max-time 3 http://localhost:8000/v1/models | grep -qi 'thinking'; then
    echo "FATAL: vLLM at :8000 is not serving a Qwen3-Thinking model"
    echo "  Stop current:  pkill -f 'vllm.entrypoints.openai.api_server'"
    echo "  Start qwen3-thinking:"
    echo "      nohup ./serving/qwen_serve.sh cyankiwi/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit \\"
    echo "          >/tmp/qwen_thinking_serve.log 2>&1 & disown"
    echo "  Wait until:    curl -s http://localhost:8000/v1/models | grep -qi thinking"
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
    run_id="pr_stage_k_qwen3_thinking_pokemon_iter${iter}_$(date -u +%Y%m%dT%H%M%SZ)"
    started=$(date +%s)
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] STAGE K n=3 iter $iter/$n"
    echo "  config:  qwen3_thinking + Stage D"
    echo "  target:  >= 71.43% (5/7) to beat plateau"
    echo "  vs:      Stage H 57.14% × n=3 (no thinking)"
    echo "  run_id:  $run_id"
    echo "================================================================"

    if ! uv run python run.py \
        -c qwen3_thinking \
        --local --games pokemon_red \
        --run-id "$run_id" \
        -d "Stage K iter $iter: Qwen3-Thinking 30B-A3B-Int4 — extended-reasoning budget at decision time"; then
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
echo "[$(date -u +%H:%M:%SZ)] STAGE K n=3 SUMMARY"
echo "================================================================"
python3 <<PYEOF
import json, statistics, datetime as dt
from pathlib import Path

scores = [$(IFS=,; echo "${scores[*]}")]
fmt = ", ".join(f"{s:.2f}%" for s in scores)
mean = statistics.mean(scores) if scores else 0
std = statistics.stdev(scores) if len(scores) > 1 else 0.0
print(f"  Scores:                                {fmt}")
print(f"  Mean +/- std:                          {mean:.2f}% +/- {std:.2f}pp")
print(f"  Min/Max:                               {min(scores):.2f}% / {max(scores):.2f}%")
print(f"  Stage H (Qwen 3.5 no-thinking) n=3:    57.14% +/- 0pp")
print(f"  Delta vs Stage H:                      {mean - 57.14:+.2f} pp")
print(f"  Target (>= 71.43%):                    {'HIT' if mean >= 71.43 else 'MISS'}")

out = Path("experiments/stage_k_qwen_thinking/qwen3_thinking/results.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
existing = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
row = {
    "experiment": len(existing) + 1,
    "variant": "stage_k_qwen3_thinking_pokemon_n3",
    "game": "pokemon_red",
    "model": "cyankiwi/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit",
    "evaluation_score": mean,
    "evaluation_score_std": std,
    "evaluation_score_min": min(scores),
    "evaluation_score_max": max(scores),
    "n_episodes": len(scores),
    "scores": scores,
    "steps": 300,
    "status": "KEEP",
    "description": "Stage K: Qwen3-Thinking 30B-A3B-Int4 extended-reasoning budget at decision time",
    "notes": f"n=3: mean={mean:.2f}% std={std:.2f}pp scores={fmt}. delta_vs_stage_h={mean-57.14:+.2f}pp",
    "tags": ["qwen", "qwen3-thinking-30b-a3b", "stage_k_thinking_mode", "stage_k_qwen3_thinking_pokemon_n3"],
    "wandb_url": "",
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    "config_name": "qwen3_thinking",
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
