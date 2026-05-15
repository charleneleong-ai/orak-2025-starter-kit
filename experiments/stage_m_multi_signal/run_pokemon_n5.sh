#!/usr/bin/env bash
# Stage M — multi-signal procedure quality + exploration novelty (n=5).
#
# Builds on Stage L (PR #85). Stage L confirmed map-aware procedure keys +
# iter-TTL deliver monotonic M4 banking speedup (259 → 229 → 172 → 140
# across the four passing iters) but no iter crossed the M5 (Viridian)
# gate. The remaining ceiling is past M4 in cognitive exploration +
# action-quality signal, not procedure-cache context.
#
# This sweep tests two generalisable interventions added in
# agents/macla/macla_lib.py:
#   (a) state_delta_confidence multiplicative on EU — downweights
#       procedures whose past executions produced no salient game-state
#       change
#   (b) novelty theta bump — raises effective theta_conf on unvisited
#       maps so cached procs rarely fire and the LLM explores
#
# Baseline = Stage L (this same agent, prior sweep): [57.14, 57.14, 57.14,
# 28.57, 57.14], M4 steps 259→229→172→140.
#
# Minimum bar: late_mean >= early_mean (no negative transfer) AND match
# Stage L's M4 banking speed by iter 5 (≤ 140 steps).
# Lift bar: any iter reaches Viridian (M5) OR scores above 57.14%.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG_NAME="gemma_26b"
AGENT_CFG="configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml"
N=5
TAG="stage_m_multi_signal"
RESULTS_DIR="experiments/stage_m_multi_signal"
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/tmp/orak-stage-m-multi-signal"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Pre-flights — same shape as Stage L launcher
ASM_DIR="evaluation_utils/mcp_game_servers/pokemon_red/game/pokered/data/maps/objects"
asm_count=$(find "$ASM_DIR" -maxdepth 1 -name "*.asm" 2>/dev/null | wc -l)
[[ "$asm_count" -ge 100 ]] || { echo "FATAL: only $asm_count .asm files"; exit 1; }
echo "[preflight] $asm_count .asm files in pokered/data/maps/objects/"

[[ -s "executables/pokemon_red/pyboy/pokered.gbc" ]] || { echo "FATAL: ROM missing"; exit 1; }

# Pre-flight: vLLM serving Gemma 4-26B-A4B-AWQ-4bit
served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
    | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
declared=$(grep '^model:' "$AGENT_CFG" | head -1 | sed 's/model: *"//;s/" *$//')
if [[ "$served" != "$declared" ]]; then
    echo "FATAL: vLLM mismatch. declared=$declared served=$served"
    exit 1
fi
echo "[preflight] vLLM serving $served"

# Pre-flight: confirm we are running with the Stage M code (multi-signal
# patch present in agents/macla/macla_lib.py).
if ! grep -q "Stage M (a): multiplicative state-delta confidence" agents/macla/macla_lib.py; then
    echo "FATAL: agents/macla/macla_lib.py is missing the Stage M multi-signal patch"
    echo "  Are you on the feat/macla-multi-signal-quality branch?"
    exit 1
fi
if ! grep -q "Stage M (b): record the map and bump theta on first visit" agents/macla/macla_lib.py; then
    echo "FATAL: agents/macla/macla_lib.py is missing the Stage M novelty-bump patch"
    exit 1
fi
if ! grep -q "Stage M (third signal): logprob_confidence" agents/macla/macla_lib.py; then
    echo "FATAL: agents/macla/macla_lib.py is missing the Stage M logprob-confidence patch"
    exit 1
fi
if ! grep -q "logprobs=True" agents/macla/base.py; then
    echo "FATAL: agents/macla/base.py is missing logprobs=True on the ChatOpenAI client"
    exit 1
fi
echo "[preflight] Stage M three-signal (state_delta + novelty + logprob) code present"

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
        -d "Stage M (multi-signal + novelty): iter $iter (inherit from ${prev_run_id:-NONE})"
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
    "description": f"Stage M: multi-signal procedure quality + exploration novelty; {len(scores)}x pokemon cumulative memory",
    "notes": f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores=[{fmt}] learning_delta={delta:+.2f}pp",
    "tags": ["stage_m_multi_signal", "cumulative_memory", "pokemon_red"],
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
