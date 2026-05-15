#!/usr/bin/env bash
# Post-asm-fix rerun: n=3 of pokemon_red Stage D / Stage H baselines,
# now that pokered/ data/maps/objects/*.asm is on disk and the harness
# emits real SPRITE_OAK / SPRITE_POKE_BALL tokens instead of OBJ_n_n
# placeholders. See docs/experiments/pokemon-asm-gap.md (commit 264c693).
#
# Pre-fix baselines (placeholder reasoning surface):
#   Stage D Gemma-26B (PR #31, n=1):   57.14%
#   Stage D Gemma-26B Stage B' n=3:    42.86% ± 14.29pp
#   Stage G Gemma-26B procedure-escape: 47.62% ± 16.49pp
#   Stage H Qwen3.5-35B-A3B-Int4 n=3:  47.62% ± 16.49pp
#   Stage K Gemma-26B cumulative n=5:  48.57% ± 12.78pp (REGRESS)
#
# Hypothesis: the 57.14% ceiling across all stages was a *harness artifact*,
# not a model ceiling. With real sprite names visible, the agent can
# distinguish SPRITE_OAK from SPRITE_BOOKSHELF / SPRITE_NPC etc.,
# unlocking the M3/M4 OaksLab dialogue chain past 57.14%.
#
# Usage:
#   bash run_pokemon_rerun.sh <agent_config> <n_iters> <tag>
# Example:
#   bash run_pokemon_rerun.sh gemma_26b 3 stage_d_post_asm
#   bash run_pokemon_rerun.sh qwen35_a3b_int4 3 stage_h_post_asm

set -uo pipefail

AGENT_CFG_NAME="${1:?must pass agent config name (e.g. gemma_26b)}"
N="${2:?must pass iter count (e.g. 3)}"
TAG="${3:?must pass tag (e.g. stage_d_post_asm)}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

ENV_CFG="configs/pokemon_red/env/default.yaml"
AGENT_CFG="configs/pokemon_red/agent/${AGENT_CFG_NAME}.yaml"
RESULTS_DIR="experiments/post_asm_rerun/${TAG}"
mkdir -p "$RESULTS_DIR"

export GAME_DATA_DIR="/tmp/orak-post-asm-rerun"
mkdir -p "$GAME_DATA_DIR/game_logs/pokemon_red"

# Pre-flight 1: agent config exists
[[ -f "$AGENT_CFG" ]] || { echo "FATAL: agent config $AGENT_CFG missing"; exit 1; }

# Pre-flight 2: pokered/ asm files populated (the whole point of this rerun)
ASM_DIR="evaluation_utils/mcp_game_servers/pokemon_red/game/pokered/data/maps/objects"
asm_count=$(find "$ASM_DIR" -maxdepth 1 -name "*.asm" 2>/dev/null | wc -l)
if [[ "$asm_count" -lt 100 ]]; then
    echo "FATAL: only $asm_count .asm files in $ASM_DIR — expected ~248."
    echo "  This is the whole reason for the rerun. Populate with:"
    echo "    git clone --depth 1 https://github.com/pret/pokered.git evaluation_utils/mcp_game_servers/pokemon_red/game/pokered"
    exit 1
fi
echo "[preflight] $asm_count .asm files in pokered/data/maps/objects/"

# Pre-flight 3: ROM
[[ -s "executables/pokemon_red/pyboy/pokered.gbc" ]] || {
    echo "FATAL: pyboy ROM missing"; exit 1; }

# Pre-flight 4: vLLM serving the model declared in the config
served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
    | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
declared=$(grep '^model:' "$AGENT_CFG" | head -1 | sed 's/model: *"//;s/" *$//')
if [[ "$served" != "$declared" ]]; then
    echo "FATAL: vLLM mismatch."
    echo "  agent config declares: $declared"
    echo "  vLLM is serving:       ${served:-NONE}"
    echo "  Restart vLLM with: ./serving/gemma_serve.sh $declared (or the qwen equivalent)"
    exit 1
fi
echo "[preflight] vLLM serving $served"

# Restore max_steps=300 on EXIT (it gets mutated in some experiments)
restore() { sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"; }
trap restore EXIT
sed -i 's/^max_steps: .*/max_steps: 300/' "$ENV_CFG"

scores=()
for iter in $(seq 1 "$N"); do
    run_id="post_asm_${TAG}_iter${iter}_$(date -u +%Y%m%dT%H%M%SZ)"
    started=$(date +%s)
    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] $TAG iter $iter/$N"
    echo "  config:  $AGENT_CFG_NAME"
    echo "  run_id:  $run_id"
    echo "================================================================"

    if ! uv run python run.py \
            -c "$AGENT_CFG_NAME" \
            --local --games pokemon_red \
            --run-id "$run_id" \
            -d "Post-asm-fix rerun: $TAG iter $iter (real sprite names visible)"; then
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
    echo "[iter $iter] eval=${score}%, runtime=${elapsed}min"
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
fmt = ", ".join(f"{s:.2f}%" for s in scores)
print(f"  Per-iter scores: {fmt}")
print(f"  Mean +/- std:    {mean:.2f}% +/- {std:.2f}pp")
print(f"  Min/Max:         {min(scores):.2f}% / {max(scores):.2f}%")

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
    "n_episodes": len(scores),
    "scores": scores,
    "steps": 300,
    "status": "KEEP",
    "description": f"Post-asm-fix rerun: {len(scores)}x pokemon Stage D-stack with real sprite names",
    "notes": f"n={len(scores)}: mean={mean:.2f}% std={std:.2f}pp scores=[{fmt}]",
    "tags": ["post_asm_rerun", "$TAG", "pokemon_red"],
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  Appended to {out}")
PYEOF
echo "================================================================"
