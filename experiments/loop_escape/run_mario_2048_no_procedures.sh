#!/usr/bin/env bash
# Stage B' cross-game extension — mario + 2048, n=1 each, no procedures.
#
# Pokemon Stage B' (run_pokemon_no_procedures_n3.sh) covers pokemon at n=3.
# This script extends the disambiguation across games. Cheap signal:
# mario ~10 min, 2048 ~30 min, total ~40 min.
#
# Expected reads:
#   mario lifts to ~61% (PR #31 Stage B baseline)  → procedures hurt mario;
#                                                    PR #68 is right play AT LEAST for mario
#   mario stays ~35% (Stage D baseline)            → LLM ceiling on mario too
#   2048 stays ~64% (Stage D baseline)             → procedure layer was neutral
#   2048 lifts                                     → procedures were dampening 2048 too
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

export GAME_DATA_DIR="/tmp/orak-planner-prompt"

run_game() {
    local game="$1" env_cfg_path="$2"
    local agent_cfg_path="configs/$game/agent/gemma_26b_no_procedures.yaml"

    grep -q "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" "$agent_cfg_path" || {
        echo "FATAL: $agent_cfg_path not AWQ — vLLM will 404 on every LLM call"
        return 1
    }
    grep -q "use_procedure_layer: false" "$agent_cfg_path" || {
        echo "FATAL: $agent_cfg_path missing use_procedure_layer: false"
        return 1
    }

    local run_id="pr_procesc_stage_b_prime_${game}_$(date -u +%Y%m%dT%H%M%SZ)"
    local game_logs="$GAME_DATA_DIR/game_logs/$game/$run_id"
    local started; started=$(date +%s)

    sed -i 's/^max_steps: .*/max_steps: 300/' "$env_cfg_path"
    mkdir -p "$GAME_DATA_DIR/game_logs/$game"

    echo "================================================================"
    echo "[$(date -u +%H:%M:%SZ)] STAGE B' ($game, no procedures)"
    echo "  run_id=$run_id"
    echo "================================================================"

    if ! uv run python run.py \
        -c gemma_26b_no_procedures \
        --local --games "$game" \
        --run-id "$run_id" \
        -d "Stage B' baseline: $game Stage D - procedure cache, reflection ON"; then
        echo "[FAIL] $game run.py exited non-zero — continuing"
        return 1
    fi

    local actual_dir="$GAME_DATA_DIR/$game/$run_id"
    [ -d "$actual_dir" ] && ln -sfn "$actual_dir" "$game_logs"
    local elapsed=$(( ( $(date +%s) - started ) / 60 ))
    echo "[$game] runtime=${elapsed}min"

    # Append a row directly (no shared append.py on this branch).
    local summary="$actual_dir/evaluation_summary.json"
    python3 <<PYEOF
import json, datetime as dt
from pathlib import Path
d = json.load(open("$summary"))
eps = d.get("episodes", [])
raw = max((float(e.get("final_score", 0.0)) for e in eps), default=0.0)
eval_score = (raw / 7.0) * 100.0 if "$game" == "pokemon_red" else raw
out = Path("experiments/loop_escape/gemma_26b/results.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
existing = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
row = {
    "experiment": len(existing) + 1,
    "variant": f"stage_b_prime_no_procedures_$game",
    "game": "$game",
    "evaluation_score": eval_score,
    "game_score": raw,
    "steps": int(d.get("total_inference_calls", 300)),
    "runtime_min": $elapsed,
    "status": "KEEP",
    "description": f"Stage B' cross-game: $game Stage D - procedure cache",
    "notes": f"max_eval={eval_score:.2f}, {len(eps)} ep, 300 steps",
    "tags": ["loop_escape", "stage_b_prime_no_procedures", "stage_b_prime_no_procedures_$game"],
    "wandb_url": "",
    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
    "config_name": "gemma_26b_no_procedures",
}
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(f"appended: $game eval={eval_score:.2f}")
PYEOF
}

echo "[$(date -u +%H:%M:%SZ)] === Stage B' cross-game (mario + 2048) ==="
run_game super_mario configs/super_mario/env/default.yaml
run_game twenty_fourty_eight configs/twenty_fourty_eight/env/default.yaml

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] STAGE B' CROSS-GAME DONE"
echo "================================================================"
