#!/usr/bin/env bash
# GSPO re-roll launcher — run K rollouts that share MACLA agent state,
# tag them with a common group_id so the collator can build a real
# group-relative advantage distribution.
#
# What "share a checkpoint" means here (verified against runner.py:360-411):
#   --load-checkpoint --prev-run-id X restores ONLY the agent's learned
#   state (procedural memory + vector memory via agent.load_state). The
#   env is created fresh — the game starts at ROM start every rollout,
#   not at a mid-game world position. Transient counters (_step_count,
#   _last_score, _last_action) are explicitly reset on the prev_run_id
#   branch.
#
# Variance source: vLLM sampling at temperature>0 (gemma_26b: 0.7) makes
# K stochastic rollouts of the same policy diverge into different
# trajectories with different final scores. That divergence — not shared
# world state — is what gives the group non-zero variance and thus a
# meaningful group-relative advantage.
#
# Why: train.py refuses on degenerate datasets where every group_id is
# a singleton (variance=0 → no gradient signal). The default collation
# uses group_id=run_id so each iter is its own group; this launcher
# breaks that by re-rolling K times from one MACLA state with a shared
# group_id, written to gspo_group.json in each rollout's iter dir.
#
# Usage:
#   ./experiments/gspo/reroll.sh \
#       --checkpoint-run-id stage_s_v1_viridian_bridge_iter1_20260521T062940Z \
#       --k 4 \
#       --group-id g_v1_iter1_reroll
#
# Pre-flights:
#   * vLLM serving the same model as configs/pokemon_red/agent/<cfg>.yaml
#   * Agent config temperature > 0 (otherwise K rollouts are identical)
#   * The checkpoint run_id must exist under $GAME_DATA_DIR/<game>/
#   * macla state-load mechanism must be intact (--load-checkpoint flag)

set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

# Load API keys if present
[[ -f "$REPO/.env" ]] && set -a && source "$REPO/.env" && set +a

# ── args ──────────────────────────────────────────────────────────────
CHECKPOINT_RUN_ID=""
K=4
GROUP_ID=""
AGENT_CFG_NAME="gemma_26b"
GAME="pokemon_red"
MAX_STEPS=600
POLICY_ID="base"

usage() {
    cat <<USAGE
GSPO re-roll launcher — K rollouts from one checkpoint, shared group_id.

  --checkpoint-run-id <id>   prior iter's run_id to load state from (required)
  --k <int>                  number of rollouts (default 4)
  --group-id <name>          shared group_id for all K rollouts
                             (default: <checkpoint_run_id>_reroll)
  --policy-id <name>         tag for the policy vLLM is serving during the
                             rollout. "base" = no LoRA (default; iter 1).
                             For iter 2+ pass the saved adapter name/path
                             (e.g. "lora_v1") so the trainer can load it
                             as pi_old when computing the importance ratio.
  --agent <name>             agent config name (default gemma_26b)
  --game <name>              game name (default pokemon_red)
  --max-steps <int>          per-rollout step budget (default 600)

Produces K iter dirs under \$GAME_DATA_DIR/<game>/<run_id>/, each with
the standard game_states.jsonl + evaluation_summary.json, plus a
gspo_group.json sidecar marking them as one GSPO group and recording
which policy generated them.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-run-id) CHECKPOINT_RUN_ID="$2"; shift 2;;
        --k) K="$2"; shift 2;;
        --group-id) GROUP_ID="$2"; shift 2;;
        --policy-id) POLICY_ID="$2"; shift 2;;
        --agent) AGENT_CFG_NAME="$2"; shift 2;;
        --game) GAME="$2"; shift 2;;
        --max-steps) MAX_STEPS="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "unknown flag: $1"; usage; exit 1;;
    esac
done

[[ -n "$CHECKPOINT_RUN_ID" ]] || { echo "FATAL: --checkpoint-run-id required"; usage; exit 1; }
[[ "$K" =~ ^[0-9]+$ ]] && [[ "$K" -ge 2 ]] || { echo "FATAL: --k must be int >= 2"; exit 1; }
[[ -n "$GROUP_ID" ]] || GROUP_ID="${CHECKPOINT_RUN_ID}_reroll"

AGENT_CFG="configs/${GAME}/agent/${AGENT_CFG_NAME}.yaml"
ENV_CFG="configs/${GAME}/env/default.yaml"
TAG="gspo_reroll_${GROUP_ID}"
RESULTS_DIR="experiments/${TAG}"
mkdir -p "$RESULTS_DIR"

# Step-budget override + restore-on-exit trap. Mirrors the Stage R/S
# launcher convention: sed the env yaml, trap to restore the default,
# so a crashed sweep doesn't leave a permanent edit.
ORIG_MAX_STEPS=$(grep -oE '^max_steps: [0-9]+' "$ENV_CFG" | awk '{print $2}')
restore_env_cfg() { sed -i "s/^max_steps: .*/max_steps: ${ORIG_MAX_STEPS:-300}/" "$ENV_CFG"; }
trap restore_env_cfg EXIT
sed -i "s/^max_steps: .*/max_steps: ${MAX_STEPS}/" "$ENV_CFG"

# GAME_DATA_DIR must be set by caller; default to repo-local game_logs
# matching the runner's evaluation_utils.commons default.
export GAME_DATA_DIR="${GAME_DATA_DIR:-$REPO/game_logs}"

# ── pre-flights ───────────────────────────────────────────────────────
[[ -d "$GAME_DATA_DIR/$GAME/$CHECKPOINT_RUN_ID" ]] \
    || { echo "FATAL: checkpoint dir missing: $GAME_DATA_DIR/$GAME/$CHECKPOINT_RUN_ID"; exit 1; }
echo "[preflight] checkpoint run_id: $CHECKPOINT_RUN_ID"

served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
    | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
declared=$(grep '^model:' "$AGENT_CFG" | head -1 | sed 's/model: *"//;s/" *$//')
[[ "$served" == "$declared" ]] || { echo "FATAL: vLLM mismatch $served vs $declared"; exit 1; }
echo "[preflight] vLLM serving $served"

# Temperature gate: K rollouts only diverge if the policy decodes
# stochastically. If temperature is 0 (or absent → defaults vary by
# provider), every rollout is identical → group variance=0 → no gradient.
temp=$(grep '^temperature:' "$AGENT_CFG" | head -1 | awk '{print $2}')
awk_temp_ok=$(awk -v t="${temp:-0}" 'BEGIN{print (t+0 > 0) ? 1 : 0}')
[[ "$awk_temp_ok" == "1" ]] \
    || { echo "FATAL: temperature=$temp in $AGENT_CFG — K rollouts would be identical"; exit 1; }
echo "[preflight] sampling temperature: $temp"

# ── per-rollout loop ──────────────────────────────────────────────────
echo "================================================================"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] GSPO re-roll: K=$K from $CHECKPOINT_RUN_ID -> group_id=$GROUP_ID"
echo "================================================================"

scores=()
for k in $(seq 1 "$K"); do
    run_id="${GROUP_ID}_k${k}_$(date -u +%Y%m%dT%H%M%SZ)"
    started=$(date +%s)
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] rollout $k/$K: $run_id"

    if ! uv run python run.py \
            -c "$AGENT_CFG_NAME" \
            --local --games "$GAME" \
            --run-id "$run_id" \
            --load-checkpoint --prev-run-id "$CHECKPOINT_RUN_ID" \
            -d "GSPO re-roll k=$k group=$GROUP_ID from $CHECKPOINT_RUN_ID"; then
        echo "[FAIL] rollout $k exited non-zero"
        scores+=("0.0")
        continue
    fi

    # Write the sidecar so collate_iter tags this rollout with the
    # shared group_id + policy_id at collation time.
    actual_dir="$GAME_DATA_DIR/$GAME/$run_id"
    [[ -d "$actual_dir" ]] || { echo "WARN: rollout dir not found: $actual_dir"; continue; }
    printf '{"group_id": "%s", "k": %d, "checkpoint": "%s", "policy_id": "%s"}\n' \
        "$GROUP_ID" "$k" "$CHECKPOINT_RUN_ID" "$POLICY_ID" > "$actual_dir/gspo_group.json"

    elapsed=$(( ($(date +%s) - started) / 60 ))
    summary="$actual_dir/evaluation_summary.json"
    score="0.0"
    if [[ -f "$summary" ]]; then
        score=$(python3 -c "
import json
d = json.load(open('$summary'))
eps = d.get('episodes', [])
raw = max((float(e.get('final_score', 0.0)) for e in eps), default=0.0)
print(f'{(raw/7.0)*100:.2f}')
" 2>/dev/null || echo "0.0")
    fi
    scores+=("$score")
    echo "[rollout $k] eval=${score}%, runtime=${elapsed}min, sidecar=$actual_dir/gspo_group.json"
done

echo
echo "================================================================"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $TAG SUMMARY"
echo "================================================================"
echo "  group_id:    $GROUP_ID"
echo "  K rollouts:  $K"
echo "  scores:      ${scores[*]}"
echo
echo "Next step: collate into a real-variance dataset:"
echo "  uv run python -m experiments.gspo.collate sweep \\"
echo "    \$GAME_DATA_DIR/$GAME --out gspo_reroll.jsonl"
echo "  uv run python -m experiments.gspo.train info gspo_reroll.jsonl"
