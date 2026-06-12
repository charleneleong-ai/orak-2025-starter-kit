#!/usr/bin/env bash
# Qwen 3.6 35B-A3B-Reasoning n=3 cross-game ceiling check.
#
# Orchestrates n=3 seeds × 4 games (pokemon_red, super_mario, twenty_fourty_eight,
# star_craft) against the Qwen3.6-35B-A3B-Reasoning vLLM served at localhost:8000.
#
# Successor to the lost (untracked) experiments/qwen35_cross_game_n3/run_sweep.sh
# — recreated here as committed code. The two weave fixes from this branch's
# parent (feat/episode-credit-assignment) are in scope:
#
#   c5222e9  run.py monkey-patches WeaveTracer.on_chat_model_start
#            + setdefault WEAVE_TRACE_LANGCHAIN=false (langchain auto-tracer
#            thread-leak hang fix from seed1 v1 2026-05-28).
#
#   b590844  evaluation_utils/runner.py disabled-stub guard short-circuits the
#            per-step weave with-block — finish() no longer spawns a rich-
#            progress refresh thread per step (the residual leak after c5222e9).
#            This is also the hypothesized fix for super_mario's silent stop at
#            step ~181 in seed1 v2 2026-05-29 — to be verified by this sweep.
#
# Env-var prefix: WEAVE_DISABLED=1 keeps weave's tracker off (we're not on
# Option B yet — that's task #33). WEAVE_TRACE_LANGCHAIN=false is the real
# kill-switch for the langchain auto-tracer (WEAVE_ENABLED=false in the old
# script only gated weave.finish() at process exit via config/base.py:34 —
# it did NOT touch the langchain BaseTracer hook that caused the leak).
#
# Launch detached so it survives SSH/CC death:
#   cd /workspace/orak-qwen36-n3 && setsid nohup ./experiments/qwen36_cross_game_n3/run_sweep.sh \
#     </dev/null >>logs/qwen36_n3_sweep_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
#
set -euo pipefail

SWEEP_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

VENV_PY="${VENV_PY:-/workspace/orak-qwen36-n3/.venv/bin/python}"
CONFIG="${CONFIG:-qwen36_a3b_reasoning}"
N_SEEDS="${N_SEEDS:-3}"
START_SEED="${START_SEED:-1}"
GAMES_FLAGS=(--games pokemon_red --games super_mario --games twenty_fourty_eight --games star_craft)

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv python not found at $VENV_PY"
    exit 1
fi

echo "[$(date -Iseconds)] vLLM serving: palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4 (reasoning)"
echo "[$(date -Iseconds)] running seeds $START_SEED..$N_SEEDS  sweep_ts=$SWEEP_TS"
echo "[$(date -Iseconds)] config=$CONFIG  games=${GAMES_FLAGS[*]}"
echo

for SEED in $(seq "$START_SEED" "$N_SEEDS"); do
    RUN_ID="qwen36_n3_seed${SEED}_${SWEEP_TS}"
    LOG="${LOG_DIR}/${RUN_ID}.log"
    echo "============================================"
    echo "[$(date -Iseconds)] Seed $SEED / $N_SEEDS   run_id=$RUN_ID"
    echo "============================================"

    if env WEAVE_DISABLED=1 WEAVE_TRACE_LANGCHAIN=false WANDB_MODE=offline \
        "$VENV_PY" -u run.py -c "$CONFIG" --local \
        "${GAMES_FLAGS[@]}" \
        --run-id "$RUN_ID" \
        --experiment-description "Qwen3.6-35B-A3B-Reasoning n=3 ceiling check seed=$SEED (sweep $SWEEP_TS)" \
        >> "$LOG" 2>&1; then
        echo "[$(date -Iseconds)] Seed $SEED OK"
    else
        echo "[$(date -Iseconds)] Seed $SEED FAILED — see $LOG"
    fi
    echo
done

echo "[$(date -Iseconds)] sweep complete"
