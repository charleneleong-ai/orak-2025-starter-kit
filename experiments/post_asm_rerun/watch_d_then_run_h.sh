#!/usr/bin/env bash
# Orchestrator daemon: waits for Stage D n=3 rerun to finish, swaps vLLM
# Gemma→Qwen, then launches Stage H n=3 rerun. Designed to run as a
# fully-detached PPID=1 daemon (setsid + nohup + disown) so it survives
# SSH / CC death across the ~5h total.
#
# Exit conditions:
#   - Stage H n=3 results.jsonl present  -> success exit 0
#   - 7h hard timeout (D should be ~2.5h + swap + H ~2.5h + 1h slack)
#   - Stage D launcher process gone AND no D results.jsonl  -> assume D
#     crashed; abort H launch
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

D_RESULTS="experiments/post_asm_rerun/stage_d_post_asm/results.jsonl"
H_RESULTS="experiments/post_asm_rerun/stage_h_post_asm/results.jsonl"
HARD_TIMEOUT_SEC=$((7 * 3600))
POLL_SEC=60

GEMMA_MODEL="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
QWEN_MODEL="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"

start_ts=$(date +%s)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] orchestrator: waiting for Stage D to complete"
echo "  D results path: $D_RESULTS"
echo "  H results path: $H_RESULTS"

# Phase 1: wait for D to finish (n_episodes=3 in results.jsonl)
while true; do
    elapsed=$(( $(date +%s) - start_ts ))
    [[ $elapsed -gt $HARD_TIMEOUT_SEC ]] && { echo "[$(date -u +%H:%M:%SZ)] HARD TIMEOUT — aborting"; exit 2; }

    if [[ -f "$D_RESULTS" ]]; then
        n=$(tail -1 "$D_RESULTS" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('n_episodes',0))" 2>/dev/null || echo 0)
        if [[ "$n" -ge 3 ]]; then
            echo "[$(date -u +%H:%M:%SZ)] Stage D complete (n_episodes=$n)"
            break
        fi
    fi
    # If the bash launcher is gone AND no results, abort
    if ! pgrep -f "run_pokemon_rerun.sh gemma_26b" >/dev/null && [[ ! -f "$D_RESULTS" ]]; then
        echo "[$(date -u +%H:%M:%SZ)] Stage D launcher gone with no results — aborting H"
        exit 3
    fi
    sleep "$POLL_SEC"
done

# Phase 2: swap vLLM Gemma -> Qwen
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] swapping vLLM Gemma -> Qwen35"
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 10

# wait for port :8000 to be free
for i in {1..30}; do
    if ! ss -ltn | grep -q ':8000 '; then break; fi
    sleep 2
done

# Launch qwen vLLM as PPID=1 daemon
QWEN_LOG="logs/qwen_serve_$(date -u +%Y%m%dT%H%M%SZ).log"
setsid nohup bash serving/qwen_serve.sh "$QWEN_MODEL" </dev/null >>"$QWEN_LOG" 2>&1 & disown
echo "[$(date -u +%H:%M:%SZ)] qwen vLLM launched, log=$QWEN_LOG"

# Phase 3: wait for qwen to serve
echo "[$(date -u +%H:%M:%SZ)] waiting for qwen35 vLLM to be ready"
for i in {1..120}; do
    served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
    if [[ "$served" == "$QWEN_MODEL" ]]; then
        echo "[$(date -u +%H:%M:%SZ)] qwen ready: $served"
        break
    fi
    [[ $i -eq 120 ]] && { echo "[$(date -u +%H:%M:%SZ)] qwen never came up — aborting"; exit 4; }
    sleep 10
done

# Phase 4: launch Stage H rerun
H_LOG="logs/post_asm_rerun_stage_h_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[$(date -u +%H:%M:%SZ)] launching Stage H rerun, log=$H_LOG"
bash experiments/post_asm_rerun/run_pokemon_rerun.sh qwen35_a3b_int4 3 stage_h_post_asm >>"$H_LOG" 2>&1
ret=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage H launcher exited with $ret"
exit $ret
