#!/usr/bin/env bash
# Post-Stage-K daemon: waits for the running Stage K cumulative-memory rerun
# (on master) to finish, then launches Stage L (map-aware procedures, n=5)
# under the modified MACLA on this branch.
#
# Stage L cohabits the GPU with Gemma vLLM — same model serving Stage K, so
# no vLLM swap is needed. The only switch is the agents/macla/* code path,
# which is already in place on this branch.
#
# Designed as a PPID=1 daemon (setsid+nohup+disown). Survives SSH/CC death.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

K_RESULTS="experiments/post_asm_rerun/stage_k_post_asm/results.jsonl"
L_RESULTS="experiments/stage_l_map_aware/results.jsonl"
HARD_TIMEOUT_SEC=$((10 * 3600))   # 10h: K leftover (~1h) + L (~4h) + slack
POLL_SEC=90

start_ts=$(date +%s)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wait_for_k_then_run_l daemon up"
echo "  K results path: $K_RESULTS"
echo "  L results path: $L_RESULTS"

# Phase 1: wait for Stage K results.jsonl with n_episodes>=5
while true; do
    elapsed=$(( $(date +%s) - start_ts ))
    [[ $elapsed -gt $HARD_TIMEOUT_SEC ]] && { echo "HARD TIMEOUT — aborting"; exit 2; }

    if [[ -f "$K_RESULTS" ]]; then
        n=$(tail -1 "$K_RESULTS" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('n_episodes',0))" 2>/dev/null || echo 0)
        [[ "$n" -ge 5 ]] && { echo "[$(date -u +%H:%M:%SZ)] Stage K complete (n=$n)"; break; }
    fi
    # If K launchers are all gone AND no results, abort
    if ! pgrep -f "run_pokemon_cumulative_rerun.sh" >/dev/null \
       && ! pgrep -f "post_h_decision.sh" >/dev/null \
       && [[ ! -f "$K_RESULTS" ]]; then
        echo "[$(date -u +%H:%M:%SZ)] K launchers gone with no results — aborting L"
        exit 3
    fi
    sleep "$POLL_SEC"
done

# Phase 2: confirm gemma vLLM still up (Stage K finished gracefully should
# leave it running for Stage L to reuse).
served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
if [[ -z "$served" ]] || [[ "$served" != *gemma* ]]; then
    echo "[$(date -u +%H:%M:%SZ)] gemma vLLM not serving (got '$served') — restarting"
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5
    GEMMA_LOG="logs/gemma_serve_$(date -u +%Y%m%dT%H%M%SZ).log"
    setsid nohup bash serving/gemma_serve.sh "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit" </dev/null >>"$GEMMA_LOG" 2>&1 & disown
    for i in {1..120}; do
        served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
        [[ "$served" == *gemma* ]] && { echo "[$(date -u +%H:%M:%SZ)] gemma ready"; break; }
        [[ $i -eq 120 ]] && { echo "gemma never came up"; exit 4; }
        sleep 10
    done
fi
echo "[$(date -u +%H:%M:%SZ)] vLLM ready: $served"

# Phase 3: launch Stage L n=5 sweep
L_LOG="logs/stage_l_map_aware_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[$(date -u +%H:%M:%SZ)] launching Stage L, log=$L_LOG"
bash experiments/stage_l_map_aware/run_pokemon_n5.sh >>"$L_LOG" 2>&1
ret=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage L launcher exited with $ret"
exit $ret
