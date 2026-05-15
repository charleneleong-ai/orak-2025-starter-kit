#!/usr/bin/env bash
# Post-Stage-H decision daemon: waits for Stage H rerun to complete, reads
# D + H results, and conditionally auto-queues Stage K cumulative-memory
# rerun (n=5, Gemma-26B) under the asm fix.
#
# Decision rule:
#   - If max(D_mean, H_mean) >= 50%   -> launch Stage K rerun. Covers LIFT
#     and FLAT outcomes; Stage K is the highest-leverage follow-up since
#     PR #75's REGRESS verdict was at 48.57% with placeholder reasoning.
#   - If max(D_mean, H_mean) <  50%   -> write REGRESS_HALT.txt marker
#     and exit. Real sprite names probably destabilised the prompt;
#     burning another 4h on K isn't worth it without investigation first.
#
# Designed as a PPID=1 daemon (setsid+nohup+disown). Survives SSH/CC death.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

D_RESULTS="experiments/post_asm_rerun/stage_d_post_asm/results.jsonl"
H_RESULTS="experiments/post_asm_rerun/stage_h_post_asm/results.jsonl"
HALT_MARKER="experiments/post_asm_rerun/REGRESS_HALT.txt"
GEMMA_MODEL="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
HARD_TIMEOUT_SEC=$((9 * 3600))   # 9h: D ~2.5h + H ~2.5h + K ~4h
POLL_SEC=90

start_ts=$(date +%s)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] post_h_decision daemon up"

# Phase 1: wait for Stage H results.jsonl with n_episodes>=3
while true; do
    elapsed=$(( $(date +%s) - start_ts ))
    [[ $elapsed -gt $HARD_TIMEOUT_SEC ]] && { echo "HARD TIMEOUT — aborting"; exit 2; }

    if [[ -f "$H_RESULTS" ]]; then
        n=$(tail -1 "$H_RESULTS" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('n_episodes',0))" 2>/dev/null || echo 0)
        [[ "$n" -ge 3 ]] && { echo "[$(date -u +%H:%M:%SZ)] Stage H complete (n=$n)"; break; }
    fi
    # If both bash launchers are gone AND no H results, abort
    if ! pgrep -f "run_pokemon_rerun.sh" >/dev/null && ! pgrep -f "watch_d_then_run_h" >/dev/null && [[ ! -f "$H_RESULTS" ]]; then
        echo "[$(date -u +%H:%M:%SZ)] all upstream launchers gone with no H results — aborting"
        exit 3
    fi
    sleep "$POLL_SEC"
done

# Phase 2: parse D + H means, classify max
read_mean() {
    local f="$1"
    [[ -f "$f" ]] || { echo "-1"; return; }
    tail -1 "$f" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('evaluation_score', -1))" 2>/dev/null
}
d_mean=$(read_mean "$D_RESULTS")
h_mean=$(read_mean "$H_RESULTS")
max_mean=$(python3 -c "print(max($d_mean, $h_mean))")
echo "[$(date -u +%H:%M:%SZ)] D_mean=${d_mean}%  H_mean=${h_mean}%  max=${max_mean}%"

# Decision
proceed=$(python3 -c "print(1 if $max_mean >= 50.0 else 0)")
if [[ "$proceed" -eq 0 ]]; then
    cat > "$HALT_MARKER" <<EOF
HALT: post-asm rerun shows max(D=$d_mean%, H=$h_mean%) below 50% threshold.
Hypothesis: real sprite names destabilised the prompt. Need manual
investigation before burning 4h on a Stage K cumulative-memory rerun.
Decision daemon exited at $(date -u +%Y-%m-%dT%H:%M:%SZ).
EOF
    echo "[$(date -u +%H:%M:%SZ)] REGRESS — wrote $HALT_MARKER and halting"
    exit 0
fi

echo "[$(date -u +%H:%M:%SZ)] PROCEED — launching Stage K cumulative rerun (n=5)"

# Phase 3: swap vLLM Qwen -> Gemma
pkill -f "vllm.entrypoints.openai.api_server" || true
for i in {1..30}; do
    ss -ltn | grep -q ':8000 ' || break
    sleep 2
done

GEMMA_LOG="logs/gemma_serve_$(date -u +%Y%m%dT%H%M%SZ).log"
setsid nohup bash serving/gemma_serve.sh "$GEMMA_MODEL" </dev/null >>"$GEMMA_LOG" 2>&1 & disown
echo "[$(date -u +%H:%M:%SZ)] gemma vLLM relaunched, log=$GEMMA_LOG"

# Phase 4: wait for gemma to serve
for i in {1..120}; do
    served=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
    [[ "$served" == "$GEMMA_MODEL" ]] && { echo "[$(date -u +%H:%M:%SZ)] gemma ready"; break; }
    [[ $i -eq 120 ]] && { echo "gemma never came up"; exit 4; }
    sleep 10
done

# Phase 5: launch Stage K cumulative rerun
K_LOG="logs/post_asm_rerun_stage_k_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[$(date -u +%H:%M:%SZ)] launching Stage K, log=$K_LOG"
bash experiments/post_asm_rerun/run_pokemon_cumulative_rerun.sh >>"$K_LOG" 2>&1
ret=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Stage K launcher exited with $ret"
exit $ret
