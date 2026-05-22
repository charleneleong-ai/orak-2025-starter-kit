#!/usr/bin/env bash
# Watcher daemon — polls for the GSPO re-roll launcher to finish, then
# launches the Stage S v2 sweep with the same PPID=1 daemon pattern.
#
# Intended to be launched detached so it survives SSH/CC death:
#
#   cd /workspace/orak-stage-s
#   setsid nohup bash experiments/stage_s_cache_veto/wait_for_reroll_then_v2.sh \
#       </dev/null >>logs/wait_for_reroll_then_v2_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 & disown
#
# Why a watcher rather than a cron entry: the v2 launcher needs the
# local repo + ROM + vLLM server + GAME_DATA_DIR — none of which a
# remote scheduled agent in the cloud could touch.
#
# Behavior:
#   - Polls every 60s for any reroll.sh process.
#   - Once the reroll completes (no matching pid), launches v2 as a
#     detached daemon and exits the watcher.
#   - Hard timeout at 12h so a hung reroll doesn't keep this alive
#     forever.

set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 1

V2_LAUNCHER="experiments/stage_s_cache_veto/run_pokemon_n5_v2_post_viridian_chain.sh"
POLL_S=60
TIMEOUT_S=$((12 * 3600))
started=$(date +%s)

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher started — polling every ${POLL_S}s for reroll exit"

while true; do
    # Find any reroll.sh process (parent or child). We match on the
    # script basename so this survives if the launcher gets re-pathed.
    reroll_pids=$(pgrep -f "experiments/gspo/reroll.sh" || true)
    if [[ -z "$reroll_pids" ]]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] reroll launcher gone — kicking off v2"
        break
    fi

    elapsed=$(( $(date +%s) - started ))
    if (( elapsed > TIMEOUT_S )); then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] FATAL: reroll still running after ${TIMEOUT_S}s — aborting watcher (v2 NOT launched)"
        exit 1
    fi

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] reroll still running (pids: $(echo "$reroll_pids" | tr '\n' ' ')); sleeping ${POLL_S}s"
    sleep "$POLL_S"
done

# Also gate on vLLM being responsive — without it the v2 sweep's
# preflight fails immediately. If vLLM is down we leave a note and exit
# so the user notices, rather than letting v2 self-FATAL on iter 1.
vllm_ok=$(curl -s --max-time 5 http://localhost:8000/v1/models 2>/dev/null | grep -c '"id"' || true)
if (( vllm_ok == 0 )); then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] FATAL: vLLM not responding on :8000 — v2 NOT launched. Restart vLLM and re-run this watcher manually."
    exit 2
fi

# Same daemon pattern as the rest of the project: setsid + nohup +
# closed stdin + appended timestamped log + disown so the v2 sweep
# itself runs PPID=1.
ts=$(date -u +%Y%m%dT%H%M%SZ)
log_file="logs/stage_s_v2_${ts}.log"
mkdir -p logs

setsid nohup bash "$V2_LAUNCHER" </dev/null >>"$log_file" 2>&1 &
v2_pid=$!
disown

# Give it ~3s to actually fork into the orchestrator before we report.
sleep 3
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] v2 sweep launched (pid=$v2_pid, log=$log_file)"
ps -ef | grep -F "$V2_LAUNCHER" | grep -v grep | head -3
