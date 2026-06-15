#!/usr/bin/env bash
# Restart vLLM with --enable-lora (plus optional initial adapter).
#
# Used in the GSPO training cycle: rollout → train → restart vLLM with new
# adapter → next rollout uses the adapter. After this returns, subsequent
# adapter swaps don't need another restart — POST to /v1/load_lora_adapter
# via serving/load_lora.sh (enabled by VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# which we set unconditionally for the restarted process).
#
# Usage:
#   ./serving/restart_vllm.sh                                  # restart fresh
#   ./serving/restart_vllm.sh --adapter current=/path/lora_v1  # with adapter preloaded
#   ./serving/restart_vllm.sh --gpu-mem 0.55                   # leave room for training
#
# The model name + max_model_len + dtype are inherited from the currently
# running vLLM (queried via /v1/models and /proc/<pid>/cmdline) so we don't
# accidentally swap models during a training cycle.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ADAPTERS=()
GPU_UTIL=""
WAIT_S=180

usage() {
    cat <<USAGE
Restart vLLM with --enable-lora.

  --adapter <name=path>   register a LoRA adapter at startup (repeatable)
  --gpu-mem <float>       --gpu-memory-utilization (default: same as currently
                          serving, fall back to 0.85)
  --wait <int>            seconds to wait for new vLLM to respond (default 180)

After restart, the new vLLM accepts POST /v1/load_lora_adapter for
subsequent in-place adapter swaps. See serving/load_lora.sh.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --adapter) ADAPTERS+=("$2"); shift 2;;
        --gpu-mem) GPU_UTIL="$2"; shift 2;;
        --wait) WAIT_S="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "unknown flag: $1"; usage; exit 1;;
    esac
done

# Pre-flight: every adapter path must exist AND have both config + weights
# before we shut down the current vLLM — otherwise we kill a running server
# and then can't bring it back up. config alone passes for incomplete
# downloads / partial commits, so check for weights too.
for spec in "${ADAPTERS[@]+"${ADAPTERS[@]}"}"; do
    path="${spec#*=}"
    [[ -d "$path" ]] || { echo "FATAL: adapter dir missing: $path"; exit 1; }
    [[ -f "$path/adapter_config.json" ]] \
        || { echo "FATAL: $path/adapter_config.json missing — not a valid LoRA adapter"; exit 1; }
    # peft saves weights as adapter_model.safetensors (preferred) or adapter_model.bin (legacy)
    [[ -f "$path/adapter_model.safetensors" || -f "$path/adapter_model.bin" ]] \
        || { echo "FATAL: $path missing adapter_model.{safetensors,bin} — incomplete adapter"; exit 1; }
done

# Inherit current model name from /v1/models so we don't accidentally swap models.
served_model=$(curl -sf --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
    | grep -oE '"id":"[^"]+"' | head -1 | sed 's/"id":"//;s/"$//')
[[ -n "$served_model" ]] || { echo "FATAL: vLLM not responding on :8000, can't infer model"; exit 1; }

existing_pid=$(pgrep -f "vllm.entrypoints.openai.api_server" | head -1)
echo "[restart] currently serving: $served_model (pid=${existing_pid:-?})"

# Inherit gpu-memory-utilization from the running process cmdline if not overridden.
if [[ -z "$GPU_UTIL" && -n "$existing_pid" && -r "/proc/$existing_pid/cmdline" ]]; then
    GPU_UTIL=$(tr '\0' ' ' < "/proc/$existing_pid/cmdline" \
        | grep -oE 'gpu-memory-utilization[= ]+[0-9.]+' | head -1 | grep -oE '[0-9.]+$' || true)
fi
GPU_UTIL="${GPU_UTIL:-0.85}"
echo "[restart] gpu-memory-utilization: $GPU_UTIL"

# Graceful stop. SIGTERM, give it 60s, then SIGKILL.
if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "[restart] stopping pid $existing_pid (SIGTERM)..."
    kill -TERM "$existing_pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        kill -0 "$existing_pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$existing_pid" 2>/dev/null; then
        echo "[restart] still alive after 60s, SIGKILL"
        kill -KILL "$existing_pid" 2>/dev/null || true
        sleep 2
    fi
fi

# Wait for GPU memory to drop. vLLM workers are subprocesses that don't all
# release immediately on parent exit — poll until used < 10GB or 30s elapsed.
echo "[restart] waiting for GPU memory release..."
used=0
for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [[ "$used" -lt 10000 ]] && break
    sleep 1
done
echo "[restart] GPU mem now ${used} MiB"

# Build --enable-lora plus any preloaded adapter flags.
lora_flags=(--enable-lora)
for spec in "${ADAPTERS[@]+"${ADAPTERS[@]}"}"; do
    lora_flags+=(--lora-modules "$spec")
done
if [[ ${#ADAPTERS[@]} -gt 0 ]]; then
    echo "[restart] starting with --enable-lora + ${#ADAPTERS[@]} preloaded adapter(s)"
else
    echo "[restart] starting with --enable-lora (no preloaded adapters; use load_lora.sh)"
fi

# Daemonize per CLAUDE.md daemon convention: PPID=1, stdout/stderr → timestamped log,
# stdin closed. VLLM_ALLOW_RUNTIME_LORA_UPDATING=True enables hot-swap via HTTP.
mkdir -p "$REPO/logs"
LOG="$REPO/logs/vllm_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[restart] log: $LOG"
VENV="${GEMMA_VENV:-/workspace/vllm-serve/.venv}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

setsid nohup "${VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$served_model" \
    --port 8000 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len 16384 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser pythonic \
    --dtype bfloat16 \
    "${lora_flags[@]}" \
    </dev/null >>"$LOG" 2>&1 & disown

new_pid=$!
echo "[restart] launched pid $new_pid; waiting up to ${WAIT_S}s for /v1/models..."
for i in $(seq 1 "$WAIT_S"); do
    # If the launched daemon died (e.g. CUDA OOM, bad adapter), bail
    # immediately rather than wait for the full timeout — the curl below
    # could otherwise succeed against a leftover stale vLLM.
    if ! kill -0 "$new_pid" 2>/dev/null; then
        echo "FATAL: launched vLLM (pid $new_pid) died during startup"
        tail -50 "$LOG"
        exit 1
    fi
    if curl -sf --max-time 2 http://localhost:8000/v1/models >/dev/null 2>&1; then
        echo "[restart] up after ${i}s"
        ps -o pid,ppid,cmd -p "$new_pid" 2>/dev/null | tail -1
        exit 0
    fi
    sleep 1
done
echo "FATAL: vLLM didn't respond within ${WAIT_S}s; killing orphaned daemon pid $new_pid"
kill -TERM "$new_pid" 2>/dev/null || true
sleep 2
kill -KILL "$new_pid" 2>/dev/null || true
tail -50 "$LOG"
exit 1
