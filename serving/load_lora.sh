#!/usr/bin/env bash
# Hot-swap a LoRA adapter into the running vLLM (no restart).
#
# Requires vLLM was launched with --enable-lora AND
# VLLM_ALLOW_RUNTIME_LORA_UPDATING=True — both are set by
# serving/restart_vllm.sh, so the typical flow is:
#   1. ./serving/restart_vllm.sh         (one-time, switches to --enable-lora mode)
#   2. ./serving/load_lora.sh ...         (every subsequent adapter update)
#
# Usage:
#   ./serving/load_lora.sh --name current --path /workspace/orak-gspo/artifacts/gspo_lora_v1
#   ./serving/load_lora.sh --name current --path /new/path --no-replace   # fail if name exists
set -uo pipefail

NAME=""
LORA_PATH=""
INPLACE=true
PORT=8000

usage() {
    cat <<USAGE
Hot-swap a LoRA adapter into running vLLM (no restart).

  --name <name>     adapter slot name (used by clients as "model": <name>)
  --path <dir>      adapter dir containing adapter_config.json
  --no-replace      fail if name already exists (default: replace in-place)
  --port <int>      vLLM port (default 8000)

vLLM must have been launched with --enable-lora AND
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True (see serving/restart_vllm.sh).
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name) NAME="$2"; shift 2;;
        --path) LORA_PATH="$2"; shift 2;;
        --no-replace) INPLACE=false; shift;;
        --port) PORT="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "unknown flag: $1"; usage; exit 1;;
    esac
done

[[ -n "$NAME" ]] || { echo "FATAL: --name required"; usage; exit 1; }
[[ -n "$LORA_PATH" ]] || { echo "FATAL: --path required"; usage; exit 1; }
[[ -d "$LORA_PATH" ]] || { echo "FATAL: adapter dir missing: $LORA_PATH"; exit 1; }
[[ -f "$LORA_PATH/adapter_config.json" ]] \
    || { echo "FATAL: $LORA_PATH/adapter_config.json missing — not a valid LoRA adapter"; exit 1; }

curl -sf --max-time 3 "http://localhost:${PORT}/v1/models" >/dev/null \
    || { echo "FATAL: vLLM not responding on :${PORT}"; exit 1; }

echo "[load_lora] POST /v1/load_lora_adapter name=$NAME path=$LORA_PATH in_place=$INPLACE"
# Use python3 json.dumps for proper escaping — printf %s would break on
# paths with quotes/backslashes/newlines (rare on /workspace/... but cheap to fix).
payload=$(python3 -c '
import json, sys
print(json.dumps({
    "lora_name": sys.argv[1],
    "lora_path": sys.argv[2],
    "load_inplace": sys.argv[3] == "true",
}))
' "$NAME" "$LORA_PATH" "$INPLACE")
curl -sf -X POST "http://localhost:${PORT}/v1/load_lora_adapter" \
    -H "Content-Type: application/json" \
    -d "$payload"
echo
