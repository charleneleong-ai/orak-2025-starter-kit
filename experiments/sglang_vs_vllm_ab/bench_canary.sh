#!/usr/bin/env bash
# Time one fixed pokemon canary against whatever backend is serving on :8000,
# and append a labelled metrics row to results.jsonl. One run = one A/B arm.
#
# The backend is NOT launched here — bring it up first with the matching serve
# script, then run this with the arm label:
#
#   # Arm 1 — vLLM baseline (no prefix cache)
#   ./serving/qwen_serve.sh &                       # QWEN_PREFIX_CACHING unset
#   ./experiments/sglang_vs_vllm_ab/bench_canary.sh vllm-nocache
#
#   # Arm 2 — vLLM, prefix cache on
#   QWEN_PREFIX_CACHING=1 ./serving/qwen_serve.sh &
#   ./experiments/sglang_vs_vllm_ab/bench_canary.sh vllm-cache
#
#   # Arm 3 — SGLang (RadixAttention default-on)
#   ./serving/sglang_serve.sh &
#   ./experiments/sglang_vs_vllm_ab/bench_canary.sh sglang
#
# Keep the model, N_STEPS, and config identical across arms — only the backend
# changes. Env overrides: N_STEPS, CONFIG_NAME, ENV_CFG, GAME.
set -euo pipefail

ARM="${1:?usage: bench_canary.sh <arm-label>  e.g. vllm-nocache | vllm-cache | sglang}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

N_STEPS="${N_STEPS:-100}"
GAME="${GAME:-pokemon_red}"
ENV_CFG="${ENV_CFG:-configs/pokemon_red/env/default.yaml}"
OUTDIR="experiments/sglang_vs_vllm_ab"
RESULTS="$OUTDIR/results.jsonl"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="ab_${ARM}_${TS}"
export GAME_DATA_DIR="/tmp/orak-ab-${ARM}"
mkdir -p "$GAME_DATA_DIR/game_logs/${GAME}"

# Pre-flight: backend up + which model it serves (both serve scripts use :8000).
served="$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
[[ -z "$served" ]] && { echo "FATAL: no OpenAI-compatible server on :8000 (start a serve script first)"; exit 1; }
echo "[bench] arm=$ARM  served=$served  N_STEPS=$N_STEPS  run_id=$RUN_ID"

# Sample peak GPU memory + mean util in the background for the run's duration.
gpu_samples="$(mktemp)"
( while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | head -1 >> "$gpu_samples"
    sleep 2
  done ) & sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT

# Fixed canary: temporarily pin max_steps, run, restore.
orig_steps="$(grep -m1 '^max_steps:' "$ENV_CFG" | sed 's/max_steps: *//')"
restore() { sed -i "s/^max_steps: .*/max_steps: ${orig_steps:-300}/" "$ENV_CFG"; }
trap 'restore; kill "$sampler_pid" 2>/dev/null || true' EXIT
sed -i "s/^max_steps: .*/max_steps: $N_STEPS/" "$ENV_CFG"

start=$(date +%s.%N)
uv run python run.py \
    ${CONFIG_NAME:+--config-name "$CONFIG_NAME"} \
    --local --games "$GAME" \
    --run-id "$RUN_ID" \
    -d "canary A/B arm=$ARM backend=$served" || true
end=$(date +%s.%N)
restore

kill "$sampler_pid" 2>/dev/null || true
wall=$(python3 -c "print(round($end - $start, 1))")

# Throughput from the canary's per-request log (prompt + completion tokens).
raw_log="$GAME_DATA_DIR/${GAME}/$RUN_ID/logs/raw_requests.jsonl"

python3 - "$ARM" "$served" "$N_STEPS" "$wall" "$raw_log" "$gpu_samples" "$RESULTS" <<'PY'
import json, sys, pathlib
arm, served, n_steps, wall, raw_log, gpu_samples, results = sys.argv[1:8]
n_steps, wall = int(n_steps), float(wall)

prompt_tok = gen_tok = reqs = 0
p = pathlib.Path(raw_log)
if p.exists():
    for line in p.read_text().splitlines():
        try:
            u = (json.loads(line).get("usage") or {})
        except Exception:
            continue
        prompt_tok += u.get("prompt_tokens", 0)
        gen_tok += u.get("completion_tokens", 0)
        reqs += 1

util = mem = 0.0
g = pathlib.Path(gpu_samples)
if g.exists() and g.read_text().strip():
    rows = [r.split(",") for r in g.read_text().splitlines() if "," in r]
    utils = [float(x[0]) for x in rows]
    mems = [float(x[1]) for x in rows]
    util = round(sum(utils) / len(utils), 1) if utils else 0.0
    mem = max(mems) if mems else 0.0

row = {
    "arm": arm, "backend": served, "n_steps": n_steps,
    "wall_s": wall, "s_per_step": round(wall / n_steps, 3) if n_steps else None,
    "requests": reqs, "prompt_tokens": prompt_tok, "gen_tokens": gen_tok,
    "gen_tok_per_s": round(gen_tok / wall, 1) if wall else None,
    "mean_gpu_util_pct": util, "peak_gpu_mem_mib": mem,
}
with open(results, "a") as f:
    f.write(json.dumps(row) + "\n")
print("[bench] " + json.dumps(row, indent=2))
PY

echo "[bench] appended -> $RESULTS"
