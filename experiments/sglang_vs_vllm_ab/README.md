# Canary A/B — vLLM (prefix-cache) vs SGLang

Which serving backend is fastest for orak agentic eval, on the same model and the same fixed pokemon canary. Tracked as task #41.

## Hypothesis

The dominant lever is **prefix caching**, not the server choice. orak's workload is its best case: every step re-sends a long, near-identical prompt (system + game rules + history), and parallel rollouts share the same system prefix. So:

- **vLLM baseline → vLLM prefix-cache-on** should be the biggest jump. Note `serving/qwen_serve.sh` ships prefix caching **off** today, so the current production setup is the no-cache baseline.
- **vLLM-cache → SGLang** (RadixAttention, on by default) should be a smaller delta — SGLang's win is the more aggressive tree-structured reuse (helps `n>1` sampling and the short-output **fast action-LLM** path most).
- The **long-CoT planner** path is decode-bound, so caching helps it least — expect the SGLang edge to shrink on reasoning-heavy arms.

## Arms

Same model (`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` default), same `N_STEPS`, same agent config — only the backend changes. One A100-40GB fits one model, so run the arms sequentially (stop the previous server first).

| Arm | Serve command |
|---|---|
| `vllm-nocache` (baseline) | `./serving/qwen_serve.sh` |
| `vllm-cache` | `QWEN_PREFIX_CACHING=1 ./serving/qwen_serve.sh` |
| `sglang` | `./serving/sglang_serve.sh` |
| `sglang-noradix` (control) | `SGLANG_RADIX=0 ./serving/sglang_serve.sh` |

All serve on `:8000` with the same OpenAI tool-call + reasoning-parser contract, so the canary and agent harness are unchanged across arms.

## Run

```bash
# one-time: bootstrap the SGLang serving venv (authorized install, ~minutes)
uv venv /workspace/sglang-serve/.venv --python 3.11
/workspace/sglang-serve/.venv/bin/pip install "sglang[all]"

# per arm: start the backend, wait for :8000/v1/models, then bench
QWEN_PREFIX_CACHING=1 ./serving/qwen_serve.sh &        # (example: arm 2)
./experiments/sglang_vs_vllm_ab/bench_canary.sh vllm-cache
```

`bench_canary.sh` pins `max_steps`, runs `run.py --local --games pokemon_red`, samples GPU util/mem, scrapes prompt/gen tokens from the canary's `raw_requests.jsonl`, and appends a row to `results.jsonl`:

```json
{"arm": "...", "wall_s": ..., "s_per_step": ..., "gen_tok_per_s": ..., "mean_gpu_util_pct": ..., "peak_gpu_mem_mib": ...}
```

## Fairness controls

- **Warm vs cold cache:** the prefix-cache win only shows once the shared prefix is cached. Discard the first run (cold) or run `N_STEPS` large enough that warm steps dominate; compare steady-state `s_per_step`.
- **Identical canary:** same model, `N_STEPS`, agent config, and ROM/seed across arms. The script pins `max_steps` and restores it on exit.
- **One model on the GPU:** `pkill -f vllm.entrypoints.openai.api_server` (or the sglang server) before switching arms — confirm with `nvidia-smi`.
- **Sanity floor:** include `sglang-noradix` and the `vllm-nocache` baseline so a "caching helps" claim is grounded against the no-cache control.

## Read the result

Primary metric: steady-state **`s_per_step`** (wall-clock per agent step — what a sweep actually pays). Secondary: `gen_tok_per_s` and `mean_gpu_util_pct`. Before concluding "switch backends," check the real bottleneck isn't the game-server roundtrip or the per-step sequential dependency rather than the LLM server — if GPU util is low across all arms, the server isn't the constraint.
