# Stage H — Qwen 3.6 27B FP8 Ceiling Check

**Status:** scaffolded, awaiting vLLM model swap  •  **Branch:** `feat/stage-h-qwen-ceiling`

## Hypothesis

Stages A→G ([diagnosis doc](gemma/cross-stage-diagnosis.md)) ruled out the procedure layer, action validation, and self-reflection schedule density as the bottleneck for the pokemon milestone-4 → 5 ceiling at 57.14%. The convergent diagnosis pointed at **LLM reasoning at the milestone boundary** as the constraint.

Stage H tests one specific hypothesis: **is the 57.14% cap a Gemma 4-26B-A4B model-capacity ceiling, or is it the prompt/scaffold?**

The control variable is **model lineage / training**. Everything else (Stage D stack: vector memory + subtask planner + Bayesian procedures + per-game self-reflection) is held constant. Only the served model changes:

| Aspect | Gemma 4-26B-A4B AWQ-4bit (Stages A–G) | Qwen 3.6 27B FP8 (Stage H) |
|---|---|---|
| Total params | 26B (4B active, MoE) | 27B (dense) |
| Quant | AWQ-Int4 | FP8 |
| Release | mid-2025 | Apr 2026 (~3 weeks before this run) |
| `max_model_len` | 16384 | 8192 (FP8 weights heavier than Int4) |
| `--tool-call-parser` | pythonic | hermes |
| Same MACLA stack? | yes | yes (`configs/qwen36_27b.yaml` mirrors `gemma_26b.yaml`) |

## Decision criteria

| n=3 result | Interpretation |
|---|---|
| **mean ≥ 71.43%** (banks milestone 5+) | Stage D ceiling was model capacity. Qwen 3.6 breaks past it. Recommendation: switch the default served model to Qwen 3.6 27B FP8 and re-run the ablation. |
| **mean ≈ 57.14%** (ties Gemma ceiling) | Cap is **upstream** of the model — it's the prompt/scaffold/reasoning chain. Pivots us to planner-prompt overhaul as the next experiment. |
| **mean < 42%** (worse than Gemma) | Qwen 3.6 underperforms on this task. Suspect tool-call parser mismatch (`hermes` vs game's expected format) or FP8 numerical degradation. Diagnose before drawing conclusions. |

## Run

Pre-flight (manual — vLLM is single-tenant):

```bash
# 1. Stop gemma vLLM (any active gemma run will fail; coordinate with /tmp/procedure-escape state)
pkill -f 'vllm.entrypoints.openai.api_server'

# 2. Start qwen vLLM
nohup ./serving/qwen_serve.sh >/tmp/qwen_serve.log 2>&1 &
disown

# 3. Wait for vLLM ready (FP8 27B init ~2-3 min; weights download first time ~10-20 min)
until curl -s http://localhost:8000/v1/models | grep -qi 'qwen3'; do sleep 10; done

# 4. Run the n=3 launcher
nohup bash experiments/stage_h_qwen_ceiling/run_pokemon_n3.sh \
  >/tmp/stage_h_pokemon_n3.log 2>&1 &
disown
```

Wall-clock estimate (n=3 × ~50 min/iter at 300 steps): **~150 min total**.

## Files added on `feat/stage-h-qwen-ceiling`

- `serving/qwen_serve.sh` — vLLM launcher (FP8 27B, `--tool-call-parser hermes`, `max_model_len=8192`)
- `configs/qwen36_27b.yaml` — Hydra root, pokemon-only (cross-game queued separately)
- `configs/pokemon_red/agent/qwen36_27b.yaml` — agent config mirroring `gemma_26b.yaml` with model swap
- `run.py` — `ExperimentConfigName.QWEN36_27B` typer enum
- `experiments/stage_h_qwen_ceiling/run_pokemon_n3.sh` — n=3 launcher writing to `experiments/stage_h_qwen_ceiling/qwen36_27b/results.jsonl`
- This writeup

## Out of scope

- Cross-game (mario/2048) — wait for pokemon result. If Qwen 3.6 breaks the ceiling, expand to all 3 games.
- Comparison with **Qwen 3.5 27B FP8** (Feb 2026) and **Qwen 3.5 35B-A3B-GPTQ-Int4** (MoE) — keep as follow-ups if Stage H result is surprising in either direction.
- Switching the project's served default model on master — only after the n=3 result is in and reproducible.
- A truly stronger ceiling check via API (Claude / GPT-4o) — separate experiment, separate cost profile.
