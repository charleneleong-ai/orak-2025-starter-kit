# VLM Options for Agentic RL — May 2026

Model selection guide for the Orak game-playing agent stack.
Hard constraints: **vision required** (game screenshots), **tool calling required**, serve on
Cloud Run L4 (24GB VRAM), optionally train/fine-tune on A100 80GB.

> **Current baseline (post-PR #31, May 2026):** `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
> — Gemma 4 26B A4B in AWQ 4-bit (~13GB on disk, ~14GB GPU). Replaces the
> earlier E4B baseline because the AWQ-26B fits A100-40GB with KV cache headroom
> and the tau2-bench jump (E4B 57.5% → 26B A4B 85.5%) was decisive in PR #31's
> cross-game ablation (Stage D++ pokemon = 71.43% on AWQ-26B vs the 14.29%
> plateau on E4B). E4B still useful for fast-iteration harness validation
> runs (`serving/gemma_serve.sh unsloth/gemma-4-E4B-it`).

---

## Why VLM is non-negotiable

Agents use `supports_vision: true` in all game configs. Screenshots go directly into the LLM context.
Text-only models cannot process these and will error or silently ignore the images.

---

## Full open-source VLM landscape (May 2026)

### 1. Gemma 4 (Google DeepMind, Mar 2026) — current baseline family

All sizes natively multimodal (text + image). E2B/E4B also support audio.
All support function calling + Thinking mode. Apache 2.0.
vLLM 0.19.1: use `--tool-call-parser gemma4 --reasoning-parser gemma4`.

| Model | Arch | Active params | On-disk | L4 fit? | tau2-bench | MMMU Pro |
|---|---|---|---|---|---|---|
| Gemma 4 E2B | Dense + PLE | 2.3B | ~4GB | trivial | 29.4% | 44.2% |
| **Gemma 4 E4B** | Dense + PLE | 4.5B | **~16GB** | **current** | 57.5% | 52.6% |
| **Gemma 4 26B A4B** | **MoE** | **3.8B active** | ~52GB / ~13GB AWQ | **AWQ** | **85.5%** | **73.8%** |
| Gemma 4 31B | Dense | 30.7B | ~62GB / ~16GB AWQ | AWQ tight | 76.9% | 76.9% |

tau2-bench = agentic tool-use benchmark. Most relevant for game agents.

The 26B A4B is the standout: 26B total params, 3.8B active via 128 MoE experts.
Runs at 4B speed with 26B quality. In AWQ (~13GB) it fits L4 with ~11GB KV cache headroom.
The jump from E4B (57.5%) to 26B A4B (85.5%) on tau2-bench is massive.

vLLM serve command for Gemma 4 26B A4B on L4 (needs updated vLLM for gemma4 parser):
```
vllm serve google/gemma-4-26B-A4B-it \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --quantization awq --dtype auto --port 8000
```

AWQ checkpoint: community quants available on HuggingFace (search "gemma-4-26B-A4B AWQ").

---

### 2. Qwen3-VL (Alibaba, Oct 2025) — purpose-built visual agent series

Dedicated VLM for GUI/agent tasks. All sizes in Instruct + Thinking variants.
vLLM 0.19.1 fully supports. Tool parser: `hermes`. 24M+ downloads on HF.

| Model | On-disk | L4 fit? | MathVista | Notes |
|---|---|---|---|---|
| Qwen3-VL-4B | ~8GB | trivial | — | Budget |
| **Qwen3-VL-8B-FP8** | **~8GB** | **excellent** | 79-80 | Official FP8, 16GB KV headroom |
| Qwen3-VL-8B-Instruct | ~16GB | comfortable | 79-80 | BF16, battle-tested |
| Qwen3-VL-30B-A3B (AWQ) | ~15GB | yes | — | MoE, fast inference |
| Qwen3-VL-32B (AWQ) | ~17GB | tight | — | MMMU 78.1, 95% of 235B flagship |

Qwen3-VL-8B beats GPT-4o on MathVista. Purpose-built for GUI/agent workflows.
FP8 variant is the recommended safe default — 8GB leaves room for full 16K context.

---

### 3. InternVL3 (Shanghai AI Lab, 2025) — top VLM leaderboard performer

Consistently near top of OpenVLM Leaderboard. Good vision quality. vLLM supported.

| Model | On-disk | L4 fit? | Notes |
|---|---|---|---|
| InternVL3-8B | ~16GB | yes | Strong benchmark scores at 8B |
| InternVL3-14B (AWQ) | ~7GB | easy | Higher capability |
| InternVL3-38B (AWQ) | ~19GB | tight | Near leaderboard top for open models |

Tool calling requires custom chat template — less out-of-box than Qwen3-VL or Gemma 4.

---

### 4. LLaMA 4 Vision (Meta, 2025)

| Model | On-disk | L4 fit? | Notes |
|---|---|---|---|
| LLaMA 4 Scout (17B-16E MoE, AWQ) | ~17GB | tight | 3.4B active, fast |
| LLaMA 4 Maverick (17B-128E) | huge | no single GPU | Multi-GPU only |

Community license (not Apache 2.0) — check commercial restrictions. vLLM supported.

---

### 5. Phi-4-reasoning-vision-15B (Microsoft, Apr 2026)

Very new (trending May 2026). Reasoning + vision in 15B. Apache 2.0.

| Model | On-disk | L4 fit? | Notes |
|---|---|---|---|
| Phi-4-reasoning-vision-15B | ~30GB / ~8GB AWQ | AWQ | Strong reasoning, new |

vLLM compatibility and community testing limited as of May 2026 — worth monitoring.

---

### 6. Molmo2-7B (AllenAI, Jan 2026) — spatial/grounding specialist

Open weights AND open training data (no proprietary distillation). Best-in-class
video grounding and spatial pointing. Outperforms Qwen3-VL on video counting (35.5 vs 29.6).
Interesting for game agents that need pixel-level localization.

| Model | On-disk | L4 fit? | Notes |
|---|---|---|---|
| Molmo2-7B-D | ~14GB | comfortable | Strong grounding; tool calling less mature |

---

### 7. Qwen3.5 / Qwen3.6 (Alibaba, Feb-Apr 2026) — text+vision early fusion

"Unified Vision-Language Foundation" with early fusion training. Claims to outperform Qwen3-VL.
vLLM 0.19.1 lists `Qwen3_5ForConditionalGeneration` support.

| Model | L4 fit? | Notes |
|---|---|---|
| Qwen3.5-9B | BF16 (~18GB) | Vision API compatibility with vLLM unconfirmed |
| Qwen3.6-35B-A3B (4-bit) | ~17.5GB tight | MMMU 81.7, just released Apr 2026 |

**Risk:** Image input via vLLM's OpenAI-compatible API is not community-confirmed for Qwen3.5/3.6.
Prefer Qwen3-VL or Gemma 4 until this is verified.

---

## L4 (24GB) serving recommendations — ranked

| Rank | Model | VRAM | tau2-bench | Maturity | Notes |
|---|---|---|---|---|---|
| **1** | **Gemma 4 26B A4B (AWQ) — current baseline** | ~13GB | **85.5%** | Shipped via PR #31 | PR #31 Stage D++ pokemon = 71.43% |
| **2** | **Qwen3-VL-8B-FP8** | ~8GB | GUI-focused | 24M downloads | Safest swap, most context |
| 3 | Gemma 4 31B (AWQ) | ~16GB | 76.9% | Good | High capability, tight |
| 4 | Qwen3-VL-32B (AWQ) | ~17GB | — | Good | Best pure VLM quality |
| 5 | InternVL3-8B | ~16GB | — | Good | Leaderboard strong |
| 6 | Gemma 4 E4B (legacy baseline) | ~16GB | 57.5% | Proven | Use for fast-iteration smoke runs |
| 7 | Qwen3.6-35B-A3B (4-bit) | ~17.5GB | 81.7 MMMU | Very new | Wait for vLLM verify |

---

## Train on A100 80GB → Serve on L4

### Path A — Gemma 4 26B A4B QLoRA (recommended)

The MoE architecture means QLoRA trains on the active ~3.8B params while expert weights
stay frozen. Memory-efficient for a 26B model.

```
A100 80GB                                   L4 24GB
─────────────────────────                   ─────────────────────────
QLoRA (NF4 base ~13GB + adapters ~5GB)  →   AWQ quantized (~13GB)
Total: ~18GB  ← comfortable                  3.8B active inference
                                              full 16K context
```

Steps: QLoRA train → reload BF16 (~52GB on A100 80GB) → merge → AWQ quantize → deploy.

### Path B — Qwen3-VL-8B (simplest, no quantization)

Full fine-tune on A100 80GB (~42GB) or LoRA (~20GB).
Serve directly as BF16 (~16GB) or FP8 (~8GB) on L4. No quantization step needed.

### Path C — Stay on Gemma 4 E4B, run training experiment first

LoRA on the existing A100-40GB (pause vLLM during training, ~0 incremental cost).
This is the gating experiment before committing to a model switch.

---

## When to fine-tune vs just serve a better base

The MACLA findings (`macla_findings.md`) give the answer per-game:

- **mario (100% with base model):** Fine-tuning adds nothing here. MACLA carry-over
  was the bottleneck, not the model. Don't train on mario data.
- **2048 (ceiling ~6-8%):** Best candidate for RL. Clean numeric reward, combinatorial
  state space benefits from internalised strategy (corner anchoring). But first check:
  does swapping to Gemma 4 26B A4B base lift scores without any training?
- **pokemon (14.29%, post-fix):** Eval harness was confounded. Re-baseline post-fix
  before deciding whether to train.

**Gating rule (from `training_plan.md`):** Run `harness_check` sweep with new model.
If mean score > Stage A baseline +10% on at least one game → model IS the bottleneck
→ training will compound. If flat → architecture/prompting is the bottleneck → train less,
improve MACLA more.

---

## GPU cost matrix (Gemma 4 E4B / 26B A4B LoRA, 36M tokens x 3 epochs)

| GPU | Phase 1 SFT wall-clock | GRPO viable? | $/hr | Phase 1 cost |
|---|---|---|---|---|
| A100 40GB (on-hand) | ~2hr | yes (N=8) | $0 | $0 |
| A100 80GB | ~1.5hr | yes (N=16) | $1.60 | ~$2.40 |
| H100 80GB FP8 | ~45min | yes (N=16+) | $2.50 | ~$1.90 |

26B A4B QLoRA trains similarly fast to E4B SFT (active params ~equal).
Phase 2 GRPO multiplier: ~10-20x SFT wall-clock (rollouts dominate).

---

## References

- Gemma 4 vLLM recipe: https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it
- Gemma 4 HF: https://huggingface.co/google/gemma-4-26B-A4B-it
- Qwen3-VL-8B-FP8: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8
- Qwen3-VL collection: https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe
- InternVL3-8B: https://huggingface.co/OpenGVLab/InternVL3-8B
- Molmo2-7B: https://huggingface.co/allenai/Molmo2-7B-D
- OpenVLM Leaderboard: https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
- Training pipeline: docs/experiments/gemma/training_plan.md
- Experiment findings: docs/experiments/gemma/macla_findings.md

Last updated: 2026-05-11 (post-PR #31 cross-game AWQ-26B ablation merge)
