# Training Plan — Gemma 4 E4B agentic RL on Orak

How to turn the harness + cognitive infra (PRs [#25](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/25), [#26](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/26)) into a self-improving model. Phased, cost-aware, grounded in what we already have vs what's missing.

> **Status: planning doc.** No training runs yet. Numbers below are estimates from public Unsloth/TRL benchmarks + our observed inference throughput; treat as ±50%.

## Current state (2026-05-02)

| Layer | Status | Where |
|---|---|---|
| Cognitive substrate (Stage A→D) | shipped | merged via [#25](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/25) + [#26](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/26) |
| Stage D enabled on pokemon | shipped | `configs/pokemon_red/agent/gemma.yaml` |
| Stage D enabled on mario + 2048 | **PR open** | [#28](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/28) — ablation sweep running |
| `autoresearch-verdict` CLI | shipped (v0.5.1) | [autoresearch#10](https://github.com/charleneleong-ai/autoresearch/pull/10) |
| Verdict spec for ablation | shipped | [`experiments/stage_d_verdict.yaml`](../../../experiments/stage_d_verdict.yaml) |
| Stage D ablation sweeps | **running** | `logs/stage_d_sweeps_*.log` |
| Verdict daemon | **polling** | `logs/stage_d_verdict_*.log` (posts to PR #28 once both sweeps reach 2 iters) |
| Phase 1 RFT (training) | not started | needs `filter_top_k.py` + `sft_unsloth.py` |
| Phase 2 GRPO | not started | gated on Phase 1 outcome |
| Phase 3 self-improvement loop | not started | gated on Phase 2 outcome |

## Direction adjustments based on observed results

The cross-game scoreboard from Stage A vs Stage C runs already gives us actionable signal *before* training starts. Three deltas to the originally-picked path:

| Observation | Implication | Action |
|---|---|---|
| **2048 lifted +58% with vmem alone** (Stage A best=4.36 → Stage C v2 best=6.46) | Cognitive layer landing well here; training likely to compound rather than be wasted | Keep 2048 in Phase 1 RFT; expect best per-game lift here |
| **Mario neutral with vmem** (memory_count stuck at 1, retrievals fired but didn't match useful events) | Bottleneck isn't decomposition or recall; it's perception/timing latency | Don't expect huge Phase 1 lift on mario from trajectory SFT alone — reactive games need latency-aware reward shaping or a faster-throughput model |
| **Pokemon hit 14.29% / 0.00 ceiling on Stage A, B, C** | Model-too-small signal **already present** on at least one game | **Don't wait for Phase 1 to fail** to test the model swap. Run a [Kimi-VL-A3B-Thinking-2506](https://hf.co/moonshotai/Kimi-VL-A3B-Thinking-2506) baseline on pokemon **in parallel** with Phase 1 RFT — the swap is cheap-to-test (40GB-friendly, vision-native) and gives an early signal whether tier-1 model swap rescues pokemon |

This makes the path **two-track** instead of strictly sequential:

- **Track A — training**: Phases 1 → 2 → 3 on Gemma 4 E4B, driven by 2048's promising signal.
- **Track B — capacity**: Kimi-VL-A3B-Thinking-2506 baseline on pokemon first, then mario. If pokemon breaks the 14.29% ceiling on Kimi, that's the cue to migrate the substrate.

Tracks converge at the Phase 1 gate: PASS → continue Track A, FAIL but Track B promising → migrate substrate then redo Phase 1 on the new model.

## What we have

| Piece | Source | Status |
|---|---|---|
| ShareGPT-shaped trajectories | `agents/_harness/trajectory.py` (PR #25) | per-episode files in `game_logs/<game>/<run_id>/logs/trajectory_samples.jsonl` |
| Per-step rewards | `agents/macla/online_evaluator.py` | shaped: position/score/lives for mario, score/max_tile/corner-anchor for 2048, score/flags/map-transitions for pokemon |
| Episode scores | `evaluation_summary.json` per run | `evaluation_score` (0-100) |
| Failure attribution | `failed_trajectories.jsonl` + `is_fallback` flag | clean episodes vs corrupted ones split automatically |
| Eval harness | `experiments/autoresearch.py` | already runs the loop |
| LoRA inference path | vLLM `--enable-lora` | not currently configured but supported |
| Existing LoRA dir | `/workspace/gemma4_rl/gemma_4_lora` | empty placeholder |

## What's missing

| Piece | Effort | Blocker for |
|---|---|---|
| Trajectory filter (`filter_top_k.py`) | 1-2 hr | Phase 1 |
| Unsloth SFT runner (filtered ShareGPT → LoRA) | 4-6 hr | Phase 1 |
| `serving/gemma_serve.sh` w/ `--enable-lora` | 30 min | Phase 1 deploy |
| GRPO trainer config (TRL or VeRL) | 1-2 days | Phase 2 |
| Reward fn wiring (game score → numeric reward) | 2-4 hr | Phase 2 |
| Self-improvement loop scheduler | 1 hr (cron) | Phase 3 |
| Procedure → SFT data converter | 1 day | Phase 4 |

## Phases

### Phase 1 — Rejection-sampling fine-tune (RFT)

Filter `trajectory_samples.jsonl` to top-K% by `final_score` per game. LoRA-SFT Gemma 4 E4B on those via Unsloth.

**Hyperparameters (starting point):**
- LoRA rank: 16-32, alpha: 32, target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- Sequence length: 4096
- Batch size: 2 with gradient accumulation 8 (effective 16)
- LR: 2e-4 cosine, warmup 5%
- Epochs: 3
- Optimizer: AdamW 8-bit (bitsandbytes)

**Data sizing (rough):** 3 games × 50 episodes × 200 steps = 30K trajectory steps. After top-30% filter: ~9K steps × 4K tokens = **~36M tokens** to train on.

### Phase 2 — GRPO with shaped rewards

DeepSeek-R1's group-relative policy optimization. Sample N rollouts per checkpointed game state, normalize rewards within the group, no critic network.

**Wiring:**
- Reference model: base Gemma 4 E4B (frozen)
- Policy: Phase 1 LoRA
- N (group size): 8
- Reward: `OnlineAgentEvaluator.evaluate_step` shaped reward + final score bonus
- `is_fallback=True` steps → −1.0 reward (silent fallbacks become training signal)

**Cost:** ~10-50× SFT cost for the same data — most of the wall-clock is rollouts via vLLM, not the gradient update.

### Phase 3 — Self-improvement loop

Weekly cron: rollout → filter → SFT → redeploy. Curriculum: mario (highest historical baseline) → 2048 → pokemon. Track score trend across loops.

The autoresearch loop (`experiments/autoresearch.py`) is already 80% of this. What's needed: a wrapper script that runs `[autoresearch sweep, filter, SFT, deploy]` end-to-end.

### Phase 4 — Procedure distillation

Convert MACLA's `Procedure` objects to `(precondition_obs, action, postcondition_obs)` SFT triples, augment Phase 1 data. Goal: model internalizes procedures, MACLA selector can be ablated. Cleanest deployment: pure-LLM agent, no procedure runtime.

---

## GPU options + wall-clock estimates

Throughput numbers assume Gemma 4 E4B (~4B effective params), bf16 unless noted, Unsloth + LoRA rank 16. Numbers are rough — actual depends on seq len, batch, grad-checkpointing.

| GPU | VRAM | SFT throughput | Phase 1 wall-clock (36M tokens × 3 ep) | GRPO viable? |
|---|---|---|---|---|
| RTX 4090 | 24 GB | ~6K tok/s (QLoRA 4-bit) | ~5 hr | no — too tight for rollouts |
| L40S 48 GB | 48 GB | ~10K tok/s | ~3 hr | marginal |
| A100 40 GB | 40 GB | ~12K tok/s | ~2.5 hr | tight (N=4 rollouts max) |
| A100 80 GB | 80 GB | ~14K tok/s | ~2 hr | yes (N=8-16) |
| H100 80 GB (FP8) | 80 GB | ~30K tok/s | ~1 hr | yes (N=16+) |
| H200 / H100 NVL | 96-141 GB | ~35K tok/s | ~50 min | yes, comfortable |

GRPO multiplier: ~10-20× SFT wall-clock for one full GRPO sweep (rollouts dominate).

## Cloud cost matrix

Approximate **on-demand** rates from major cloud providers (RunPod / Lambda / Coreweave), April 2026. Spot is ~30-50% cheaper.

| GPU | $/hr (rough) | Phase 1 cost (one run) | Phase 2 cost (one GRPO sweep) | Total cycle |
|---|---|---|---|---|
| RTX 4090 (cloud) | $0.50 | $2.50 | not viable | $2.50 (SFT only) |
| L40S 48 GB | $1.50 | $4.50 | ~$45 | ~$50 |
| A100 40 GB | $1.20 | $3 | $25-50 (tight) | ~$30-55 |
| A100 80 GB | $1.60 | $3.20 | $30-65 | ~$33-68 |
| **H100 80 GB** | $2.50 | $2.50 | $20-50 | **~$22-52** |

**Surprising result:** H100 ends up cheapest per training cycle despite the higher hourly rate, because FP8 cuts wall-clock by ~3×.

### Self-improvement loop cost (Phase 3 — the recurring spend)

If you do **weekly RFT loops** (Phase 1 only, no GRPO yet) for a quarter:

| GPU | Cost per loop | Cost per quarter (12 loops) |
|---|---|---|
| RTX 4090 (cloud spot) | ~$1.50 | ~$18 |
| A100 40 GB | ~$3 | ~$36 |
| H100 80 GB | ~$2.50 | ~$30 |

If you graduate to **monthly GRPO loops**:

| GPU | Cost per loop | Cost per quarter (3 loops) |
|---|---|---|
| L40S 48 GB | ~$50 | ~$150 |
| A100 80 GB | ~$65 | ~$195 |
| H100 80 GB | ~$50 | ~$150 |

These numbers assume cloud spot. On-prem amortized over a year is cheaper if utilization > 30%.

## Operational note — vLLM coexistence

The current A100-40GB is **fully occupied by vLLM serving** for autoresearch (model weights + KV cache). Training on the same GPU requires:

1. **Pause autoresearch** → train → redeploy (the simple option, ~30 min downtime per loop)
2. **Two-GPU split**: vLLM stays on A100-40GB, training happens on a second GPU (RTX 4090 / L40S). Cleanest workflow, no autoresearch downtime. Cost: +cheap-GPU hours per loop.
3. **One 80GB GPU**: vLLM + Unsloth simultaneously in different CUDA contexts, no swapping. Most ergonomic but most expensive per hour.

Recommendation depends on how often you retrain. Daily → option 3. Weekly → option 2. Monthly → option 1 fine.

## Decision matrix — per-phase hardware

What we have: **A100-40GB** (currently fully used by vLLM serving Gemma 4 E4B). When does that suffice vs when do we need to upgrade?

| Phase / activity | A100-40GB sufficient? | A100/H100-80GB unlocks | Decision |
|---|---|---|---|
| **Phase 1 RFT** (LoRA-SFT on filtered trajectories) | **Yes** — pause vLLM, run SFT, redeploy. ~3 hr of inference downtime per loop, $0 incremental. | Run SFT alongside vLLM in different CUDA contexts, no autoresearch downtime. | Stay on 40GB through Phase 1. |
| **Phase 2 GRPO** (rollouts + gradient updates) | **Tight** — N=4 group size max; rollouts compete with KV cache for VRAM. | Comfortable N=8–16 group sizes; can run alongside vLLM rollouts cleanly. | Cloud H100-80GB spot for the weekend (~$25–50 per cycle), don't burn on-prem 40GB on this. |
| **Phase 3 self-improvement loop** (recurring rollout → filter → SFT → redeploy) | Yes if monthly cadence; tight if weekly. | Yes for any cadence including daily. | Match cadence to GPU plan: monthly = 40GB stays fine; weekly+ = consider second cheap GPU (RTX 4090) or upgrade primary to 80GB. |
| **Tier 1 model swap** (Kimi-VL-A3B-Thinking-2506, 16B/3B-active MoE) | **Yes** — bf16 fits ~32GB | More headroom for KV cache | 40GB OK; test pokemon ceiling cheaply. |
| **Tier 2 model swap** (Qwen3.6-35B-A3B-FP8) | **Tight** — ~35GB | Comfortable | 40GB borderline; 80GB preferred. |
| **Tier 3 model swap** (Qwen3.6-27B-FP8) | Yes | Yes | 40GB fine for FP8; bf16 needs 80GB. |
| **Tier 4 model swap** (Qwen3.6-27B at bf16) | **No** — ~54GB | Yes | Only viable on 80GB. |

**Recommendation tied to current results:** stay on the existing A100-40GB through Phase 1 RFT and Track B's Kimi-VL tier-1 swap test (both fit). Decide on 80GB upgrade based on whether **either** signal lands:

- **Phase 1 PASS but Phase 2 GRPO needed** → cloud H100-80GB spot for weekends (~$25–50/cycle), keep 40GB on-prem for continuous serving.
- **Phase 1 FAIL + Tier 1 (Kimi-VL) breaks pokemon ceiling** → migrate primary to 80GB so Tier 4 (Qwen3.6-27B bf16) is testable as a follow-on.
- **Phase 1 FAIL + Tier 1 doesn't help either** → don't upgrade GPU yet; the hypothesis is "model class," not "model size." Test Tier 2/3 on the existing 40GB first.

Avoid the "upgrade preemptively" trap — the current A100-40GB does everything we need until evidence forces the next decision.

## Next steps — execution order

| # | Step | Trigger | Outcome | Effort |
|---|---|---|---|---|
| 1 | **Wait for Stage D verdict comment** on PR #28 | both ablation sweeps reach 2 iters (≈ 1 hr wall-clock) | Verdict daemon auto-posts a HELPS / NEUTRAL / REGRESSES table | none — auto |
| 2 | **Decide Stage D disposition based on verdict** | step 1 lands | branch — see [Verdict-driven branches](#verdict-driven-branches) below | 30 min docs |
| 3 | **Merge PR #28** | step 2 done | Stage D config locked in (or reverted on regressing games) | 5 min |
| 4 | **Build `experiments/training/filter_top_k.py`** | regardless of verdict | top-K trajectory filter, ~50 LOC, no GPU | 1–2 hr |
| 5 | **Build `experiments/training/sft_unsloth.py`** | step 4 done | Phase 1 LoRA-SFT runner, ~150 LOC | 4–6 hr |
| 6 | **Run Phase 1 on A100-40GB** | step 5 done | LoRA adapter at `/workspace/gemma4_rl/gemma_4_lora` | autoresearch paused, ~3 hr |
| 7 | **Re-run autoresearch with `--lora`** | step 6 done | Phase 1 score deltas vs Stage A→D baseline | ~1 hr |
| 8 | **Phase 1 gate decision** | step 7 done | branch — see [Phase 1 gate](#phase-1-gate) | 15 min |
| 9 | **Phase 2 GRPO** (if step 8 PASS) | step 8 PASS | GRPO checkpoint vs Phase 1 LoRA | weekend, ~$25–50 cloud spot |
| 10 | **Phase 2 gate decision** | step 9 done | proceed to Phase 3 or revisit reward shaping | — |
| 11 | **Phase 3 self-improvement loop** | step 10 PASS | weekly/monthly cron on autoresearch | 1 day setup |

This keeps the spend at $0 incremental through step 8 (using the existing A100-40GB with autoresearch paused). Step 9 is the first cloud spend.

## Verdict-driven branches

Triggered by step 1's verdict on PR #28.

| Verdict | Implication | Action |
|---|---|---|
| **HELPS on 1+ game** | Stage D substrate is empirically validated, not just theoretically | Update [`agentic_rl_options.md`](agentic_rl_options.md) + this doc's *Recommendation* sections to claim Stage D substrate empirically. Proceed to step 4. |
| **NEUTRAL across all games** | Picked path stands as written; cognitive layer plateaued on Gemma | No doc change. Proceed to step 4. |
| **REGRESSES on 1+ game** | Inference cost from subtask injection > value, or model can't act on sub-goals | Revert `use_subtask_planning: true` for the regressing game(s). Document Stage D as a partial-fit primitive (pokemon-specific). Proceed to step 4 regardless. |

## Phase 1 gate

Triggered by step 8.

| Outcome | Action |
|---|---|
| **PASS:** ≥ +10% mean score lift on at least one game vs Stage A→D baseline | Proceed to Phase 2 GRPO (step 9) |
| **FAIL:** no game shows measurable lift | Do **not** burn Phase 2 compute. Treat as evidence the **base model is the bottleneck** — branch to the [model-swap path](#model-swap-path-fallback-if-phase-1-fails). |

If the model is the bottleneck, no amount of training compute helps — pick a bigger model first.

## Model-swap path (fallback, if Phase 1 fails)

This is option **#11** in [`agentic_rl_options.md`](agentic_rl_options.md). Test bigger / different-family / reasoning-distilled models *before* spending more on training compute. Order = cheapness-to-test.

| Tier | Candidate | Hardware | Why |
|---|---|---|---|
| **1 (cheap)** | [Kimi-VL-A3B-Thinking-2506](https://hf.co/moonshotai/Kimi-VL-A3B-Thinking-2506) | A100-40GB bf16 | 16B total / 3B active MoE — same per-token inference cost as Gemma 4 E4B but bigger expert pool. Vision-native. Predecessor (Kimi-VL-A3B-Instruct) tagged `agent` + `screenspot` — directly trained for screen-agent tasks. Tests "different family + reasoning + screen-agent training". |
| **2** | [Qwen3.6-35B-A3B-FP8](https://hf.co/Qwen/Qwen3.6-35B-A3B-FP8) | A100-40GB tight | Newest Qwen MoE — 35B total / 3B active, FP8 quant. Same per-token cost class. Tests "newer/bigger expert pool". |
| **3** | [Qwen3.6-27B-FP8](https://hf.co/Qwen/Qwen3.6-27B-FP8) | A100-40GB | Dense 27B, FP8. Tests "genuinely bigger model" without quant noise on dense capacity. |
| **4 (80GB)** | [Qwen3.6-27B at bf16](https://hf.co/Qwen/Qwen3.6-27B) | H100-80GB or A100-80GB | Cleanest no-quant test of capacity hypothesis. ~$2.50/hr cloud spot. Reserved for if tier 1–3 don't lift scores enough. |

**Pre-swap checklist** (any tier):

| Check | Action |
|---|---|
| vLLM architecture support | Bump serving venv from vllm 0.19.1 to current. Confirm `kimi_vl` (tier 1) or `qwen3_5` / `qwen3_5_moe` (tier 2/3) supported |
| Vision-capable | All four candidates are vision-native — drop-in for `supports_vision: true` |
| FP8 on Ampere | A100 emulates FP8 (faster than bf16 still, but less than H100). Plan accordingly for tier 2/3 |
| Re-baseline | Run Stage A→D sweeps on the new model to establish a comparison floor |

## Deferred — explicitly off the picked path

Things in [`agentic_rl_options.md`](agentic_rl_options.md) we are **not** pursuing in the current plan, and the trigger that would re-open them.

| Option | Why deferred | Re-open trigger |
|---|---|---|
| #4 Procedure distillation (Phase 4) | Stage D's SubtaskPlanner now occupies the "structured task decomposition at inference" slot | Phase 2 GRPO plateaus and we want decomposition baked into weights |
| #14 RLM-A (Recursive SubtaskPlanner) | Inference-time, deferred until a GRPO-tuned policy is the substrate | Phase 2 lands successfully + scores still gated by sub-goal granularity |
| #17 RLM-D (Full RLM) | Most novel + interesting but biggest engineering investment | Phase 2 lands + we want to push beyond what training alone gives |
| #18 Fastino MoE-of-specialists | Heaviest training investment | All training-driven options exhaust |
| #5 Self-rewarding DPO | Cheap stepping-stone | If Phase 1 RFT under-delivers but Phase 2 looks too costly to justify |

## How to follow live progress

```bash
# Stage D ablation sweep log (running now)
tail -f logs/stage_d_sweeps_*.log

# Verdict daemon log (5-min poll cadence)
tail -f logs/stage_d_verdict_*.log

# vLLM Gemma server log
tail -f logs/vllm_gemma_*.log
```

The verdict daemon will post a comment on PR #28 once both sweeps reach 2 iters (≈ 1 hour from launch); see [Verdict-driven branches](#verdict-driven-branches) for what to do with each outcome.
