# Training Plan — Gemma 4 E4B agentic RL on Orak

How to turn the harness + cognitive infra (PRs [#25](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/25), [#26](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/26)) into a self-improving model. Phased, cost-aware, grounded in what we already have vs what's missing.

> **Status: planning doc.** No training runs yet. Numbers below are estimates from public Unsloth/TRL benchmarks + our observed inference throughput; treat as ±50%.

## Picked path (2026-05-02)

**Cognitive substrate (shipped):** Stage A harness + Stage B/C VectorMemoryProvider + **Stage D SubtaskPlanner** — already merged via PRs #25 + #26.

**Training (this doc):** **Phase 1 → Phase 2 → Phase 3** in order, gated by signal from each step.

| Phase | What | Why picked | Gate to next |
|---|---|---|---|
| 1 | RFT — top-K trajectory filter → LoRA-SFT | Cheapest training signal; uses data we already have | Phase 1 lifts mean score ≥ +10% on at least one game |
| **2** | **GRPO with shaped rewards** | **Option #2 in [`agentic_rl_options.md`](agentic_rl_options.md). Fixes credit assignment — the bottleneck once SFT exhausts top-K signal.** | GRPO sweep beats Phase 1 LoRA on the same eval harness |
| **3** | **Self-improvement loop (recurring rollout → filter → SFT/GRPO)** | **Option #3 in `agentic_rl_options.md`. Compounds Phase 1+2 gains over time; autoresearch already runs 80% of this.** | Two consecutive loops show non-trivial uplift over the previous |

**Explicitly deferred:** Phase 4 procedure distillation (Stage D's SubtaskPlanner now occupies that role), the inference-time RLM family (#14/#17), and architectural swaps (#11 reasoning model, #18 Fastino MoE) — see [`agentic_rl_options.md`](agentic_rl_options.md) for the full menu and the branch-points that would re-open these.

**Fallback if Phase 1 fails the gate:** drop to option #11 (reasoning model swap) to test the "model too small" hypothesis before spending more on training compute.

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

### Phase 4 — Procedure distillation *(deferred)*

> **Deferred** in the 2026-05-02 pick. Stage D's [SubtaskPlanner](../../../agents/_cognitive/) now occupies the "model internalizes structured task decomposition" slot at inference time, without the SFT data-conversion overhead. Re-open this phase if Phase 2 GRPO plateaus and we want to bake procedure-style structure directly into the weights.

Original sketch (kept for reference): convert MACLA's `Procedure` objects to `(precondition_obs, action, postcondition_obs)` SFT triples, augment Phase 1 data. Goal: model internalizes procedures, MACLA selector can be ablated.

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

## Decision matrix

| Goal | GPU | Why |
|---|---|---|
| Phase 1 only, prove it works | A100-40GB you already have, pause vLLM during SFT | $0 incremental |
| Phase 1 weekly, autoresearch always-on | Add cheap RTX 4090 / L40S beside the A100 | ~$30/quarter |
| Phase 2 + ongoing | Move whole stack to H100-80GB single node | Best $/throughput; ergonomic |
| Just experimenting | Cloud H100-80GB spot for the weekend | ~$25 for one full Phase 1 + Phase 2 cycle |

## Gating signal — does Phase 1 actually move scores?

Don't upgrade GPUs until Phase 1 SFT shows a measurable lift. Concrete go/no-go:

- Run Phase 1 on the existing A100-40GB (one weekend, autoresearch paused).
- Re-run autoresearch with the LoRA adapter loaded.
- **Pass:** mean score > Stage A baseline + 10% on at least one game.
- **Fail:** treat as evidence Gemma 4 E4B is too small for this benchmark; consider upgrading the model (Gemma 8B, Qwen3 14B) before scaling training compute.

If the model is the bottleneck, no amount of training compute helps — pick a bigger model first.

## What I'd recommend doing next

In order, gated by signal at each step:

1. **Build `experiments/training/filter_top_k.py`** — 50 LOC, no GPU needed. Filters trajectories. Output stats: how many tokens, how many distinct games, score distribution.
2. **Build `experiments/training/sft_unsloth.py`** — runs the actual SFT (Phase 1 RFT). ~150 LOC.
3. **Run Phase 1 on A100-40GB** with autoresearch paused. Total cost: $0, ~3 hours of inference downtime.
4. **Re-run autoresearch with the LoRA adapter loaded** — same configs, just `--lora` flag. Compare scores.
5. **Phase 1 gating decision:**
   - **Pass** (≥ +10% on at least one game) → proceed to step 6.
   - **Fail** → fall back to option #11 (reasoning model swap) per [`agentic_rl_options.md`](agentic_rl_options.md); do **not** burn compute on Phase 2 if the base model is the bottleneck.
6. **Build `experiments/training/grpo_trl.py`** — Phase 2 GRPO trainer (TRL). ~250 LOC. Reuses the filtered top-K dataset as warm-start; reward function = `OnlineAgentEvaluator.evaluate_step` shaped reward + final-score bonus; `is_fallback=True` steps → −1.0 reward (silent fallbacks become training signal).
7. **Run Phase 2 GRPO** on H100-80GB cloud spot — one weekend, ~$25-50. Re-run autoresearch with the GRPO LoRA.
8. **Phase 2 gating decision:** GRPO checkpoint must beat Phase 1 LoRA on the eval harness. If yes → step 9. If no → review reward shaping before scaling.
9. **Wire Phase 3 self-improvement loop** — cron on top of `experiments/autoresearch.py`: rollout → filter → SFT/GRPO → redeploy. Curriculum: mario (highest historical baseline) → 2048 → pokemon.

This keeps spend at $0 incremental through step 5, ~$50 through step 7, and amortizes Phase 3's recurring cost based on the cadence (~$30/quarter weekly RFT, ~$150/quarter monthly GRPO — see cost matrix above).
