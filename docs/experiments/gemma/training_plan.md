# Training Plan — Gemma 4 E4B agentic RL on Orak

How to turn the harness + cognitive infra (PRs [#25](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/25), [#26](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/26)) into a self-improving model.

> **Status (2026-05-03):** Stage D cross-game ablation complete (PR #28 merged). Per-game substrate configs locked. No training runs yet.
> See [`macla_findings.md`](macla_findings.md) for full ablation results and [`../agentic-rl-research/vlm_options.md`](../../agentic-rl-research/vlm_options.md) for current model options.

---

## Current state (updated 2026-05-03)

| Layer | Status | Where |
|---|---|---|
| Cognitive substrate (Stage A→D) | ✅ shipped | PRs [#25](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/25) + [#26](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/26) |
| Stage D ablation (mario + 2048) | ✅ complete | PR [#28](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/28) merged |
| Per-game substrate configs locked | ✅ shipped | `gemma_stage_c.yaml` (2048), `gemma.yaml` Stage D (mario), `gemma_stage_a.yaml` (pokemon) |
| `autoresearch-verdict` CLI | ✅ shipped v0.5.1 | [autoresearch#10](https://github.com/charleneleong-ai/autoresearch/pull/10) |
| Eval-harness fixes | ✅ shipped | Reward hack + early-kill + obs-ambiguity fixed (PR #28) |
| Phase 1 RFT (training) | ⏳ not started | needs `filter_top_k.py` |
| Phase 2 GRPO | ⏳ not started | gated on Phase 1 |
| Phase 3 self-improvement loop | ⏳ not started | gated on Phase 2 |

---

## Stage D results — verdicts locked

Full data in [`macla_findings.md`](macla_findings.md) and PR #28.

| Game | Stage A | Stage C (vmem) | Stage D (vmem+planner) | Verdict | Config |
|---|---|---|---|---|---|
| **2048** | 4.36 | **6.46** (+48%) | 5.00 (−23%) | Stage C wins | `gemma_stage_c.yaml` |
| **mario** | 35.18 | 35.18 (0%) | **43.90** (+25%) | Stage D wins | `gemma.yaml` |
| **pokemon** | **14.29** | 14.29 (pre-fix, n=1) | 0.00 | Stage A wins | `gemma_stage_a.yaml` |

Key insight: **no single cross-game substrate configuration wins**. vmem helps 1 of 3 games,
planner helps 1 of 3. Per-game routing is the correct framing.

**Missing data point:** pokemon Stage C post-fix. The historical 14.29 used as Stage C baseline
was collected before the warp-loop reward hack, early-kill, and obs-ambiguity fixes (PR #28).
A post-fix Stage C run (2 episodes) is needed before deciding if vmem helps pokemon.

---

## Direction — two-track approach (updated with Stage D results)

| Observation | Implication | Action |
|---|---|---|
| 2048 lifted +48% with vmem (4.36→6.46) | Cognitive layer landing; training likely to compound | 2048 is Phase 1 RFT target |
| mario solved 100% via carry-over (PR #22) | Architecture was bottleneck, not model | Do not train mario — regression risk |
| pokemon ceiling 14.29% post eval-harness fixes | Model-size or subgoal bottleneck | Test better base model before training |
| Gemma 4 26B A4B exists (85.5% tau2-bench vs 57.5% E4B) | 28-point agentic jump, 13GB AWQ fits L4 | Run `harness_check` with 26B A4B before any training run |

**Track A — training on Gemma 4 E4B:** Phase 1 RFT on 2048. Gate: +10% lift on at least one game.

**Track B — model upgrade:** Swap to Gemma 4 26B A4B (AWQ) or Qwen3-VL-8B-FP8, re-baseline all games. Gate: +10% from base model swap alone.

See [`vlm_options.md`](../../agentic-rl-research/vlm_options.md) for full model comparison, memory math, and vLLM serve commands.

Tracks converge at the gate: if either passes, continue that track. If both fail, the bottleneck is MACLA architecture — improve reward shaping or state abstraction before more training.

---

## What we have

| Piece | Status |
|---|---|
| ShareGPT trajectories | `game_logs/<game>/<run_id>/logs/trajectory_samples.jsonl` per episode |
| Per-step rewards | `RewardShaper` per game in `agents/macla/online_evaluator.py` |
| Episode scores | `evaluation_score` (0–100) in `evaluation_summary.json` |
| Failure attribution | `is_fallback` flag per step |
| Eval harness | `experiments/autoresearch.py` (per-game triage thresholds fixed PR #28) |
| GRPO + SFT trainers | `UnslothGRPOTrainer`, `UnslothSFTTrainer`, `UnslothDPOTrainer` compiled in `gemma4_rl/unsloth_compiled_cache/` |
| `train.py` | Full GRPO CLI with Unsloth + TRL at `/workspace/gemma4_rl/train.py` |
| LoRA adapter dir | `/workspace/gemma4_rl/gemma_4_lora/` — empty placeholder ready |
| LoRA inference | vLLM `--enable-lora` supported but not yet in serving scripts |

## What's missing (priority order)

| Piece | Effort | Blocks |
|---|---|---|
| Post-fix pokemon Stage C baseline | 0 LOC, 2 episodes | pokemon DPO/KTO decision |
| `experiments/training/filter_top_k.py` | ~50 LOC, no GPU | Phase 1 SFT, DPO, KTO |
| `experiments/training/game_reward_fns.py` | ~100 LOC | Phase 2 GRPO (wraps `RewardShaper` → GRPO reward_funcs) |
| `serving/gemma_serve.sh` with `--enable-lora` | ~10 LOC | Serving LoRA adapters in dev |

See [`agentic_rl_pipeline.md`](../../agentic-rl-research/agentic_rl_pipeline.md) for full training loop and reward function wiring.

---

## Phases

### Phase 1 — Rejection-sampling fine-tune (RFT)

Filter `trajectory_samples.jsonl` top-K% by `final_score`. LoRA-SFT Gemma 4 E4B via Unsloth.
Target game: **2048** (strongest signal from Stage C vmem lift).

Hyperparameters: LoRA rank 16–32, alpha 2×rank, seq_len 4096, batch 2 + grad_accum 8, LR 2e-4 cosine, 3 epochs, AdamW 8-bit.
Data: ~9K filtered steps × 4K tokens ≈ **36M tokens** to train on.

### Phase 2 — GRPO with shaped rewards

GRPO on 2048 with verifiable reward (score + corner-anchor + chain bonuses + fallback penalty −1.0).
See [`agentic_rl_pipeline.md`](../../agentic-rl-research/agentic_rl_pipeline.md) for exact reward function wiring.
Group size N=8, reference = base E4B frozen, policy = Phase 1 LoRA. Cost ~10–20× SFT.

### Phase 3 — Self-improvement loop

Weekly cron: rollout → filter → SFT → redeploy. Curriculum: 2048 → pokemon → (mario only if regression).

### Phase 4 — Procedure distillation (deferred)

Convert MACLA `Procedure` objects to SFT triples. Gated on Phase 2 success.

---

## GPU options + cost

Throughput assumes Gemma 4 E4B, BF16, Unsloth LoRA rank 16.

| GPU | VRAM | SFT tok/s | Phase 1 (36M × 3ep) | GRPO viable? | $/hr | Phase 1 cost |
|---|---|---|---|---|---|---|
| A100 40GB (on-hand) | 40GB | ~12K | ~2.5hr | tight (N=4) | $0 | $0 |
| A100 80GB | 80GB | ~14K | ~2hr | yes (N=8–16) | $1.60 | ~$2.40 |
| H100 80GB FP8 | 80GB | ~30K | ~1hr | yes (N=16+) | $2.50 | ~$1.90 |

GRPO multiplier: ~10–20× SFT wall-clock. H100 cheapest per cycle at scale.

Operational rule: monthly → pause vLLM, train, redeploy ($0 extra); weekly → second GPU; daily → 80GB single node.

---

## Model upgrade path

Run this **before** committing Phase 1 compute. See [`vlm_options.md`](../../agentic-rl-research/vlm_options.md).

| Tier | Model | VRAM | tau2-bench | Recommended for |
|---|---|---|---|---|
| 1 (current) | Gemma 4 E4B | ~16GB | 57.5% | All baselines indexed here |
| **2 (next)** | **Gemma 4 26B A4B (AWQ)** | **~13GB** | **85.5%** | **Run harness_check first** |
| 3 | Qwen3-VL-8B-FP8 | ~8GB | GUI-focused | Most context headroom on L4 |
| 4 | Qwen3-VL-32B (AWQ) | ~17GB | — | Best pure VLM quality |
| 5 (future) | Qwen3.6-35B-A3B (4-bit) | ~17.5GB | MMMU 81.7 | Wait for vLLM vision API confirm |

---

## Next steps (updated 2026-05-03)

| # | Step | Status | Effort |
|---|---|---|---|
| ~~1~~ | ~~Stage D verdict~~ | ✅ done | auto |
| ~~2~~ | ~~Decide Stage D disposition~~ | ✅ done | PR #28 |
| ~~3~~ | ~~Merge PR #28~~ | ✅ done | — |
| **4** | **Run Track B: `harness_check` with Gemma 4 26B A4B AWQ** | ⏳ next | ~1hr, $0 |
| 5 | Post-fix pokemon Stage C baseline (2 episodes) | ⏳ | ~30min |
| 6 | Build `filter_top_k.py` | ⏳ gated on baseline | 1–2hr |
| 7 | Build `game_reward_fns.py` | ⏳ gated on baseline | ~2hr |
| 8 | Run Phase 1 RFT on A100-40GB | ⏳ gated on 6 | ~3hr, $0 |
| 9 | Re-run autoresearch with LoRA | ⏳ gated on 8 | ~1hr |
| 10 | Phase 1 gate decision (+10% rule) | ⏳ | 15min |
| 11 | Phase 2 GRPO (if step 10 PASS) | ⏳ | ~$25–50 cloud spot |
| 12 | Phase 3 self-improvement loop | ⏳ | 1 day setup |

**Step 4 is the cheapest next action:** just swap the model in vLLM, run existing evals.
Costs $0 and determines whether training is the right lever.

## Deferred

| Option | Re-open trigger |
|---|---|
| Procedure distillation (Phase 4) | Phase 2 GRPO plateaus |
| DPO/KTO on pokemon | Post-fix Stage C baseline exists |
| RLHF value model | Phase 2 lands |
| Qwen3.6-35B-A3B upgrade | Community confirms vLLM image input |
