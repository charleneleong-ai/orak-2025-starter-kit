# Next Steps — Gemma 4 E4B agentic RL roadmap

Snapshot of where we are and what's next, after PR #26 (Stage A→D cognitive substrate) and the in-flight Stage D cross-game ablation. Companion to [`training_plan.md`](training_plan.md) (the picked path) and [`agentic_rl_options.md`](agentic_rl_options.md) (the wider menu).

Last updated: **2026-05-02**.

## Current state

| Layer | Status | Where |
|---|---|---|
| Cognitive substrate (Stage A→D) | shipped | merged via [#25](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/25) + [#26](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/26) |
| Stage D enabled on pokemon | shipped | `configs/pokemon_red/agent/gemma.yaml` |
| Stage D enabled on mario + 2048 | **PR open** | [#28](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/28) — ablation sweep running |
| `autoresearch-verdict` CLI | shipped (v0.5.1) | [autoresearch#10](https://github.com/charleneleong-ai/autoresearch/pull/10) |
| Verdict spec for ablation | shipped | [`experiments/stage_d_verdict.yaml`](../../../experiments/stage_d_verdict.yaml) |
| Stage D ablation sweeps | **running** | `logs/stage_d_sweeps_20260502T091045Z.log` |
| Verdict daemon | **polling** | `logs/stage_d_verdict_20260502T091058Z.log` (posts to PR #28 once both sweeps reach 2 iters) |
| Phase 1 RFT (training) | not started | needs `filter_top_k.py` + `sft_unsloth.py` |
| Phase 2 GRPO | not started | gated on Phase 1 outcome |
| Phase 3 self-improvement loop | not started | gated on Phase 2 outcome |

## Next steps — execution order

| # | Step | Trigger | Outcome | Effort |
|---|---|---|---|---|
| 1 | **Wait for Stage D verdict comment** on PR #28 | both sweeps reach 2 iters (≈1 hr wall-clock) | Verdict daemon auto-posts a HELPS / NEUTRAL / REGRESSES table | none — auto |
| 2 | **Decide Stage D disposition based on verdict** | step 1 lands | branch → see [Verdict-driven branches](#verdict-driven-branches) below | 30 min docs |
| 3 | **Merge PR #28** | step 2 done | Stage D config locked in (or reverted on regressing games) | 5 min |
| 4 | **Build `experiments/training/filter_top_k.py`** | regardless of verdict | Top-K trajectory filter, ~50 LOC, no GPU | 1–2 hr |
| 5 | **Build `experiments/training/sft_unsloth.py`** | step 4 done | Phase 1 LoRA-SFT runner, ~150 LOC | 4–6 hr |
| 6 | **Run Phase 1 on A100-40GB** | step 5 done | LoRA adapter at `/workspace/gemma4_rl/gemma_4_lora` | autoresearch paused, ~3 hr |
| 7 | **Re-run autoresearch with `--lora` flag** | step 6 done | Phase 1 score deltas vs Stage A→D baseline | ~1 hr |
| 8 | **Phase 1 gate decision** | step 7 done | branch → see [Phase 1 gate](#phase-1-gate) below | 15 min |
| 9 | **Phase 2 GRPO** (if step 8 PASS) | step 8 PASS | GRPO checkpoint vs Phase 1 LoRA | weekend, ~$25–50 cloud spot |
| 10 | **Phase 2 gate decision** | step 9 done | proceed to Phase 3 or revisit reward shaping | — |
| 11 | **Phase 3 self-improvement loop** | step 10 PASS | weekly/monthly cron on autoresearch | 1 day setup |

## Verdict-driven branches

Triggered by step 1's verdict on PR #28.

| Verdict | Implication | Action |
|---|---|---|
| **HELPS on 1+ game** | Stage D substrate is empirically validated, not just theoretically | Update [`agentic_rl_options.md`](agentic_rl_options.md) + [`training_plan.md`](training_plan.md) Recommendation sections to claim Stage D substrate empirically. Proceed to Phase 1 RFT (step 4). |
| **NEUTRAL across all games** | Picked path stands as written; cognitive layer plateaued on Gemma | No doc change. Proceed to Phase 1 RFT (step 4). |
| **REGRESSES on 1+ game** | Inference cost from subtask injection > value, or model can't act on sub-goals | Revert `use_subtask_planning: true` for the regressing game(s). Document Stage D as a partial-fit primitive (pokemon-specific). Proceed to Phase 1 RFT regardless. |

## Phase 1 gate

Triggered by step 8 (Phase 1 LoRA evaluation).

| Outcome | Action |
|---|---|
| **PASS**: ≥ +10% mean score lift on at least one game vs Stage A→D baseline | Proceed to Phase 2 GRPO (step 9) |
| **FAIL**: no game shows measurable lift | Do **not** burn Phase 2 compute. Treat as evidence that the **base model is the bottleneck** — branch to model-swap path (below). |

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
tail -f /workspace/orak-2025-starter-kit/logs/stage_d_sweeps_20260502T091045Z.log

# Verdict daemon log (5-min poll cadence)
tail -f /workspace/orak-2025-starter-kit/logs/stage_d_verdict_20260502T091058Z.log

# vLLM Gemma server log
tail -f /workspace/orak-2025-starter-kit/logs/vllm_gemma_20260502T090753Z.log
```

The verdict daemon will post a comment on PR #28 once both sweeps reach 2 iters (≈ 1 hour from launch); see [Verdict-driven branches](#verdict-driven-branches) for what to do with each outcome.
