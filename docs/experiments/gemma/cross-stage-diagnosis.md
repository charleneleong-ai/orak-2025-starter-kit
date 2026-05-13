# Cross-Stage Diagnosis: The Pokemon Milestone-4 Plateau

**Status:** complete (2026-05-13)  •  **Model:** gemma-4-26B-A4B-it-AWQ-4bit on vLLM, 300 steps per run

## The puzzle

PR #31 established Stage A→D ablation baselines. On pokemon, Stage D (full stack) hit **57.14%** (4/7 milestones, n=1). Across the next six architectural interventions — three at the action layer, two at the procedure layer, one self-reflection — the pokemon score stayed pinned to **57.14%** or worse:

| Stage | PR | Intervention | Pokemon n=1 score | Banked milestones |
|---|---|---|---|---|
| D pure | #31 | Full Stage D stack | 57.14% | 4/7 |
| D + reflect v3 | #62/#64 | Per-step LLM critique injection | 57.14% | 4/7 |
| D + reflect (600 steps) | #64 follow-up #3 | 2× step budget | 57.14% (461/600 frames stuck at milestone 4) | 4/7 |
| **E**: LangGraph + verify | #66 | Reflexion-style action verification (91% revision rate) | 57.14% | 4/7 |
| **F**: plan-do-check | #67 | ToolGateValidator + LLMPlanValidator + retry | **28.57%** | 2/7 — validator over-rejected tactically correct actions |
| **B'**: no procedures (n=3) | #69 | `use_procedure_layer=False`; planner+vmem+reflect ON | 42.86% ± 14.29pp | mean below Stage D, **range 28.57–57.14** |
| **G**: procedure-escape (n=3) | #70 (closed) | failure-streak retire K=5 + force-LLM-on-stuck N=50 | 47.62% ± 16.49pp | scores `[57.14, 57.14, 28.57]` |
| **H**: Qwen 3.5 35B-A3B-Int4 (n=3) | #71 | swap model lineage (Gemma 4 → Qwen 3.5 MoE, same MACLA Stage D stack) | 47.62% ± 16.49pp | scores `[57.14, 57.14, 28.57]` — **identical numbers to Stage G** |

Three independent action-layer interventions (D + reflect, E, F), one self-reflection extension (v3), two procedure-layer experiments (B', G), and one cross-model swap (H) — none lifted the pokemon ceiling past 4/7 milestones at any point. **The Stage H Qwen result matches the Gemma plateau on 2/3 iters and collapses to 28.57% on iter 3 — bimodal at exactly the same scores as Gemma's procedure-escape Stage G.**

## What the experiments ruled out

The structure of the failures is more informative than the failures themselves.

### Action-layer interventions don't help (Stages D+reflect, E, F, 600-step)
- Stage E's `verify_action` rewrote 91% of fallback actions — and still hit 57.14%. Rewriting individual actions doesn't lift the ceiling.
- Stage F's validator-gated planning *regressed* (28.57%) because the validator over-rejected legitimate actions (warps, signs during stuck-recovery).
- The 600-step run spent 461/600 frames at milestone 4. Extra step budget ≠ progress when the agent can't reason past the boundary.

### Self-reflection doesn't lift the ceiling (Stage D + reflect v3)
- Both per-step (`every=10`) and sparser (`every=30`) reflection schedules tie Stage D within noise on pokemon.
- Cross-game retro found self-reflection produces *game-aware* critiques ("Goomba at (189, 47), stop jumping") and aids stuck-state dialog recovery on pokemon — but doesn't unblock the milestone-4 transition.

### Procedures are net-positive (Stage B' cross-game n=3, PR #69)
- Removing the procedure cache hurts every game:
  - pokemon: −14.29 pp (42.86% vs 57.14%, σ=14.29pp)
  - mario: −7.72 pp (27.49% vs 35.21%, σ=0.00pp — three identical seeds)
  - 2048: −3.03 pp (60.61% vs 63.64%, σ=5.25pp)
- So "just remove the cache" is **not** the fix.

### Procedure-layer escape doesn't lift it either (Stage G n=3, PR #70)
- failure-streak K=5 + force-LLM-on-stuck N=50 → 47.62% ± 16.49pp, scores `[57.14, 57.14, 28.57]`.
- Within noise of Stage B' (no procedures). Force-LLM fired 100+ times per iter and the LLM fallback itself couldn't break past milestone 4.

## The convergent diagnosis

**The pokemon milestone-4 → 5 ceiling lives in LLM reasoning at the milestone boundary, not in:**
- action validation (Stage E rules out per-step verification)
- planning retries (Stage F regressed; over-rejection)
- procedure cache contents (Stage B' rules out cache lock-in)
- procedure-layer escape (Stage G shows even unconditional LLM fallback can't progress)
- self-reflection schedule density (D+reflect v2 vs v3 ties baseline; 600-step also ties)
- step budget (600-step run stuck at milestone 4 for 461 frames)
- model lineage (Stage H Qwen 3.5 35B-A3B-Int4 hits the same plateau as Gemma 4-26B-A4B-AWQ; both bimodal at [57.14, 57.14, 28.57] on n=3)

Neither the 26B-AWQ Gemma nor the 35B-A3B-Int4 Qwen 3.5, with the current planner prompt + reasoning chain, **can consistently reason past the milestone 4 → 5 transition** under any tested action/procedure-layer scaffolding. Cross-model evidence is bimodal rather than σ=0, but the *upper* ceiling (57.14%) is shared.

Trajectory introspection ([`scripts/introspect_trajectory.py`](../../scripts/introspect_trajectory.py) on PR #75) revealed a key structural finding:
- Stage H iter 2 spent **226/300 steps in OaksLab** and still banked **4/7 milestones** → milestones 1-4 are all **in-town actions** (starter, Pokedex, Mom dialogue, etc.)
- Milestone 5+ requires **leaving Pallet Town** → Viridian → Forest → Pewter Gym (Brock fight). None of the tested scaffolds reliably escape Pallet Town within 300 steps.
- Self-reflection at step 149 of every iter says verbatim *"You are stuck in a movement loop, no score gain"* — the agent **diagnoses correctly but the action layer doesn't change strategy**.

## What the durable artefacts are

The code that survives this session and ships to master:
- **PR #69**: `LocalConfig.use_procedure_layer` toggle — a useful ablation knob, defaults preserve Stage D. Cross-game Stage B' results banked under `experiments/no_procedures/gemma_26b/results.jsonl`.
- **PR #64 (in-flight)**: per-game self-reflection adapter recommendations (`RECOMMENDED_USE_SELF_REFLECTION` + `RECOMMENDED_REFLECTION_EVERY` constants on `agents/{pokemon_red,super_mario,twenty_fourty_eight}/game_adapter.py`) + `UnifiedMaclaAgent` precedence (YAML > adapter > False). Even though self-reflection doesn't lift the pokemon ceiling, it improves cross-game critique quality and is a no-harm opt-in.

Closed PRs (with verdicts preserved in PR comments):
- **#66** Stage E LangGraph + verify_action — same ceiling, dead code
- **#67** Stage F plan-do-check — regressed (28.57%)
- **#68** procedure-escape (superseded by #70)
- **#70** procedure-escape rebased — wash, doesn't lift ceiling

## What to try next

After Stage H, the most informative remaining levers are at the framework level — additions to the cognitive harness that none of A→H tested:

1. **Stage J — Qwen3-Thinking** ([PR #76](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/76)): explicit thinking-mode reasoning budget per decision via `cyankiwi/Qwen3-30B-A3B-Thinking-2507-AWQ-4bit` + vLLM `--reasoning-parser qwen3`. Tests "does extended reasoning at the boundary lift the ceiling?" Single-variable swap vs Stage H non-thinking.
2. **Stage K — Cumulative memory** ([PR #75](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/75)): chain agent checkpoints iter→iter via existing `--load-checkpoint --prev-run-id` flags so each iter inherits the previous one's procedures + atomic + vector memory. Tests "does cross-episode learning break the plateau?"
3. **Planner-prompt overhaul at the milestone boundary** — when the agent detects "stuck at milestone N for K steps", inject a milestone-specific CoT scaffold. Deferred until J/K results are in.
4. **Subtask decomposition** at milestone N→N+1 boundaries. Also deferred.

Both Stage J and Stage K test **harness additions** rather than specific game knowledge — keeping the cross-game generalisation thesis intact.

## References

- Cross-game progress plot: `docs/experiments/gemma/plots/pr64_v3_crossgame.png`
- Per-experiment results: `experiments/{pr31_ablation_26b,cross_game_self_reflect,no_procedures,loop_escape}/gemma_26b/results.jsonl`
- Closed PR verdicts: #66, #67, #68, #70 (all on the orak-2025-starter-kit repo)
- Live ablation knobs (default-safe): `LocalConfig.use_procedure_layer` (#69), `agents/*/game_adapter.RECOMMENDED_USE_SELF_REFLECTION` (#64)
