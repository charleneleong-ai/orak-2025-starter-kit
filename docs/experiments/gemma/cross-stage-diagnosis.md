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

Three independent action-layer interventions (D + reflect, E, F), one self-reflection extension (v3), and two procedure-layer experiments (B', G) — none lifted the pokemon ceiling past 4/7 milestones at any point.

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

The 26B-AWQ-quantized Gemma model, with the current planner prompt + reasoning chain, **cannot consistently reason past the milestone 4 → 5 transition** under any tested action/procedure-layer scaffolding.

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

The diagnosis points at the planner-prompt / chain-of-thought path, **not** the action layer or procedure layer. Candidate directions:

1. **Planner-prompt overhaul at the milestone boundary** — when the agent detects "stuck at milestone N for K steps", inject a milestone-specific CoT scaffold (e.g. "here's what milestone N+1 requires; here's the sub-goal sequence; reason step-by-step about which sub-goal is incomplete").
2. **Multi-turn reasoning checkpoint** — at suspected milestone transitions, allow the planner an extended reasoning budget (more tokens, possibly chain-of-verification or self-consistency).
3. **Different / stronger model** — the ceiling may be a 26B-AWQ capacity ceiling. Try the unquantised model, or a stronger model (Claude/GPT-class), as a ceiling check. If a stronger model lifts past 4/7, the cap is model capacity, not architecture.
4. **Subtask decomposition** at milestone N→N+1 boundaries — the current planner emits whole-task plans; an explicit "what does milestone N+1 require?" decomposition may help.

Option 3 is the cheapest experiment with the clearest signal: does the ceiling move? If yes → model capacity. If no → option 1/4 next.

## References

- Cross-game progress plot: `docs/experiments/gemma/plots/pr64_v3_crossgame.png`
- Per-experiment results: `experiments/{pr31_ablation_26b,cross_game_self_reflect,no_procedures,loop_escape}/gemma_26b/results.jsonl`
- Closed PR verdicts: #66, #67, #68, #70 (all on the orak-2025-starter-kit repo)
- Live ablation knobs (default-safe): `LocalConfig.use_procedure_layer` (#69), `agents/*/game_adapter.RECOMMENDED_USE_SELF_REFLECTION` (#64)
