# Cross-Stage Diagnosis: The Pokemon Milestone-4 Plateau

**Status:** updated 2026-05-15 (post-asm-fix re-validation) • original diagnosis 2026-05-13  •  **Models:** gemma-4-26B-A4B-it-AWQ-4bit and Qwen3.5-35B-A3B-Int4 on vLLM, 300 steps per run

> **Headline (2026-05-15):** the 57.14% ceiling is real and lives at **M5: `'Viridian' in map_name`** ([`pokemon_red_env.py:297`](../../../evaluation_utils/mcp_game_servers/pokemon_red/game/pokemon_red_env.py#L297-L298)). Six independent runs across Gemma and Qwen post-asm-fix all scored 57.14% with **zero variance**. The bimodal `[57.14, 57.14, 28.57]` distributions observed in pre-fix Stages G/H/B' were a **placeholder-reasoning artifact** during the scripted M1-M4 phase, not a real ceiling signal. See [PR #81](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/81) for the rerun.

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
| **H**: Qwen 3.5 35B-A3B-Int4 (n=3, pre-asm-fix) | #71 (closed, superseded by #81) | swap model lineage Gemma 4 → Qwen 3.5 MoE | 47.62% ± 16.49pp | scores `[57.14, 57.14, 28.57]` — identical to Stage G |
| **D-rerun**: Gemma 26B (n=3, **post-asm-fix**) | #81 | re-run Stage D after `pokered/data/maps/objects/*.asm` populated (#80) | **57.14% ± 0.00pp** | scores `[57.14, 57.14, 57.14]` — zero variance |
| **H-rerun**: Qwen 3.5 35B-A3B-Int4 (n=3, **post-asm-fix**) | #81 | re-run Stage H under the asm fix | **57.14% ± 0.00pp** | scores `[57.14, 57.14, 57.14]` — zero variance |

Three independent action-layer interventions (D + reflect, E, F), one self-reflection extension (v3), two procedure-layer experiments (B', G), and one cross-model swap (H) all converged on the same 57.14% upper bound. The **post-asm-fix reruns (D + H, n=6 total across two models)** then nailed it down: identical 57.14%, zero variance. The pre-fix `28.57%` outliers in B', G, H were placeholder-anchored procedures dead-ending the agent during the scripted M1-M4 phase, not a ceiling signal — under the asm fix that variance collapses.

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

### Model lineage doesn't lift it either (Stage H Qwen 3.5 35B-A3B-Int4 n=3, PR #71, pre-asm-fix; PR #81 post-fix)
- Pre-asm-fix Stage H scored `[57.14, 57.14, 28.57]` = 47.62% ± 16.49pp — identical bimodal to Gemma's Stage G.
- Post-asm-fix Stage H (PR #81) collapses to `[57.14, 57.14, 57.14]` = 57.14% ± 0.00pp. Same upper bound as Gemma, now with zero variance across models.

## The convergent diagnosis (post-asm-fix, 2026-05-15)

**The pokemon ceiling is the M5 milestone gate, not a reasoning/model/architecture limitation.**

Reward function ([`pokemon_red_env.py:277-304`](../../../evaluation_utils/mcp_game_servers/pokemon_red/game/pokemon_red_env.py#L277-L304)):

| M | Trigger | Where it happens |
|---|---|---|
| M1 | leave RedsHouse | exit RedsHouse2f → RedsHouse1f → PalletTown |
| M2 | `SPRITE_OAK in map_screen` OR `OaksLab in map_name` | scripted cutscene |
| M3 | `'Name' in your_party` | scripted, name-the-starter screen |
| M4 | transition out of `Battle` state | rival battle in OaksLab, scripted enemy |
| **M5** | **`'Viridian' in map_name`** | **walk north through Route1 to Viridian City** |
| M6 | `"OAK's PARCEL" in inventory` | shopkeeper in Viridian Mart |
| M7 | parcel no longer in inventory | walk back south to OaksLab |

M1-M4 are essentially **scripted**: cutscene fires the moment the agent leaves RedsHouse, OakSpeech runs, starter is given, rival battle is forced. M5 is the **first milestone that demands real navigation** — cross Route1 from south to north (~25 tiles), avoid getting stuck in tall-grass battles, take the top exit.

**Post-asm-fix trajectory introspection (6 runs)**:
- **0/6 runs ever set foot in Viridian City.** Every run ends in Route1, PalletTown, or OaksLab.
- Late-game pathology (steps 200-300) is identical across Gemma and Qwen: 16-59 `move_to()` calls patrolling Route1 grid coordinates, 10-20 `select_move_in_battle()` losing to wild Youngsters, 10-17 `interact_with_object()` re-reading the same signs/NPCs.
- M4 locks in by step ~70-260. The remaining 40-230 steps are wasted in Route1 loops without finding the north exit tile.

Therefore **the ceiling lives at the navigation layer, not the reasoning layer**. None of the Stage A-H interventions touched navigation — they all operated on planner prompts, action validation, procedure caches, self-reflection schedules, or model lineage. The asm fix replaces `OBJ_n_n` placeholders with real sprite names — that recovers the variance but doesn't help cross a map transition tile.

The pre-asm-fix diagnosis ("LLM reasoning at the milestone boundary") was off by one layer. Self-reflection at step 149 of every iter does say *"You are stuck in a movement loop, no score gain"* — the agent diagnoses the symptom correctly. The reason it can't act on the diagnosis is that the observation doesn't surface the exit tile, the map graph, or the "you have never been to Viridian" fact.

## What the durable artefacts are

The code that survives this session and ships to master:
- **PR #69**: `LocalConfig.use_procedure_layer` toggle — a useful ablation knob, defaults preserve Stage D. Cross-game Stage B' results banked under `experiments/no_procedures/gemma_26b/results.jsonl`.
- **PR #64 (merged 2026-05-13)**: per-game self-reflection adapter recommendations (`RECOMMENDED_USE_SELF_REFLECTION` + `RECOMMENDED_REFLECTION_EVERY` constants on `agents/{pokemon_red,super_mario,twenty_fourty_eight}/game_adapter.py`) + `UnifiedMaclaAgent` precedence (YAML > adapter > False). Even though self-reflection doesn't lift the pokemon ceiling, it improves cross-game critique quality and is a no-harm opt-in.

Closed PRs (with verdicts preserved in PR comments):
- **#66** Stage E LangGraph + verify_action — same ceiling, dead code
- **#67** Stage F plan-do-check — regressed (28.57%)
- **#68** procedure-escape (superseded by #70)
- **#70** procedure-escape rebased — wash, doesn't lift ceiling
- **#71** Stage H Qwen 3.5 35B-A3B-Int4 ceiling-check — pre-fix bimodal `[57.14, 57.14, 28.57]`; superseded by [PR #81](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/81) post-asm-fix rerun (`[57.14, 57.14, 57.14]`)

Merged PRs (post-asm-fix corrections):
- **#80** hard-fail when pokered/ asm files missing — removed the `OBJ_n_n` placeholder source
- **#81** Stage D + Stage H n=3 reruns under asm fix — established 57.14% × 6 zero-variance ceiling, identified M5 navigation gate as root cause

## What to try next

The diagnosis now points at the **navigation layer**, not reasoning. The most informative remaining levers:

1. **Map-exit callouts in the observation** — when the agent is on a map adjacent to an unvisited map, surface the exit tile coordinates in every observation (`"You can leave Route1 to the north at (10, 0). You have not yet visited Viridian City."`). This is the cheapest intervention.
2. **Visited-maps sticky note** — append a running list of `(map_name, first_step, last_step)` tuples to the observation so the agent has explicit evidence of what it has and hasn't explored.
3. **Mini-map / map-graph view** — render a compact representation of the connected maps reachable from the current location, with unvisited maps highlighted.
4. **Subtask injection at stuck-state** — when self-reflection emits *"stuck in movement loop"*, inject an explicit subtask: `"goal: leave the current map. Try moving to each edge."` (current self-reflection produces the diagnosis but no behavioural change.)
5. **Stage K — Cumulative memory** ([PR #75](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/75)): chain agent checkpoints iter→iter via existing `--load-checkpoint --prev-run-id`. *Only helps if iter 1 stumbles into Viridian once and captures a navigation procedure; otherwise inherited Route1-loop procedures will pollute downstream iters (pre-fix Stage K showed `-14.28pp` learning_delta, confirming this risk).*

**Update 2026-05-24:** Options 1-2 were partially implemented as Stage Q exit-tile coords + Stage R subgoal hints ([PRs #92, #97](https://github.com/charleneleong-ai/orak-2025-starter-kit/pulls)); Option 5 (Stage K cumulative memory) merged as [PR #75](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/75) on 2026-05-15 with a FLAT-but-no-compounding verdict — see Stage L ([PR #85](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/85)) for the map-aware procedure key follow-up, and Stage S ([PRs #103, #104](https://github.com/charleneleong-ai/orak-2025-starter-kit/pulls)) for `MilestoneSpec.requires_location` auto-bridging that lifted pokemon 4/7 → 6/7 baseline. The per-game scaffold work is now superseded by the cross-game MVA roadmap — see [`docs/generalized-agent-mva.md`](../../generalized-agent-mva.md).

## References

- Cross-game progress plot: `docs/experiments/gemma/plots/pr64_v3_crossgame.png`
- Per-experiment results: `experiments/{pr31_ablation_26b,cross_game_self_reflect,no_procedures,loop_escape}/gemma_26b/results.jsonl`
- Closed PR verdicts: #66, #67, #68, #70 (all on the orak-2025-starter-kit repo)
- Live ablation knobs (default-safe): `LocalConfig.use_procedure_layer` (#69), `agents/*/game_adapter.RECOMMENDED_USE_SELF_REFLECTION` (#64)
