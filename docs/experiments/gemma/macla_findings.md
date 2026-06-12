# MACLA on Orak 2025 — Findings Report

A 3-day study of MACLA (Memory-Augmented Contextual Learning Agent) running
unsloth/gemma-4-E4B-it via vLLM on a single A100-40GB across three games:
super_mario, twenty_fourty_eight (2048), pokemon_red. Driven by an autoresearch
loop (`experiments/autoresearch.py`) that proposes per-game `macla_*`
hyperparameters, runs experiments, and applies triage early-kill rules.

This report records what we set out to test, what we actually saw, and where
the unified-architecture claim holds vs breaks. It is deliberately scoped to
the empirical findings — no recommendations beyond honest framings of the
options at end.

> **Update (2026-05-03, PR #28):** A cross-game Stage D ablation invalidated
> the "uniform substrate" framing. The substrate does not generalize — each
> game wants a different optimal stack. Per-game verdicts and the pokemon
> reward-hack audit are incorporated below.

## TL;DR

| Game | Best score | Where | Verdict |
|---|---|---|---|
| **super_mario** | **100.0%** (W1-1 complete) | `macla_procedure_carryover/` iter 5 | **Strong fit.** Procedures generalise; carry-over compounds 35→44→52→100. |
| **twenty_fourty_eight** | **8.40%** | `macla_procedure_carryover/` iter 5 | **Partial fit.** Lucky one-shot per sweep; ceiling around 8% across configurations. |
| **pokemon_red** | **14.29%** (1 of 7 flags) | Stage A baseline (post-fix, n=2 identical) | **Eval-harness confounded.** Original 0% readings were triple-confounded (reward hack, early-kill, obs ambiguity). Post-fix, reliably scores 14.29 but only under Stage A. |

**The unified-arch claim works within a sub-class of games — those with
repeating contexts and dense reward.** It does not hold uniformly.

**The substrate does not generalize across games** (PR #28). vmem helps
1 of 3 games, planner helps 1 of 3, and no single config wins everywhere:

| | vmem (Stage A → C) | planner (Stage C → D) | Best stage |
|---|---|---|---|
| **2048** | **+48% HELPS** | −23% REGRESSES | Stage C |
| **mario** | 0% null | **+25% HELPS** | Stage D |
| **pokemon** | ? *(not measured post-fix)* | −100% REGRESSES | Stage A |

## Sweeps run

Three full autoresearch sweeps with progressively layered architectural
changes. Each sweep is a separate per-config sub-directory under
`experiments/unified_macla/`:

| Sub-dir | What's enabled | Iters | Result |
|---|---|---|---|
| `macla_checkpoint_carry/` (3 rows) | Pre-fix scaffolding test | aborted | Discarded — pre-typo-fix attempt; kept in tree as historical artefact only. |
| `macla_procedure_carryover/` (33 rows) | Checkpoint carry-over (PR #22) — MACLA procedures persist iter→iter via `--prev-run-id` | 15 | mario **100%**, 2048 **8.40%**, pokemon **14.29%**. Patience 5/5 stop. |
| `macla_state_abstraction/` (19 rows) | + reward shaping (corner-anchor for 2048, tighter pokemon stagnation) + state abstraction (`StrategicGridExtractor` for 2048 procedure keys) — PR #23 | 9 | mario 44%, 2048 **7.04%**, pokemon **14.29%**. Patience 5/5 stop. |

## Per-game findings

### super_mario — strong fit ✅

`macla_procedure_carryover/` trajectory (iter index → score):

```
0: 35.18% KEEP
1: 44.02% KEEP
2: 51.55% KEEP
3: 40.05  DISCARD
4: 35.18  DISCARD
5: 100.0% KEEP        ← W1-1 completed
6: 38.09  DISCARD
7: 35.21  DISCARD
8: 63.02  DISCARD
```

Three monotonic KEEP iters in a row (35→44→52) followed by a regression and
then the 100% breakthrough at iter 5. Each KEEP was loading the previous
iter's MACLA checkpoint, so the 51.55% iter started with the procedures iter 1
had learned, etc.

**Why mario fits:** local visual contexts (gap, ledge, Goomba ahead) recur
across the level, so procedures learned in early iters fire later. Reward
signal is dense (x_pos progress on every step). Procedure preconditions match
a meaningful fraction of subsequent observations.

The `RegexSpatialExtractor` is a good fit for mario's observation format —
keys procedures by entity-relative-to-player tokens (e.g. `goomba_ahead_near`)
which are short, repeating, semantically meaningful.

**Stage D ablation (PR #28):** vmem gave 0% lift (Stage A 35.18 == Stage C
35.18). The full +25% lift to 43.90 comes from the planner's sub-goal
scaffolding helping the agent persist past death clusters (x=1000-1100
region). vmem is dead weight here — a `Stage D − vmem` config could match
43.90 at lower memory cost.

### twenty_fourty_eight — partial fit ⚠️

`macla_procedure_carryover/` 2048 trajectory:

```
0: 4.88% KEEP
1: 3.08  DISCARD
2: 0.56  DISCARD
3: 0.30  DISCARD
4: 1.94  DISCARD
5: 8.40% KEEP    ← best
6-11:  drift between 0.4 and 5.3% (DISCARD)
```

Two KEEP iters across 12 attempts. The 8.40% breakthrough at iter 5 is
the high-water mark across all sweeps in this study.

`macla_state_abstraction/` 2048 trajectory (PR #23, with strategic-feature
keying + corner-anchor reward):

```
0: 7.04% KEEP    ← cold-start beat PR #22's iter 1 (4.88%)
1: 3.76 DISCARD
2: 0.62 DISCARD
3: 6.78 DISCARD
4: 3.94 DISCARD
5: 1.10 DISCARD
6: 5.38 DISCARD
```

State abstraction *did* help the cold-start (iter 0 = **7.04% vs 4.88%, +44%**)
because procedures actually fire across boards under the strategic-feature
key. But the param search (autoresearch.py's `propose_next_params`) was
calibrated for the prior literal-grid keying scheme; once we changed the
keying, the search direction-finding lost the working region and never
returned. Patience tripped at 5/5 by iter 9.

**Why 2048 doesn't fully fit:**

- Boards almost never repeat literally → procedures keyed on raw context
  rarely fire on the next board → pruned for low utility.
- Strategic features help, but the autoresearch param-search is itself
  per-game tuned and didn't transfer cleanly under the new key.
- Game ends fast (most failure iters: 13-50 steps) — limited window for
  procedure learning to compound within a single iter.

The combinatorial state space (~`4^16` possible boards) is the structural
challenge. Strategic abstraction collapses it to ~hundreds of clusters but
the procedure landscape and the param search interact in ways we didn't
fully untangle in this study.

**Stage D ablation (PR #28):** vmem is the real mover here (+48%, Stage A
4.36 → Stage C 6.46). Planner regresses −23% (Stage C 6.46 → Stage D 5.00)
— inference cost overhead without compensating lift. Best config: **Stage C**
(vmem on, planner off). Verdict applied in `6db13b6`.

### pokemon_red — eval-harness confounded ⚠️ *(revised from ❌)*

> **Revision note (PR #28):** The original "doesn't fit / LLM bottleneck"
> framing was partially wrong. The 0% readings that led to that conclusion
> were triple-confounded by eval-harness bugs. Post-fix, pokemon reliably
> scores 14.29 under Stage A. The LLM-side task-decomposition bottleneck
> is real but was not the *proximate* cause of the 0% scores.

#### What was actually broken

**Root cause #1 — warp-loop reward hack.** `_reward_pokemon` in
`online_evaluator.py` gave +1.5 for every `map_name` change. Warping
`RedsHouse1f → RedsHouse2f → RedsHouse1f` via the staircase gave +1.5 each
transition. Agent learned warping = reliable reward and never explored past
the starting house. 150 of 190 steps in a typical run orbited the warp tile +
TV. Fix: track visited maps per episode; first visit → `map_discovery_bonus`
(1.5), re-visit → `repeat_visit_bonus` (default 0). Shipped in `f37765f`.

**Root cause #2 — autoresearch early-kill triage.** `TRIAGE_SCORE_PLATEAU_STEPS`
was hardcoded to 80. Pokemon's first scoring event is sparse (Stage C only
got it at ~step 100-150). Stage A and D got killed at step 80-93 before they
could prove anything. Fix: per-game thresholds
`TRIAGE_SCORE_PLATEAU_STEPS_PER_GAME = {"pokemon_red": 200}`. Shipped in
`f37765f`.

**Root cause #3 — obs-label ambiguity.** The staircase warp point and the
exit door were both labeled `WarpPoint` in observations. Agent couldn't
distinguish "go upstairs" from "leave the house". Fix: prompt-level
exit-vs-staircase heuristic (bottom-edge convention). Shipped in `ccc0be0`.

#### Post-fix results (PR #28 v6)

| Stage | Score | n | Steps | Notes |
|---|---|---|---|---|
| **A** (no substrate) | **14.29** | 2 (identical) | 168 | Reaches PalletTown reliably; 4 maps visited |
| **D** (vmem + planner) | **0.00** | 2 (identical) | 150-181 | Planner's per-step latency leaves too few steps to reach Oak's Lab |

Stage A's 168 steps barely reach the first scoring event. Stage D's planner
overhead reduces effective steps to ~150-181 in the same 30-min budget,
which isn't enough. **The planner is actively harmful** on pokemon — not
because it decomposes poorly, but because it eats the step budget a
sparse-reward game needs.

**Missing: post-fix Stage C.** The 14.29 used as Stage C in the cross-game
scoreboard is pre-fix historical data (before the warp-loop fix, early-kill
fix, and obs-ambiguity fix). Whether vmem helps or hurts pokemon under fair
conditions is unknown — a post-fix Stage C run is needed to fill that cell.

#### Retrospective — was the warp-loop always there?

~80% of all gemma-4 pokemon runs in the project's history were stuck in the
starter house. The historical 14.29 from `pokemon_check` (used as Stage C
baseline in earlier scoreboards) was n=1 of 3 — the only iter that randomly
walked south. The "below capability floor" reading in v5 was wrong; it was
obs-layer ambiguity that occasionally gave random-exploration luck.

#### What would still unblock pokemon further

The LLM-side bottleneck is real even post-fix — the agent reliably exits the
house now but still doesn't reach Oak's Lab or get a starter Pokemon.
Curriculum / subgoal decomposition at the prompt level remains the path
forward for pokemon, but that is outside the scope of this study.

## Stage D cross-game ablation (PR #28)

PR #28 ran a controlled ablation of the two substrate components (vmem,
planner) across all three games. Previously, planner was only enabled on
pokemon based on the theory-driven assumption that mario (perception/timing)
and 2048 (lookahead-search) wouldn't benefit from task decomposition. **The
assumption was never measured** — PR #28 flipped the configs and let
autoresearch produce the data.

### Cross-game scoreboard

| Game | Stage A baseline | Stage C (vmem) | Stage D (vmem + planner) |
|---|---|---|---|
| 2048 | 4.36 _(n=2)_ | **6.46** _(n=4)_ ✅ | 5.00 _(n=2)_ |
| mario | 35.18 _(n=2, 0.1% spread)_ | 35.18 _(n=2)_ | **43.90** _(n=2)_ ✅ |
| pokemon | **14.29** _(n=2, identical)_ ✅ | 14.29 _(n=1 historical, pre-fix)_ ⚠️ | 0.00 _(n=2, identical)_ |

### Per-game verdicts

- **2048 → Stage C** (vmem on, planner off). Applied in `6db13b6`.
- **mario → Stage D** (vmem on, planner on), but vmem is dead weight (A==C).
  The +25% lift is planner-only. Applied in `6db13b6`.
- **pokemon → Stage A** (no substrate). Planner's per-step latency starves
  the step budget. Config pending separate commit.

### What this means

The data didn't just disprove the theory; it disproved the **uniform
substrate** premise underneath it. There is no single (vmem, planner)
configuration that wins everywhere. vmem helps 1 of 3 games, planner helps
1 of 3, and the pokemon vmem effect is unmeasured post-fix. The correct
framing is **per-game routing**, not a shipped cross-game stack.

## The unified-arch claim — where it holds vs breaks

| Property of game | Examples | MACLA fit |
|---|---|---|
| Bounded local visual contexts that recur | mario (Goomba, ledge, gap) | Strong |
| Dense reward on continuous progress signal | mario (x_pos delta per step) | Strong |
| Strategic invariants encodable as features | 2048 (corner-anchor, chain) | Partial — needs the right state abstraction *and* the param-search retuned for it |
| Combinatorial state space, sparse reward | 2048 with 16 cells × ~10 values | Partial — state abstraction helps but doesn't solve |
| Long-horizon exploration, sparse reward, agent doesn't know task structure | pokemon (open-world RPG) | Confounded — eval-harness bugs masked signal; post-fix, Stage A works but planner hurts |

The phrase "unified architecture" describes **the agent's core** (memory
system, Bayesian selector, contrastive refinement, meta-learner). But
everything that connects that core to a specific game is bespoke:

- Per-game `macla_*` parameter bounds
- Per-game `_analyze_*` failure-pattern detectors
- Per-game state extractors (`RegexSpatialExtractor` /
  `DictFieldExtractor` / `GeometricExtractor` /
  `StrategicGridExtractor`)
- Per-game reward shapers via `RewardShaper` registry (`MarioShaper`,
  `TwentyFortyEightShaper`, `PokemonShaper`) in `online_evaluator.py`
- Per-game system prompts in `configs/<game>/agent/<type>.yaml`
- Per-game env configs (`max_steps`, `target_tile`, `success_condition`)
- Per-game game-server runners under `evaluation_utils/mcp_game_servers/<game>/`
- Per-game wandb project routing
- **Per-game substrate routing** (PR #28): which combination of vmem +
  planner to enable

A more honest phrasing: **MACLA is a unified core wrapped in per-game
adapters.** The adapters are non-trivial — they encode meaningful researcher
knowledge about each game. Adding a 4th game is real work, not a config
change. The substrate itself is another per-game adapter, not a universal
primitive.

## What the architectural fixes did and didn't do

Four substantive architecture improvements landed across PR #20, #22, #23,
#28:

| Fix | Source | What it actually moved |
|---|---|---|
| Triage thresholds + preserve-new-best guard | PR #20 / #22 | Caught mario's 77% breakthrough that an earlier triage would have killed |
| Checkpoint carry-over (`--prev-run-id`) | PR #22 | mario climbed 35→100 across 5 iters; 2048 reached 8.40% (vs 6.02% prior); pokemon flat |
| Reward shaping + strategic state abstraction | PR #23 | 2048 cold start +44% (4.88→7.04) but param search couldn't navigate the new landscape; ceiling regressed to 7.04 |
| Cross-game substrate ablation + eval-harness fixes | PR #28 | Revealed non-generalization; fixed pokemon reward hack + early-kill + obs ambiguity; per-game `RewardShaper` registry; `autoresearch.retrospective` failure-mode detectors |

The strongest result is checkpoint carry-over (PR #22) — mario went from
a one-shot 61.13% ceiling under PR #20 to a sustained climb to 100%. That
is a real demonstration that procedures can compound across iterations
when the underlying game has repeating contexts.

The state-abstraction result (PR #23) is suggestive but inconclusive: the
cold-start lift is real, but the ceiling regression suggests the
autoresearch param search itself is part of the architecture and needs
re-tuning when keying scheme changes. We did not run the additional
sweeps that would pin this down.

PR #28's main contribution is methodological: the cross-game ablation
exposed that substrate components aren't general-purpose primitives, and the
pokemon eval-harness audit uncovered three nested failure modes (reward hack,
early-kill, obs ambiguity) that had been masking signal since the project's
start. The `autoresearch.retrospective` module (v0.9.0+) now surfaces these
patterns automatically after each iter.

## Limitations of this study

- **Single GPU, sequential sweeps.** All comparisons are between
  separate full sweeps, not parallel ablations. Variance across sweeps is
  not measured.
- **Small N for best scores.** Each game's "best" is a single iter's
  result. PR #22 mario had 9 iters; PR #23 mario had only 5. Mario's
  apparent regression (100% → 44%) in PR #23 may be variance, not signal.
- **No carry-over OFF ablation** ran to completion. The cleanest A/B
  for the carry-over hypothesis was cancelled mid-plan. The PR #20 run
  effectively serves as the "no carry-over" baseline (61% mario ceiling),
  but PR #20 also lacks other PR #22 fixes (preserve-new-best triage),
  so it's a confounded comparison.
- **PR #23's autoresearch param bounds were inherited from PR #22**
  rather than re-tuned for the new state-abstraction keying scheme. This
  likely undersells state abstraction's potential.
- **Pokemon eval-harness was confounded for the entire MACLA study
  (PRs #20-#23).** The warp-loop reward hack, early-kill triage, and
  obs-label ambiguity were only discovered and fixed in PR #28. All
  pokemon readings in the MACLA sweeps above are suspect — the agent may
  have been reward-hacking rather than genuinely failing. The "doesn't
  fit" verdict for pokemon in the original TL;DR was based on confounded
  data.
- **Pokemon Stage C baseline is n=1 historical.** The 14.29 used for
  Stage C in the cross-game scoreboard comes from a single iter in
  `pokemon_check` that happened to escape the starting house pre-fix.
  Not a controlled comparison.

## Honest options going forward

The study produced enough signal to support several different next moves
without strong grounds to prefer one over another:

1. **Ship per-game routing, close the substrate question.** The cross-game
   ablation (PR #28) gives a clear verdict per game. Ship the per-game
   configs (2048 → Stage C, mario → Stage D, pokemon → Stage A) and move
   on. The substrate is a per-game adapter, not a universal primitive.
2. **Re-tune autoresearch under state abstraction.** PR #23's cold-start
   lift suggests the architecture extension works; the autoresearch
   harness just wasn't ready for it. Two more sweeps with widened
   bounds and adjusted `propose_next_params` direction-finding could
   close that loop.
3. **Address the pokemon class of game directly.** Build subgoal
   decomposition / curriculum / task-aware prompting and re-run pokemon.
   This is a different research direction (hierarchical agents, not
   memory-augmented procedural learning). The eval-harness fixes from
   PR #28 are a prerequisite — now done.
4. **Stop here, write the paper / blog post.** The findings as documented
   are sufficient for a results write-up. The honest "where it works,
   where it doesn't, why" framing — including the eval-harness confound
   narrative — is more valuable than another sweep.

No recommendation is encoded in this report.

## Reproducibility

All sweep results live under `experiments/unified_macla/<config>/results.jsonl`.
Cross-game ablation results under `experiments/<tag>/gemma/results.jsonl`
(tags: `harness_check`, `harness_check_mario`, `harness_check_pokemon_v4`,
`cognitive_check_v2`, `mario_check`, `pokemon_check`, `stage_d_ablation_2048`,
`stage_d_ablation_mario_v3`, `pokemon_check_v7`).
Trajectories and game-state logs are under `game_logs/<game>/<run_id>/`.
WandB projects: `orak-pokemon-red`, `orak-2048`, `orak-super-mario` (per-run
URLs in each `results.jsonl` row).

Reference PRs:
- [#20](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/20) — autoresearch framework + initial tuning fixes
- [#22](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/22) — checkpoint carry-over
- [#23](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/23) — reward shaping + state abstraction (still open; not merged because validation didn't break PR #22's ceiling)
- [#28](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/28) — Stage D cross-game ablation + eval-harness fixes (reward hack, early-kill, obs ambiguity, `RewardShaper` registry, `autoresearch.retrospective`)
