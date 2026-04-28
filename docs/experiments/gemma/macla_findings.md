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

## TL;DR

| Game | Best score | Where | Verdict |
|---|---|---|---|
| **super_mario** | **100.0%** (W1-1 complete) | `macla_procedure_carryover/` iter 5 | **Strong fit.** Procedures generalise; carry-over compounds 35→44→52→100. |
| **twenty_fourty_eight** | **8.40%** | `macla_procedure_carryover/` iter 5 | **Partial fit.** Lucky one-shot per sweep; ceiling around 8% across configurations. |
| **pokemon_red** | **14.29%** (1 of 7 flags) | All sweeps, iter 1 (cold) | **Doesn't fit.** Bottleneck is LLM-side task decomposition, not MACLA's procedure-learning. |

**The unified-arch claim works within a sub-class of games — those with
repeating contexts and dense reward.** It does not hold uniformly.

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

### pokemon_red — doesn't fit ❌

Across all three sweeps, pokemon's best score is **14.29% — exactly 1 flag
of 7** — and never improves past iter 1 / iter 5 / iter 11 (the iters that
happen to break out of the starting building). Every other iter sits at 0%.

The diagnostic action distribution from the only successful 14.29% iter
(`game_logs/pokemon_red/20260427_224956/`):

```
Steps 1-40:  oscillates RedsHouse1f ↔ RedsHouse2f via warp_with_warp_point
             mashes interact_with_object SIGN_REDSHOUSE1F_TV
Step 41:     finally exits to PalletTown via warp_with_warp_point  ← +1 flag
Steps 41-157: wanders PalletTown, interacts with SIGN_PALLETTOWN sign
              never reaches Route 1, never visits Oak's lab,
              never gets a starter Pokemon
```

Action histogram for that iter:

```
19× interact_with_object SIGN_PALLETTOWN_PLAYERSHOUSE_SIGN
12× move_to (3, 6)
11× warp_with_warp_point (7, 1)
10× move_to (4, 6)
 8× use_tool(move_to (3, 5))
 8× interact SIGN_REDSHOUSE1F_TV
```

**Why pokemon doesn't fit:**

The bottleneck is **upstream of MACLA**. The LLM-side agent doesn't know
the canonical Pokemon Red opening sequence (leave house → Oak's lab →
choose starter → Route 1 → Brock). It emits plausible-looking but useless
actions: "interact with TV", "warp through doorway", "move one square".
MACLA can only learn procedures from actions the LLM emits, and the LLM
emits sign-mashing.

We confirmed this is not fixable by anything in MACLA's scope:

- ❌ Checkpoint carry-over (PR #22): procedures persisted across iters,
  but the procedures were "in RedsHouse1f, talk to TV". Persistence of
  bad procedures is worthless.
- ❌ Reward shaping (PR #23 reward changes): tighter pokemon stagnation
  (-0.2→-0.4 after 3 turns). Agent still doesn't know which direction to
  walk; penalty doesn't generate the missing knowledge.
- ❌ State abstraction (PR #23 extractor): only changed 2048's key
  scheme. Pokemon used `DictFieldExtractor` already; the issue isn't the
  context key, it's that the LLM never generates a useful action
  sequence to abstract.

**What would unblock pokemon** is curriculum / subgoal decomposition at
the prompt level: explicit "Step 1: leave the player's house. Step 2:
visit Professor Oak…". That is a different research direction (task
decomposition / hierarchical RL) and outside the scope of this study.

## The unified-arch claim — where it holds vs breaks

| Property of game | Examples | MACLA fit |
|---|---|---|
| Bounded local visual contexts that recur | mario (Goomba, ledge, gap) | Strong |
| Dense reward on continuous progress signal | mario (x_pos delta per step) | Strong |
| Strategic invariants encodable as features | 2048 (corner-anchor, chain) | Partial — needs the right state abstraction *and* the param-search retuned for it |
| Combinatorial state space, sparse reward | 2048 with 16 cells × ~10 values | Partial — state abstraction helps but doesn't solve |
| Long-horizon exploration, sparse reward, agent doesn't know task structure | pokemon (open-world RPG) | Doesn't fit — bottleneck is LLM-side |

The phrase "unified architecture" describes **the agent's core** (memory
system, Bayesian selector, contrastive refinement, meta-learner). But
everything that connects that core to a specific game is bespoke:

- Per-game `macla_*` parameter bounds
- Per-game `_analyze_*` failure-pattern detectors
- Per-game state extractors (`RegexSpatialExtractor` /
  `DictFieldExtractor` / `GeometricExtractor` /
  `StrategicGridExtractor`)
- Per-game reward shapers in `online_evaluator.py`
- Per-game system prompts in `configs/<game>/agent/<type>.yaml`
- Per-game env configs (`max_steps`, `target_tile`, `success_condition`)
- Per-game game-server runners under `evaluation_utils/mcp_game_servers/<game>/`
- Per-game wandb project routing

A more honest phrasing: **MACLA is a unified core wrapped in per-game
adapters.** The adapters are non-trivial — they encode meaningful researcher
knowledge about each game. Adding a 4th game is real work, not a config
change.

## What the architectural fixes did and didn't do

Three substantive architecture improvements landed across PR #20, #22, #23:

| Fix | Source | What it actually moved |
|---|---|---|
| Triage thresholds + preserve-new-best guard | PR #20 / #22 | Caught mario's 77% breakthrough that an earlier triage would have killed |
| Checkpoint carry-over (`--prev-run-id`) | PR #22 | mario climbed 35→100 across 5 iters; 2048 reached 8.40% (vs 6.02% prior); pokemon flat |
| Reward shaping + strategic state abstraction | PR #23 | 2048 cold start +44% (4.88→7.04) but param search couldn't navigate the new landscape; ceiling regressed to 7.04 |

The strongest result is checkpoint carry-over (PR #22) — mario went from
a one-shot 61.13% ceiling under PR #20 to a sustained climb to 100%. That
is a real demonstration that procedures can compound across iterations
when the underlying game has repeating contexts.

The state-abstraction result (PR #23) is suggestive but inconclusive: the
cold-start lift is real, but the ceiling regression suggests the
autoresearch param search itself is part of the architecture and needs
re-tuning when keying scheme changes. We did not run the additional
sweeps that would pin this down.

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
- **Pokemon trajectories were inspected from one successful iter
  (run_id 20260427_224956) and one failed iter (20260428_000044).**
  Conclusions about pokemon's LLM-side bottleneck are based on the
  pattern of action histograms but a more rigorous test would replay
  multiple failed iters and trace the LLM reasoning step by step.

## Honest options going forward

The study produced enough signal to support several different next moves
without strong grounds to prefer one over another:

1. **Rename the claim, ship what works.** Reframe MACLA as
   *unified-core-with-game-adapters that fits action-platformers*. Ship the
   PR #22 carry-over work as the demonstrated win. Move on to other game
   classes that fit (other Atari-style games, perhaps).
2. **Re-tune autoresearch under state abstraction.** PR #23's cold-start
   lift suggests the architecture extension works; the autoresearch
   harness just wasn't ready for it. Two more sweeps with widened
   bounds and adjusted `propose_next_params` direction-finding could
   close that loop.
3. **Address the pokemon class of game directly.** Build subgoal
   decomposition / curriculum / task-aware prompting and re-run pokemon.
   This is a different research direction (hierarchical agents, not
   memory-augmented procedural learning).
4. **Stop here, write the paper / blog post.** The findings as documented
   are sufficient for a results write-up. The honest "where it works,
   where it doesn't, why" framing is more valuable than another sweep.

No recommendation is encoded in this report.

## Reproducibility

All sweep results live under `experiments/unified_macla/<config>/results.jsonl`.
Trajectories and game-state logs are under `game_logs/<game>/<run_id>/`.
WandB projects: `orak-pokemon-red`, `orak-2048`, `orak-super-mario` (per-run
URLs in each `results.jsonl` row).

Reference PRs:
- [#20](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/20) — autoresearch framework + initial tuning fixes
- [#22](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/22) — checkpoint carry-over
- [#23](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/23) — reward shaping + state abstraction (still open; not merged because validation didn't break PR #22's ceiling)
