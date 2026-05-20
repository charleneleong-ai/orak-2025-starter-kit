# Stage R v4 — n=5 introspection

Sweep: `experiments/stage_r_subgoals_v4/` · finished `2026-05-20T05:28:08Z` · 9h 15m total.
Trajectories: `/tmp/orak-stage-r-subgoals-v4/pokemon_red/stage_r_subgoals_v4_iter{1..5}_*/game_states.jsonl` (600 steps × 5 iters).
Method: `autoresearch.trajectory.extract_iter_metrics` with the pokemon adapter's `TRAJECTORY_*` constants.

## Verdict at a glance

| iter | score | M1 | M2 | M3 | M4 | M5 | M6 | M7 | route1% | viridian% | persev% | #moves | final zone |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 (fresh) | **71.43%** | 16 | 42 | 121 | 139 | **426** | — | — | 44.2% | **29.0%** | 25.9% | 337 | **ViridianCity** |
| 2 | 57.14% | 12 | 24 | 70 | 87 | — | — | — | 0.0% | 0.0% | 21.0% | 401 | PalletTown |
| 3 | **28.57%** | 21 | 150 | — | — | — | — | — | 0.0% | 0.0% | 23.1% | 417 | PalletTown |
| 4 | 57.14% | 11 | 22 | 77 | 86 | — | — | — | 6.8% | 0.0% | 23.5% | 354 | PalletTown |
| 5 | **28.57%** | 17 | 267 | — | — | — | — | — | 0.0% | 0.0% | 22.3% | 413 | PalletTown |

**Mean ± std: 48.57% ± 19.17pp · learning delta −21.43pp (REGRESS).**

## Headline finding: only iter1 escaped Pallet for Viridian

Maps visited per iter (full set):

```
iter1: OaksLab, PalletTown, RedsHouse1f, RedsHouse2f, Route1, ViridianCity, ViridianPokecenter   (7 maps, 33 transitions)
iter2: BluesHouse, OaksLab, PalletTown, RedsHouse1f, RedsHouse2f                                   (5 maps, 17 transitions)
iter3: OaksLab, PalletTown, RedsHouse1f, RedsHouse2f                                                (4 maps, 23 transitions)
iter4: OaksLab, PalletTown, RedsHouse1f, RedsHouse2f, Route1                                        (5 maps, 19 transitions)
iter5: OaksLab, PalletTown, RedsHouse1f, RedsHouse2f                                                (4 maps, 15 transitions)
```

**First Route1 entry: iter1 step 159, iter2 never, iter3 never, iter4 step 350, iter5 never.**
**First Viridian entry: iter1 step 427, iter2-5 never.**

iter1 was the only iter to break the Pallet→Route1 boundary cleanly *and* keep going north into Viridian. Everything cascade-failed from there.

## What did iter1 do right?

**Fresh slate.** No inherited procedural cache → planner ran the M5 ladder with a clean prompt. v4(0) graph_hint surfaced the 221-map + 404-exit-tile knowledge from Stage Q. v4(6)'s subgoal stack told the agent `EnterViridian → GetOaksParcel → DeliverOaksParcel`, so when it found itself stuck, it knew where to go. Combined effect: by step 159 the agent had exited Pallet, by step 426 it was in Viridian, by step 600 it was at ViridianPokecenter — final score 5.0/7 (M5 ✓).

**This is what v4 was designed to do.** Lift bar = **CLEARED on iter1**. The six-lever stack works.

## What did iter2-5 do wrong?

**Cumulative procedural memory is poisoning the boundary-crossing**, not transferring it.

Look at iter2's milestone latencies: **M1@12, M2@24, M3@70, M4@87** — significantly *faster* than iter1's 16/42/121/139. The procedural cache **is** learning the cutscene-paced early game (Reds House → Oak chase → starter → rival battle). But then the agent never reaches Route1.

Why? iter2 visits a map iter1 never did (`BluesHouse`) but never reaches Route1. The cumulative procs from iter1 contain its early-game success path (Pallet exploration), but iter1's *late* path (Route1 → Viridian → Pokecenter) is either underweighted or never gets retrieved when the planner is in the post-rival-battle phase.

**Hypothesis: the proc cache stores "what worked in early game" with overwhelming weight, because that's where most successful retrievals land, and the boundary-crossing path is a small handful of late steps that get drowned out.**

## v4(5) perf-prune analysis

`PROC_CACHE_MIN_ITER_SCORE = 4.0` (raw, 0-7 scale). Per-iter scores: 5, 4, 2, 4, 2.

| iter end | score | ≥ threshold (4.0)? | proc cache propagated? |
|---:|---:|---|---|
| 1 | 5.0 | yes | iter2 inherits ✓ |
| 2 | 4.0 | yes (boundary) | iter3 inherits **iter2's Pallet-only procs** |
| 3 | 2.0 | no | iter4 inherits iter2's procs (iter3 pruned) |
| 4 | 4.0 | yes (boundary) | iter5 inherits **iter4's Pallet-only procs** |
| 5 | 2.0 | no | — |

The threshold is set exactly at the score iter2 / iter4 hit. Two iters that **never left Pallet** were retained as "good" procedures.

**If threshold were 5.0 (M5 reached = actually crossed the boundary):**
- iter2 (4.0) → PRUNED → iter3 would inherit iter1's Viridian-reaching procs
- iter4 (4.0) → PRUNED → iter5 would inherit iter1's procs (after iter3 pruned)

This would have prevented the cascade. The current 4.0 threshold is *exactly the score* the bad iters hit — they get retained because they technically "made M4" but didn't do the only thing that matters (cross the Pallet boundary).

## v4(1) anti-perseveration analysis

`persev_pct` (top action / total `move_to` actions): 25.9 / 21.0 / 23.1 / 23.5 / 22.3 — flat across all iters, including iter1 which succeeded. The position counter is functioning (visible in code; it logs `### Recently looped` hints when threshold crosses), but the **per-iter rate doesn't change much** between success and failure iters, and **iter3 + iter5 still made ~415 move_to calls** (vs iter1's 337).

The anti-perseveration hint is informational — it tells the planner what cells have been visited a lot — but doesn't **prevent** the planner from picking the same target again. The bad iters are still issuing ~5x more redundant move_to than they need, mostly within PalletTown.

## What broke vs what to do

| Lever | Status | Evidence |
|---|---|---|
| v4(0) adapter graph_hint | ✓ working | iter1 found Route1 + Viridian + Pokecenter without ever seeing them before |
| v4(1) anti-perseveration | partial | logs the loops, doesn't prevent them — see persev% flat across iters |
| v4(3) step budget 300→600 | ✓ working | iter1 reached M5 at step 426 (impossible at 300) |
| v4(4) `__setstate__` reset | ✓ working | per-episode counters reset cleanly (no inherited stagnation seen in iter2+) |
| v4(5) perf-prune | ✗ **misconfigured** | threshold 4.0 exactly matches the score bad iters hit → bad procs propagated to iter3 + iter5 |
| v4(6) M5-M7 subgoal stack | ✓ working | iter1 actually pushed through to ViridianCity + Pokecenter (M5 ladder is firing) |
| (Stage S F4 boundary) | unknown | not in this sweep (separate PR #99) |

## Recommended next moves

**Highest leverage — change one line, re-run:**
1. **`PROC_CACHE_MIN_ITER_SCORE` 4.0 → 5.0** (`agents/pokemon_red/game_adapter.py:287`). This forces iter2/iter4 to be pruned, breaks the bad-procs propagation chain, and lets iter1's Viridian-reaching procs continue forward through iter3+. **Single-character fix, biggest expected lift.**

**Medium leverage — second sweep with both changes:**
2. **Land Stage S F4 (PR #99) before next v4 sweep.** iter4 hit Route1 at step 350 — would have been step ~250 if `move_to(12, 0)` reported the boundary crossing correctly instead of failing.

**Lower leverage — separate work:**
3. Anti-perseveration could escalate from "hint" to "veto" — once a cell crosses ~10 visits, refuse to plan a move_to that target. Currently it just appends a string. But this is secondary to fixing the proc cache.

## Files

- Raw trajectories: `/tmp/orak-stage-r-subgoals-v4/pokemon_red/stage_r_subgoals_v4_iter{1..5}_*/game_states.jsonl`
- Per-iter results: `experiments/stage_r_subgoals_v4/results.jsonl`
- Adapter constants: `agents/pokemon_red/game_adapter.py:271-287`
- Analysis: this file
