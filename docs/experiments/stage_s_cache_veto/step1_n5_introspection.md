# Stage S Step 1 n=5 introspection — FLAT, cache veto reproduces v3 exactly

Sweep: `experiments/stage_s_cache_veto_step1_no_inherit/results.jsonl` (finished 2026-05-21T01:42:15Z, 8h49m total).

## Final result table

| Iter | v3 (inherit ON, no veto) | v4 (inherit ON, lever stack) | v5 (inherit ON, perf-prune 5.0) | **Step 1 (inherit OFF, cache veto)** |
|---:|---:|---:|---:|---:|
| 1 | 57.14 | **71.43** (M5) | **71.43** (M5) | 57.14 (M4) |
| 2 | 57.14 | 57.14 | 28.57 | 57.14 (M4) |
| 3 | 57.14 | 28.57 | 57.14 | 57.14 (M4) |
| 4 | 57.14 | 57.14 | 57.14 | 57.14 (M4) |
| 5 | 57.14 | 28.57 | 0.00 (ENOSPC) | 57.14 (M4) |
| **mean ± std** | **57.14 ± 0** | **48.57 ± 19** | **42.86 ± 28** (n=4: 53.57 ± 18) | **57.14 ± 0** |
| **verdict** | FLAT | MIXED | REGRESS | **FLAT** |

> Step 1 reproduces v3 exactly — zero variance, M4 ceiling, +0.00pp learning delta. The v4/v5 iter1 spikes at 71.43% were **variance, not signal**. With cache veto active, the agent reliably hits M4 and stops.

## Headline

**The Stage S hypothesis was wrong.** We thought cache inheritance was the wall — that v4/v5's inheriting iters got stuck because the inherited cache poisoned the selector. Step 1 removed inheritance and added the cache veto, and we landed at exactly the v3 ceiling. **Without inheritance, the M4 ceiling holds firm.** v4 iter1's 71.43% and v5 iter1's 71.43% were lucky one-off variance — when run n=5 with no inheritance, fresh iters reliably stop at M4.

## What the trajectories show

Every iter showed the same shape:

1. Reach Route1 within 50-150 steps (NavigateToMap(Route1) pops, EnterViridian becomes top)
2. Bounce back to PalletTown some time later
3. Stuck in PalletTown for hundreds of steps with EnterViridian stagnation climbing to 200-400+
4. Escape valve fires constantly, cache veto fires constantly, looped-positions hint fires (10-37 cells over threshold)
5. Run out of steps at 600

The 600-step budget caps each iter at ~100min wall-clock. The planner gets ~30 step windows of "no cache" (cache veto window) but can't break the loop on its own.

## Why this happens — sharper diagnosis

The agent is structurally unable to plan the Pallet→Route1→Viridian transition reliably. Possible root causes (Stage T territory):

1. **Planner blindness to the map graph at the transition** — `graph_hint` shows Pallet→Route1 reachable, but the planner doesn't use it to emit `overworld_map_transition(Route1)` or `move_to` toward the southern Pallet exit tile. The agent reaches Route1 (which means it CAN cross), but only as a side-effect of random-walk move_to attempts, not deliberate planning.
2. **Subgoal stack ordering — `EnterViridian` activates before the agent is anywhere near Viridian's gate**. The planner's "Currently pursuing: EnterViridian" prompt may push it toward Viridian-adjacent actions before it's even at Route1's north edge.
3. **Stage Q's exit-tile coordinates** — `graph_hint` includes the exact Pallet→Route1 exit tile (`(11, 0)`), but the planner may not be wired to read coord-tagged hints. Worth grep'ing whether the exit_tiles dict actually reaches the prompt.

The cache veto is **not** the right lever here. Inheritance isn't the wall. The wall is somewhere in the planner ↔ adapter ↔ graph_hint plumbing for the specific Pallet→Route1 boundary crossing.

## Implication — DO NOT FIRE Step 2

Step 2 (cache veto with inheritance ON) is mechanistically the same situation as Step 1 plus inheritance speedup. The speedup helps early-game (M1-M4 cutscene) but inheritance was *never the wall*. Step 2 would produce: 5 × 57.14% or worse (the cache veto suppresses inherited procs that could have helped, no upside). **Skipped.**

## What Stage S still ships

- ✅ **move_to boundary detection** (`856f84e`) — orthogonal efficiency fix, still valuable. Reduces wall-clock for runs that *do* cross boundaries.
- ✅ **Checkpoint hygiene** (`561eec7`) — `CheckpointManager(keep_last_n=N)` + autoresearch `warn_if_tmp_data_dir` import. Already preventing the v5 ENOSPC failure mode.
- ❌ **Cache veto under escape-valve fire** (`d070f07`) — *the policy is wrong*. The mechanism (per-step veto window with __setstate__ reset and 13 unit tests) is sound, but vetoing the cache during stagnation hurts in this domain. Two options for the PR:
  - **Disable by default**: set `CACHE_VETO_K_STEPS = 0` in `unified.py` so the veto is a no-op until someone sets a positive value at runtime. Keeps the code mergeable; future sweeps can experiment with it.
  - **Revert the commit**: remove the cache veto entirely from the PR.

Recommend **disable by default** — preserves the wiring, lets us tune K later (maybe K=3-5 is right, not 30), and avoids re-doing the TDD work if the policy turns out to be salvageable.

## What to investigate next (Stage T scope)

Beyond Stage S. Need a separate branch.

1. **Trace the Pallet→Route1 transition** in iter1's `game_states.jsonl` — find the steps where `map_name` changed and look at the planner's emitted subtask + tool call at each. Does the planner ever emit a tool call that intentionally targets the boundary?
2. **Check graph_hint payload** at run time — log the actual prompt segment the planner sees. Is `(11, 0)` exit-tile coords getting rendered, or just adjacent map names?
3. **Subgoal-stack ordering bug?** — Does `NavigateToMap(Route1)` stay on top until the agent is genuinely on Route1, or does it pop on a transient observation? The "agent reached Route1 then bounced back to Pallet" pattern suggests the stack popped too eagerly.
4. **Compare to Stage Q's iter1** which DID hit M5 at 71.43%. What did the planner do differently there? Was it the same prompt structure?

These are diagnostic experiments, not lever changes. Light-touch; should run quickly with the trajectory introspection framework that already exists.

## Cross-reference

- Stage R v5 introspection: `docs/experiments/stage_r_subgoals/v5_n5_introspection.md` (the v5 sweep that motivated cache veto — confirmed write-side gate is correct, hypothesised read-side as the wall, which Step 1 has now disproven)
- Stage S design doc: `docs/macla/stage_s_cache_veto.md` (Options A-D for cache-arbitration fixes; this writeup falsifies A and B)
- PR #101 — Stage S branch; will be updated with FLAT verdict + disable-by-default proposal
