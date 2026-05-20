# Stage S — cache ↔ escape-valve arbitration (and existing move_to boundary detection)

This branch consolidates two follow-ups deferred from Stage R (PR #97). Both are MACLA / agent-side improvements identified by the v4 and v5 n=5 introspections, but they sit at very different priorities and don't share files.

## Primary: cache veto under escape-valve fire

**Source of the problem.** Stage R v5 n=5 introspection — `docs/experiments/stage_r_subgoals/v5_n5_introspection.md` on the `feat/macla-stage-r-subgoals` branch — proved that bumping `PROC_CACHE_MIN_ITER_SCORE` 4.0 → 5.0 *correctly gates the proc cache on the write side* (iter2/iter3 procs are discarded as designed) but doesn't break the M5 ceiling. The bug is on the read side:

1. iter1 (fresh slate) succeeds because it has no cache pressure competing with the `graph_hint` + escape valve.
2. iter2/3/4 inherit iter1's procs. Those procs encode `(map=PalletTown, obs ≈ X) → action Y` — fine in iter1's trajectory, but at a *fresh* iter boot the selector picks the cached proc at `theta=0.050`, the agent loops on iter1's mid-trajectory PalletTown actions, and the escape valve's planner-prompt drop is irrelevant because the cache wins arbitration before the planner is even consulted.

Triple-confirmed signals (across v4 + v5 introspections):
- iter1 (fresh) reaches Viridian; every inheriting iter stays in PalletTown.
- M4 (score 4/7) is achievable inside PalletTown — confirms M5 (5.0) is the correct write-side gate; the wall is read-side.
- Perseveration % went **down** in inheriting iters (the agent explores *different* Pallet cells), so the failure isn't action repetition; it's map-boundary lock-in.
- `subgoal escape valve fired: EnterViridian stagnation=249` fires repeatedly in the log; the cache keeps winning.

**Recommended fix order.** Four candidates from the v5 verdict, ranked:

| # | Candidate | Generalisable? | Cost | Notes |
|---|---|---|---|---|
| 1 | **Cache veto under escape-valve fire** — when the escape valve fires for step t, suppress cached-proc selection (force planner fallthrough) for the next K steps | Yes — MACLA-level, task-agnostic | Small | Primary target. Reuses existing `subgoal_stagnation_steps` signal. |
| 2 | Per-proc eligibility ratchet — only inherit procs that fired during a score-increment event ≥ M5 | Yes in principle | Medium — needs proc→step attribution infra | Right long-term answer, more invasive |
| 3 | No-inherit baseline — flip `inherit_from=None` for n=5 | Yes (config flag) | Trivial | Diagnostic only — quick empirical size-of-win check |
| 4 | Zone-tag purge — drop `map=PalletTown` procs from inherited cache at boot | Pokemon-specific | Small | Included for completeness; not the generalisable fix |

**Suggested staged rollout:**
1. Run **#3** first as a one-iter sweep to size the ceiling lift when inheritance is fully off. Cheap; sets the upper bound.
2. Implement **#1** (cache veto under escape-valve fire). Test against the same n=5 schedule that v5 used; bars are: minimum no iter < 50%, lift mean > 57.14%, stretch ≥2 iters > 71.43%.
3. If #1 lifts but doesn't fully match #3's ceiling, layer in **#2** as the principled boundary-crossing-verified inheritance gate.

**Files likely touched (for #1):**
- `agents/macla/macla_lib.py` — `select_procedure` (the `theta=0.050` selector site) needs a `veto_until_step` check.
- `agents/macla/unified.py` — `_base_fallback` already detects the escape-valve fire; set the veto window there.
- `agents/macla/base.py` — `EnhancedHierarchicalMemorySystem` may need a `_cache_veto_until_step` field (per-episode; clears via `__setstate__` like the other Stage R v4 per-episode fields).
- `tests/test_macla_stage_s_cache_veto.py` — new test module (parametrised: K step values, veto-clears-on-mutation, veto-doesn't-leak-across-iters).

## Secondary: `move_to` boundary detection (already at `ff88e84`)

The first commit on this branch — `ff88e84 feat(pokemon): Stage S — move_to boundary detection (F4)` — was originally scoped from the v4 introspection (deferred for clean attribution across the six v4 levers). It addresses a different bug:

- The pokemon navigation tool silently lands on a nearby reachable tile and reports `success=True` when the requested destination is unreachable or across a map boundary.
- v4 iter4 hit Route1 only at step 350 because of this (the executor kept retrying the same `move_to(Route1)` call thinking it had succeeded when it had stalled at the Pallet→Route1 edge).
- Fix returns a structured failure on misaligned final position, or auto-promotes to `overworld_map_transition` when the target is past a known exit-tile.

This is an **efficiency** fix, not a capability fix — it makes successful runs faster but won't lift the ceiling on its own. Keep it in this branch alongside the cache-veto work since both belong to "agent-side improvements that came out of Stage R sweep introspections", but ship behind whichever capability fix lands first if PR size becomes a concern (split the commits at review time).

## Test plan (placeholder — fill in as commits land)

- [x] Stage R v5 introspection signed off (PR #97) — confirms the diagnosis
- [ ] **#3 no-inherit baseline** — n=5 sweep, single config flip, sizes the upper bound
- [ ] **#1 cache veto** — TDD: failing tests in `tests/test_macla_stage_s_cache_veto.py` before implementation
- [ ] n=5 sweep with cache veto enabled — bars: minimum no iter < 50% · lift mean > 57.14% · stretch ≥2 iters > 71.43%
- [ ] move_to boundary detection — pre-existing commit `ff88e84` needs its own test coverage check (was this already TDD'd?)

## Tertiary: move sweep checkpoints out of `/tmp`

The Stage R v5 sweep was bitten by `/tmp` ENOSPC on 2026-05-20 — 4 iters × 600 steps of `UnifiedMaclaAgent_step_*.pkl` checkpoints filled the partition, iter5 died at launch with `error: No space left on device (os error 28) at /tmp/.tmpH1Rrar`, and the iter4 checkpoint pickle was then lost during the disk cleanup (couldn't re-run iter5).

The default path is `evaluation_utils/checkpoint_manager.py`'s `/tmp/orak-<branch>-<sweep>/.../checkpoints/`. Two changes worth bundling here:

1. **Move default to `/workspace`** — `/workspace` is the 199G partition; `/tmp` was the much smaller scratch budget. Either change the checkpoint-manager default or make the launcher set `CKPT_ROOT=/workspace/orak-<branch>/checkpoints` explicitly.
2. **Rolling-window cleanup** — keep only the N most recent checkpoints per iter (e.g. N=3) and the *final* checkpoint per iter. Current behaviour writes every step's pickle and never garbage-collects within an iter.

Plus a launcher hygiene fix: `run_pokemon_n5_v5.sh` (and successors) should `rm -rf` the prior sweep's checkpoint dir on start to prevent cross-rerun buildup.

These don't block #1 (cache veto) but should land in the same Stage S PR — preventing a recurrence of the v5 outcome.

## Out of scope for this branch

- Anything that requires changes to PR #97's Stage R levers (those are stable and shipping)
- GSPO data collation work — that's PR #100, different lever entirely (gradient-based policy update vs symbolic procedural memory)
- The Stage T "auto-derive milestone slot list from `TRAJECTORY_SCORE_MAX`" follow-up from PR #97
