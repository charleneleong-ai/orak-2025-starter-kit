# Stage S v2 post-Viridian chain — KEEP, but M5 wall holds

Sweep: `experiments/stage_s_v2_post_viridian_chain/results.jsonl` (finished 2026-05-22T12:25:00Z, 8h56m total).

## Hypothesis

v1 broke the M4 wall by inserting a `NavigateToMap(ViridianCity)` bridge above the M5 score gate (mean 74.29%, +17.15pp over baseline). 4/5 v1 iters stalled at M5 — entered Viridian, never picked up Oak's Parcel; 1/5 punched M6.

v2 extends `_POKEMON_MILESTONE_LIBRARY` using the framework hook from PR #103:

```python
6: MilestoneSpec(..., requires_location="ViridianMart")
7: MilestoneSpec(..., requires_location="OaksLab")
```

`build_score_milestone_stack` auto-inserts the matching bridge above each gate. The top→bottom pop order becomes `Route1 → ViridianCity → M5 → ViridianMart → M6 → OaksLab → M7` — same recipe v1 used at M5, applied at three levels.

Prediction: lift v1's mean by reaching M6 / M7 in some fraction of iters.

## Final result table

| Iter | v1 viridian bridge | v2 post-Viridian chain |
|---:|---:|---:|
| 1 | 71.43 (M5) | 57.14 (M4) |
| 2 | 71.43 (M5) | 71.43 (M5) |
| 3 | 71.43 (M5) | 57.14 (M4) |
| 4 | **85.71 (M6)** | 71.43 (M5) |
| 5 | 71.43 (M5) | 71.43 (M5) |
| **mean ± std** | **74.29 ± 6.39** | **65.71 ± 7.83** |
| **verdict** | KEEP (+17.15pp) | **KEEP** (+7.15pp early→late) |

Gap = -8.58pp (~1.1 std). Fisher exact on "hit ≥M5" (v1 5/5 vs v2 3/5): p ≈ 0.17 — not significant.

## Headline

**v2's added bridges never got exercised**, because M5 firing is itself the rate-limiter. 3/5 v2 iters reached Viridian and popped M5 cleanly; 2/5 stalled at M4 (Route1→Viridian transition failed). No iter ever reached M6 — so the `NavigateToMap(ViridianMart)` and `NavigateToMap(OaksLab)` bridges sat dormant in the stack for every iter, never becoming active.

The v2 - v1 gap looks like a regression but is **statistically indistinguishable from sample variance**. The orchestrator tagged v2 KEEP based on the +7.15pp early→late learning delta (iters 1-2 mean 64.28 → iters 4-5 mean 71.43).

## What the trajectories show

Cross-sweep introspection via `autoresearch.trajectory.extract_iter_metrics`:

| sweep | iter | score | M4@ | M5@ | Route1 dwell | Viridian dwell | end zone |
|---|---:|---:|---:|---:|---:|---:|---|
| v1 | 1 | M5 | 110 | 286 | 160 | 314 | ViridianCity |
| v1 | 2 | M5 | 115 | 245 | 91 | 355 | ViridianCity |
| v1 | 3 | M5 | 142 | 443 | 261 | 157 | ViridianCity |
| v1 | 4 | M6 | 125 | 215 | 78 | 385 | ViridianCity |
| v1 | 5 | M5 | 205 | 433 | 177 | 167 | ViridianCity |
| v2 | 1 | M4 | 99 | — | 256 | **0** | **PalletTown** |
| v2 | 2 | M5 | 145 | 415 | 251 | 185 | ViridianCity |
| v2 | 3 | M4 | 137 | — | 209 | **0** | **PalletTown** |
| v2 | 4 | M5 | 99 | 278 | 166 | 322 | ViridianCity |
| v2 | 5 | M5 | — | — | — | — | ViridianCity |

Two patterns:

1. **The wall is Route1→Viridian, not Pallet→Route1.** All 10 iters across both sweeps reach Route1 and spend 78-261 steps there. The failure mode is being on Route1 and not finding the north-edge transition tile.
2. **v2's failing iters retreat to PalletTown after Route1 wandering** (iter 1, iter 3 both ended in PalletTown despite 200+ steps on Route1). v1 never showed this pattern. With n=5 this is suggestive but not conclusive — could be the same underlying stochastic transition with a different unlucky draw.

## Procedural / vector memory audit

Searched every `_subgoal_stack` reference in `agents/`:

- `peek_subgoal()` returns single top element
- `check_active_subgoal_completion` cascade-pops from top
- `subgoal_depth()` is read in two log strings
- No procedural memory key, vector memory embedding, procedure cache, or context hash uses the stack composition

The dormant `NavigateToMap(ViridianMart)` / `NavigateToMap(OaksLab)` entries at depths 1 and 3 **cannot influence behavior** until they bubble to the top via prior pops. Prior pops never happened in the failing iters. The planner prompt during the M4→M5 transition is byte-identical between v1 and v2.

So the v2 - v1 gap is either sample variance (most likely) or external state drift (vLLM cache, GSPO reroll's effect on serving state between the two sweeps).

## Verdict — KEEP, with caveats

- ✅ **Framework lift validated**: PR #103's `MilestoneSpec.requires_location` + `build_score_milestone_stack` auto-bridge generates the correct stack composition. Tests pass; runtime stack shape matches the spec.
- ✅ **Sweep tagged KEEP** by the orchestrator's early→late delta. Late-iter mean (71.43) is the v1 baseline.
- ❌ **The actual hypothesis is untested**: no iter reached M5's pop event, so the M6/M7 bridges never had a chance to fire. We don't know if they'd work.
- ❌ **Apparent regression vs v1 is most likely noise**, but n=5 isn't enough to prove either direction.

## Next move — attack the Route1→Viridian wall, not add more bridges

Adding more `requires_location` entries deeper in the stack does nothing as long as M5 firing rate stays at 60%. The bottleneck is structural: the agent on Route1 ~40% of the time doesn't cross to Viridian. Three concrete attacks worth considering:

1. **Stage Q exit-tile coordinate plumbing audit** — `graph_hint` exposes Route1's north-edge transition, but the failing iters spend 200+ steps without using it. Verify the coord-tagged hint actually reaches the planner prompt and whether the planner is wired to emit `move_to` toward those coords.
2. **n=10 v2 rerun** to nail down whether the v2 - v1 gap is real or noise before committing to a structural fix.
3. **Per-iter step budget bump** (600 → 1000) for the failing-pattern iters — give the agent more attempts at the Route1 north edge. Cheap to try, doesn't touch any agent code.

Recommend **(1) audit first** — if `graph_hint`'s exit-tile coord isn't reaching the planner, fixing that would explain *every* sweep's Route1→Viridian variance, not just v2's.
