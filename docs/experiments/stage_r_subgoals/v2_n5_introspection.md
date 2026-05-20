# Stage R v2 (active subgoal as planner hard constraint) — n=5 introspection (cancelled @ iter 4)

**Verdict:** REGRESS on iter 1-2, recovery on iter 3 — sweep cancelled before iter 4-5. Scores **28.57 / 28.57 / 57.14** vs Stage R v1's flat **57.14 × 5**. The "hard constraint" planner phrasing (commit `59a66b5`) locks the agent in PalletTown when the executor's `move_to` can't traverse the Pallet→Route1 map edge, and the cumulative-memory chain inherits the bad procs into iter 2, making it worse.

**Cancelled:** 2026-05-18 20:29Z (mid iter 4, step 50/300)
**Branch:** `feat/macla-stage-r-subgoals` (PR draft)
**Worktree:** `/workspace/orak-stage-r-subgoals`
**Log:** `logs/stage_r_v2_sweep_20260518T173929Z.log`

## Hypothesis

Stage R v1 (hierarchical subgoals + Reflexion summary, off master) was FLAT at 57.14% × 5 — the planner saw the subgoal stack but the executor ignored it. v2 added the active subgoal directly into the planner prompt as a **hard constraint**:

> `[Active subgoal — pursue this until its completion predicate fires]`

Expected effect: planner stops drifting between candidate subgoals on the same step, executor commits, agent breaks through the M5 ceiling. Lift bar: any iter > 57.14%.

## Schedule

| Setting | Value |
|---|---|
| Game | pokemon_red |
| Agent | gemma_26b (Gemma 4-26B-A4B-AWQ-4bit on vLLM :8000) |
| Max steps / iter | 300 |
| Iters | 5 planned, 3 completed + 1 partial (cancelled) |
| Launcher | `experiments/stage_r_subgoals/run_pokemon_n5.sh` |
| Proc-cache prune | `PROC_CACHE_MIN_ITER_SCORE = 4.0` (Stage Q v2 threshold — see "Why the prune didn't fire" below) |

## Results

| iter | score | M-progression | top map dwell | PalletTown move_to (post-starter) | inherits from |
|---:|---:|---|---|---:|---|
| 1 | **28.57%** | M1@27 · M2@73 · stuck | PalletTown **78.3%** | 130 moves / **46 unique waypoints** | NONE (fresh, v1 iter 5 Reflexion) |
| 2 | **28.57%** | M1@24 · M2@94 · stuck | PalletTown **90.0%** | 158 moves / **51 unique waypoints** | iter 1 |
| 3 | **57.14%** | M1@25 · M2@49 · M3@92 · M4@108 | Route1 42.7% · PalletTown 32% | 38 moves / 21 unique waypoints | iter 2 |
| 4 | (cancelled, step 50) | — | — | — | iter 3 |
| (v1 iter 5 baseline) | 57.14% | M1@25 · M2@47 · M3@68 · M4@80 | Route1 41.3% · PalletTown 31% | — | iter 4 |

## Failure mode — PalletTown lock

After getting the starter at Oak's Lab (M2, score 2.0), iter 1 ping-pongs between the lab and PalletTown then enters a dart-throwing wander:

```
73 PalletTown  warp(12,11) → OaksLab     (M2 fires: score 1.0 → 2.0)
74 OaksLab     interact(SPRITE_GIRL_9)
75 OaksLab     warp(4,11) → PalletTown
76 PalletTown  warp(12,11) → OaksLab      ← pointless ping-pong
77 OaksLab     warp(4,11) → PalletTown
78 PalletTown  move_to(12, 0)             ← *correct* — north exit to Route1
79-83 PalletTown move_to(12, 5) × 5       ← move_to stalls at (12,5), retries 5×
84 PalletTown  warp(12,11) → OaksLab      ← gives up, ping-pongs again
…
240-260 PalletTown move_to({(3,0),(6,16),(3,16),(7,9),(3,15),(2,16),(4,14),(2,15)…})
                                          ← directionless dart throwing
```

The planner **knew the goal** — `y_dest=0` (north edge) was attempted 9 times in iter 1, 9 in iter 2. `overworld_map_transition` was called **33 times** in iter 1 and **21 times** in iter 2 — and still couldn't escape PalletTown. This is not "the planner forgot the goal." It's `move_to` failing to land on the exact tile that triggers the Route1 transition, while the hard-constraint phrasing forbids the planner from trying anything else.

## Why iter 3 recovered

Cumulative memory crossed a tipping point: enough good Route1-bound procedures from prior v1 runs (inherited via `--load-checkpoint --prev-run-id`) finally outvoted the bad PalletTown-wandering procs that iter 1 + iter 2 piled in. Iter 3's M-ladder timing (M2@49 → M3@92 → M4@108) is *tighter on the early milestones* than v1 iter 5 (M2@47 → M3@68 → M4@80) but ends at the same 57.14% ceiling — i.e., the recovery matched v1's behaviour exactly. v2 added zero ceiling lift.

## Why the perf-prune didn't fire

Stage Q v2 added `prune_low_score_iter(score_threshold=4.0)`: drop procedures whose `origin_iter` matches a prior iter that scored below `PROC_CACHE_MIN_ITER_SCORE = 4.0`. Both iter 1 and iter 2 scored **2.0** (raw), which is **below 4.0** — so by spec the prune should have fired and dropped iter 1's bad procs before iter 2 loaded. Two possibilities to investigate before rerunning:

1. The prune predicate uses the **normalized** evaluation_score (28.57 ≥ 4.0) instead of the raw game score (2.0 < 4.0). Threshold semantics drift.
2. The prune only fires when the *previous* iter's score is known via `last_iter_score`, and that state didn't propagate across the iter-1 → iter-2 checkpoint reload.

Either way, the prune was a no-op here — the v2 hard-constraint amplified a failure the perf-prune was supposed to absorb.

## Diagnosis

Three failure layers, ordered by depth:

1. **Tool layer (root cause):** `move_to(12, 0)` cannot traverse the PalletTown→Route1 map edge; it stops at `(12, 5)`. The executor has no recovery — same waypoint is retried 5× then abandoned for random other waypoints.
2. **Planner layer (v2 amplifier):** The hard-constraint phrasing forbids the planner from demoting the failing subgoal and trying alternatives. The result is wide dart-throwing — 46 unique waypoints tried in iter 1, 51 in iter 2 — instead of a structured search.
3. **Memory layer (compounding):** The proc cache learns the bad pattern; iter 2 inherits and *worsens* (PalletTown dwell 78% → 90%). The perf-prune that was supposed to catch this never fires (see above).

## Recommended fixes (no action yet — flagging for discussion)

**Cheapest first:**

- **F1. Soften the constraint phrasing.** Change `[Active subgoal — pursue this until its completion predicate fires]` → `[Currently pursuing: <subgoal>. Prefer continuing unless evidence suggests it's blocked.]`. Removes the hard lockout while preserving the v2 intent of "commit, don't drift every step." → revert/replace `agents/macla/unified.py` planner-prompt change in commit `59a66b5`.
- **F2. Subgoal-failure escape valve.** If the active subgoal's completion predicate hasn't fired in N steps (suggested N=30), demote it from hard constraint to one candidate among many — let the planner pick freely until it fires, then re-promote. → `agents/macla/unified.py:_active_subgoal_for_planner` or wherever the prompt is composed.
- **F3. Fix the perf-prune semantics bug first.** Verify whether `PROC_CACHE_MIN_ITER_SCORE = 4.0` is being checked against raw game score (right) or normalized eval score (wrong). A failing-but-not-yet-debugged perf-prune undermines every cumulative-memory experiment downstream. → `agents/macla/macla_lib.py:prune_low_score_iter` + `agents/pokemon_red/game_adapter.py`.

**Deeper, separate workstream:**

- **F4. `move_to` boundary detection.** The tool should detect "destination unreachable, stuck on impassable tile" and either (a) automatically issue `overworld_map_transition` when adjacent to a map boundary tile, or (b) return a structured failure that the executor can react to. Currently it silently lands on a nearby tile and reports success. → `evaluation_utils/mcp_game_servers/pokemon_red/` (the executor side, not MACLA).

**Suggested next sweep:** F1 + F2 combined (planner-side, no tool changes), n=5 — if those alone match v1's 57.14% baseline without the iter 1-2 sag, the hard-constraint was unambiguously net-negative. Defer F4 until F1+F2 lift past the M4 ceiling.

## Out-of-scope but worth noting

The sweep orchestrator never wrote the iter 1-3 outcomes to `experiments/stage_r_subgoals/results.jsonl` — the file still contains only the stale Stage R v1 row from 06:00Z. All iter scores in this writeup were recovered by grepping `\[iter [0-9]+\] eval=` from `logs/stage_r_v2_sweep_20260518T173929Z.log`. Worth confirming whether `experiments/autoresearch.py` appends per-iter or only at sweep completion, because cancelling mid-sweep should still leave a record. (Plausibly a v0.24.2 quirk; the rebase to v0.26.2 followups may already fix it.)
