# Stage R v5 n=5 introspection — perf-prune threshold bump didn't break the ceiling

Sweep: `experiments/stage_r_subgoals_v5/results.jsonl` (finished 2026-05-20T14:47:23Z).

## Result table (v5 vs v4)

| Iter | v4 score% | v5 score% | v5 first-M5 | v5 #move_to | v5 maps (top 3, dwell) |
|---:|---:|---:|:--:|---:|---|
| 1 | 71.43 | **71.43** | step 319 | 344 | Viridian 281, Route1 157, OaksLab 74 |
| 2 | 57.14 | 28.57 | — | 433 | PalletTown 565 (94%), RedsHouse1f 18, OaksLab 11 |
| 3 | 28.57 | 57.14 | — | 373 | PalletTown 476 (79%), OaksLab 98, RedsHouse1f 18 |
| 4 | 57.14 | 57.14 | — | ~370 | PalletTown ~470, OaksLab ~110, RedsHouse 30 |
| 5 | 28.57 | **0.00**† | — | — | — |

† iter5 killed by ENOSPC at launch — `error: No space left on device (os error 28) at "/tmp/.tmpH1Rrar"` at `14:47:23Z` immediately after `Iter 5/5 inherit_from=stage_r_subgoals_v5_iter4` was logged. 4 iters × 600 steps of `UnifiedMaclaAgent_step_*.pkl` checkpoints had filled `/tmp`. iter4 pickle was lost during the subsequent cleanup, so iter5 can't be re-run in its original chain form. Not a real score — placeholder 0.00.

## Headline

The threshold bump 4.0 → 5.0 changed iter ordering but **not the ceiling**. Mean stayed at ~53% (vs v4 ~48%), iter1 still alone at M5, every inheriting iter still stuck in PalletTown.

## What the v5 trajectories actually show

- **iter1 (fresh slate, score 5/M5 EnterViridian)**: 53% of cells in Viridian/Route1 (281 + 157 / 600). Genuine Pallet → Route1 → Viridian crossing at step 319.
- **iter2 (inherits iter1)**: 94% PalletTown dwell, *zero* Route1, *zero* Viridian. M2 reached at step 88, then nothing.
- **iter3 (inherits iter2's chain → still iter1's procs)**: Reaches M4 at step 106 — **without ever stepping on Route1 or Viridian**. M4 is therefore a Pallet-internal event (likely "got starter Pokemon"), confirming v4 introspection's "M4 isn't a boundary signal" thesis.
- **iter4 (inherits iter3)**: same shape as iter3 — M4 at step 160, zero Route1/Viridian dwell.

## Why the threshold bump didn't help

The perf-prune gate works correctly on the **write** side:

- iter1 final=5.0 ≥ 5.0 → iter1's procs are kept and passed forward.
- iter2 final=2.0 < 5.0 → iter2's procs are discarded.
- iter3 final=4.0 < 5.0 → iter3's procs are discarded.

So iter2/3/4 all inherit the same cache: iter1's procs. The log confirms this — at every iter boot we see only the staleness prune (`Stage L: pruned 7 stale procedures (unused for >=2 iters)`), not the perf-prune.

**The bug is on the read side, not the write side.** iter1's procs — the only ones being passed forward — are themselves the PalletTown-loop source when re-loaded into a fresh boot:

1. iter1 succeeded by *exiting* Pallet via the graph_hint + escape valve, with **no prior cache pressure** competing.
2. Cached, those exit-trajectory procs encode `(map=PalletTown, obs ≈ X) → action Y`. At iter2/3/4 boot, Y matches PalletTown obs again, the selector picks the cached proc, and the agent loops on iter1's mid-trajectory PalletTown actions — but no longer in the spatial state that made those actions productive.
3. graph_hint and escape valve still fire (visible in the live log: `looped_positions_hint (26 cells over threshold)`, `subgoal escape valve fired: EnterViridian stagnation=249`) but they only **drop** the subgoal from the planner prompt — they don't **veto** cache selection, which keeps winning at `theta=0.050`.

## Cross-iter signals

- **Perseveration%** went *down* (33.5 → ~22) for inheriting iters — the agent isn't repeating targets, it's exploring *different* PalletTown cells each step. So the anti-perseveration lever is doing its job; this confirms the bug isn't "same move over and over", it's "stuck within map boundary".
- **#move_to calls** went *up* (344 → 433 in iter2) for inheriting iters — more activity, less progress. Cache is producing rapid-fire micro-moves inside Pallet.
- **First-M2 step** stayed fast (12–88 across all iters) → leaving Oaks Lab is solved. The wall is M3 → M5, specifically the Pallet → Route1 → Viridian boundary crossing.

## Diagnosis

v4 framing was: "low-score iters poison the cache" → bumping the threshold should fix it.

v5 disproves that. **The real failure mode is**: *any* PalletTown-tagged proc inherited from a prior iter overrides the escape signals in a fresh boot. iter1's "good" procs are the poison when re-applied.

## Options for next stage → moved to Stage S

Four candidate fixes, full design in `docs/macla/stage_s_cache_veto.md` on the `feat/macla-stage-s` branch:

- **A. No-inherit baseline** (one-line, diagnostic): drop inheritance entirely (`inherit_from=None` for every iter). Confirms upper bound when inheritance is fully off.
- **B. Cache veto under escape-valve fire** (primary): when the subgoal escape valve fires for the current step, also veto cached-proc selection for the next K steps. Lets fresh planning win without nuking the cache. Task-agnostic MACLA-level change.
- **C. Zone-tag purge at boot**: drop all procs whose recorded firing context = `map=PalletTown` from the inherited cache. Pokemon-specific.
- **D. Cache eligibility ratchet**: only inherit procs that were active during a *score-increment* event with score ≥ M5. Generalisable in principle; needs proc→step attribution infra.

Recommended staged rollout on Stage S: **A** first (sizes the ceiling lift), then **B** as the principled fix, **D** if **B** alone doesn't match **A**'s ceiling.

Plus tertiary scope: move sweep checkpoints out of `/tmp` (the v5 iter5 ENOSPC was preventable — `/workspace` has 199G, `/tmp` did not).

## Out of scope

- The `M{i}: score >= i` milestone labels in `TRAJECTORY_MILESTONES` are abstract; iter3/4 reaching M4 without leaving Pallet is now triple-confirmed (v4 introspection, v5 iter3 fresh, v5 iter4 fresh) — the threshold being correct at M5 stands.
- vLLM, GPU, daemon health, log triage are all clean; the bug is purely in cache-inheritance semantics.
