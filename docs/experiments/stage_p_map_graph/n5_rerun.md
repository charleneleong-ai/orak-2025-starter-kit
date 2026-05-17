# Stage P (every-step map-graph observation hint) — n=5 cumulative-memory rerun

**Verdict:** FLAT — Stage P locked at the 57.14% M4 ceiling for all 5 iters (σ=0). The map-graph hint reached the planner 1,406 times across 5 iters (≈281/iter, every step) and the planner *consumed* it — last_subtask strings like *"Move north through Route 1 to reach the entrance of Viridian City"* prove the model became goal-aware. The agent still failed to find the Route 1 → Viridian exit tile on every iter. Cache growth bested both Stage M (4) and Stage N+O (26) at **30 procs by iter 5**, but cache size alone is not the bottleneck.

**Closed:** 2026-05-17 13:29Z (~4h38m wall-clock)
**Branch:** `feat/macla-stage-p-map-graph` (PR #90)
**Worktree:** `/workspace/orak-stage-p`
**Log:** `logs/stage_p_n5_20260517T085134Z.log`

## Hypothesis

The 2026-05-15 cross-stage diagnosis (`docs/experiments/gemma/cross-stage-diagnosis.md`) identified the pokemon_red 57.14% ceiling as living at the M5 milestone gate (`'Viridian' in map_name`). Across all post-asm-fix sweeps (Stages D, H, K, L, M, N+O — ~33 episodes), **0 runs ever set foot in Viridian.** The agent reaches the Route 1 / Viridian boundary but never finds the north exit.

Diagnosis primary recommendation: *"Surface unvisited adjacent maps + visited-maps memory into the observation string every step."* That is Stage P.

Mechanism:

- Hand-authored `MAP_GRAPH` adjacency for M1–M6 territory (14 maps, symmetric edges).
- `EnhancedHierarchicalMemorySystem.map_graph_hint(current_map)` returns a multi-line natural-language hint prepended to the observation string every step (vs Stage N's one-shot novelty hint that lives in the history block).
- Planner therefore sees `### Map graph\nUnvisited maps reachable from PalletTown: OaksLab, Route1\n...` every frame.

Minimum bar: any iter `'Viridian' in final_map` OR Viridian dwell > 0 steps. Lift bar: any iter past 57.14%.

## Schedule

| Setting | Value |
|---|---|
| Game | pokemon_red |
| Agent | gemma_26b (Gemma 4-26B-A4B-AWQ-4bit on vLLM :8000) |
| Max steps / iter | 300 |
| Iters | 5 (cumulative via `--load-checkpoint --prev-run-id`) |
| Launcher | `experiments/stage_p_map_graph/run_pokemon_n5.sh` |

## Results

```
scores=[57.14, 57.14, 57.14, 57.14, 57.14]
mean=57.14% std=0.00pp
early_mean(iter 1-2)=57.14%, late_mean(iter 4-5)=57.14%, learning_delta=+0.00pp
```

Pinned at the M4 ceiling across all 5 iters. σ=0 — every iter banked exactly the same milestone set and never advanced.

| iter | score | procs in cache (end) | cumulative procs_learned | successful_execs | last_subtask |
|---:|---:|---:|---:|---:|---|
| 1 | 57.14% | 12 | 22 | 1 | "Move north toward the exit of Pallet Town to reach Route 1." |
| 2 | 57.14% | 21 | 34 | 2 | "Move towards the northern exit of Route 1 to reach Viridian City." |
| 3 | 57.14% | 31 | 50 | 3 | "Move north through Route 1 to reach the entrance of Viridian City." |
| 4 | 57.14% | 32 | 64 | 3 | "Move towards the edge of Route 1 to find the transition to the next area." |
| 5 | 57.14% | **30** | 82 | 3 | "Defeat the Pidgey in battle using available moves." |

Cache growth comparison across stages:

| | Stage M (FLAT) | Stage N+O (NEUTRAL+) | **Stage P (FLAT)** |
|---|---:|---:|---:|
| End-iter-1 procs | 1 | ~7 | **12** |
| End-iter-5 procs | 4 | 26 | **30** |

## What the hint actually did

**Goal-awareness:** The planner started naming Viridian City as the destination by iter 2. Compare to Stage N+O where the planner kept naming "Pallet Town" as the next move even after reaching Route 1 — the hint visibly nudged the planner's intent.

**Map-graph hint fired 1,406 times across 5 iters**, averaging 281 fires per 300-step iter. Every step where `current_map` was in MAP_GRAPH, the planner saw `### Map graph\nUnvisited maps reachable from <current>: ...` prepended to its observation.

**But movement stayed stuck.** Iter 5 made it as far as a wild Pidgey encounter on Route 1 (the gauntlet north of Viridian's south gate) — the cache picked up battle policies — but still 0 Viridian dwell. Iters 1–4 hit Route 1 (10, 35) and milled around the grid without ever finding the actual map-edge transition tile.

## Why FLAT (the bottleneck this surfaced)

Stage P confirms the diagnosis's caveat verbatim: *"It still might not work — the LLM might still fail to find the right exit tile. But it shifts the failure mode from 'no information' to 'information present but unusable', which is more diagnostically tractable."*

That's exactly what happened. The planner knows it should go to Viridian. It does not know **which tile to walk to**. Map-name-level adjacency is the wrong granularity for a movement bottleneck.

The Route 1 → Viridian transition is an **outdoor connection** — `headers/Route1.asm` says `connection north, ViridianCity, VIRIDIAN_CITY, -5` — meaning the agent has to walk off the *exact* north-edge tile, which is not at any of the (10, 35) waypoints the agent settled into. The planner's `move_to(x, y)` tool needs a coordinate; the hint gave it a map name.

## Cross-stage M5 ceiling — now diagnostically tight

| Stage | Mechanism | Verdict | Viridian dwell |
|---|---|---|---:|
| K (cumulative memory) | n/a | FLAT | 0 |
| L (map_aware) | static spatial summary | FLAT | 0 |
| M (multi-signal selector) | logprob + state-delta confidence | FLAT | 0 |
| N+O (bootstrap-neutral + state-delta acquisition) | better proc cache | NEUTRAL+ | 0 |
| **P (map-graph hint)** | **planner knows Viridian exists** | **FLAT** | **0** |

Five distinct interventions across the planner, selector, and acquisition layers — none move M5. The remaining bottleneck is mechanically clear: **the agent needs the (x, y) of the exit tile, not the name of the destination map**.

## Recommended next move: exit-tile coordinates

PR #90's "Out-of-scope follow-ups" already sketches the implementation:

- **Indoor warps** (doors/stairs): `objects/<Map>.asm` `warp_event x, y, TARGET, _` carries the exact tile. Hint becomes: *"To reach RedsHouse1f from PalletTown: stand on (5, 5) and walk in."*
- **Outdoor connections** (the M5 unblock): `headers/<Map>.asm` `connection north, ...` only gives direction. Combine with the connection's `bigdw` offset (the `-5` in `connection north, ViridianCity, VIRIDIAN_CITY, -5`) to compute the exact column where the agent crosses the north edge. Hint becomes: *"To reach ViridianCity from Route1: walk to the north edge near column N."*

The extractor for both already exists in this PR — `agents/macla/pokered_map_extractor.py: build_exit_tiles()` returns `(src, dst) → (x, y) | direction`. The runtime swap is to render those into the hint instead of (or alongside) the bare map name.

## Code that landed on this PR

Three commits on `feat/macla-stage-p-map-graph`:

| SHA | Subject | Runtime impact |
|---|---|---|
| `3be9857` | Stage P map-graph hint (core code + 11 tests) | **In flight during sweep** |
| `332cc90` | Auto-extract MAP_GRAPH + exit tiles from pokered .asm (+13 tests) | Additive (extractor + tests only; not wired) |
| `27e7f14` | Generalisable graph_hint via per-game adapter (+6 tests) | Additive (parity test guarantees byte-identical strings; not wired) |

The two additive commits sit ready for the exit-tile follow-up:
1. `pokered_map_extractor.build_exit_tiles()` produces the exit-coords dict from the same .asm parse as the auto graph.
2. `pokemon_red.game_adapter.graph_hint()` is the per-game seam where the new exit-aware hint will plug in.

## Verdict and PR disposition

- **Verdict:** FLAT (locked at 57.14%, σ=0, 0 Viridian dwell).
- **PR #90:** keep open with this writeup linked. The map-graph hint is cheap (one log line + one preprocessor call), strictly informative, and now compositionally needed by the exit-tile follow-up (you can't render an exit-tile hint without knowing the map graph).
- **Next stage:** Stage Q — exit-tile coordinates hint, building on Stage P's hint shape and the auto-extracted `build_exit_tiles()` dict.
- **Cross-stage chart:** flip Stage P from PENDING to FLAT (57.14% / 71.43% bars same as M / N+O); add Stage Q PENDING row.
