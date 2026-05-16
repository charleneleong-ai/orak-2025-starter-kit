# Stage M (multi-signal procedure quality + novelty + logprob) — n=5 cumulative-memory rerun

**Verdict:** FLAT-on-score, but the trajectory introspection reveals a **deeper problem the selector redesigns cannot reach**: across K → L → M, MACLA's procedural memory has plateaued at 4 procedures, all rooted at one OaksLab tile (the Charmander nickname dialog). Selector signal tuning is tuning a cache that does not grow.

**Closed:** 2026-05-16 03:45Z (~4h36m wall-clock, 5 iters × ~55min)
**Branch:** `feat/macla-multi-signal-quality`
**Worktree:** `/workspace/orak-stage-m`
**Log:** `logs/stage_m_v2_20260515T230924Z.log`

## Hypothesis (entering Stage M)

Stage L (PR #85, NEUTRAL+) confirmed map-aware procedure keys + iter-TTL bank M4 monotonically (259 → 229 → 172 → 140) but no iter crossed the M5 (Viridian) gate. The remaining ceiling was hypothesised to be in **action-quality signal** past M4. Stage M added three multiplicative signals to expected utility:

- **(a) State-delta confidence** — fraction of past `success_contexts` where the salient game state (score/hp/pos/map/in_battle/board/minerals/gas/supply/lives) moved forward. Bootstrap 0.5.
- **(b) Map-novelty θ-bump** — raise effective `theta_conf` to `max(theta_conf, 0.6)` on first visit to a map so cached procs rarely fire and the LLM is forced to explore.
- **(c) Logprob percentile-rank** — store `mean_logprob` per procedure, rank against rolling `deque(maxlen=50)`. Bootstrap 0.5.

**Minimum bar:** `late_mean >= early_mean` (no negative transfer) AND match Stage L M4 banking by iter 5 (≤ 140 steps).
**Lift bar:** any iter reaches Viridian OR scores > 57.14%.

## Schedule

| Setting | Value |
|---|---|
| Game | pokemon_red |
| Agent | gemma_26b (Gemma 4-26B-A4B-AWQ-4bit on vLLM :8000) |
| Max steps / iter | 300 |
| Iters | 5 (cumulative via `--load-checkpoint --prev-run-id`) |
| Launcher | `experiments/stage_m_multi_signal/run_pokemon_n5.sh` |

## Results

```
scores=[57.14, 57.14, 28.57, 57.14, 57.14]
mean=51.43% std=12.78pp
early_mean(iter 1-2)=57.14%, late_mean(iter 4-5)=57.14%, learning_delta=+0.00pp
```

| iter | score | M4 step | Δ vs iter 1 | Route1 steps | Viridian steps | final map | persev % |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 57.14% | **92** | — | 76 | 0 | Route1 | 21.0 |
| 2 | 57.14% | 140 | +48 | 28 | 0 | PalletTown | 19.2 |
| 3 | **28.57%** | n/a | n/a | 0 | 0 | PalletTown | 24.9 |
| 4 | 57.14% | 122 | +30 | 0 | 0 | PalletTown | 21.2 |
| 5 | 57.14% | 191 | +99 | 53 | 0 | Route1 | 24.1 |

Numbers from `experiments/stage_m_multi_signal/introspect.py`.

## Stage L vs Stage M — score-level comparison (identical)

| | Stage L (PR #85) | Stage M (this PR) |
|---|---|---|
| Scores | `[57.14, 57.14, 57.14, 28.57, 57.14]` | `[57.14, 57.14, 28.57, 57.14, 57.14]` |
| Mean ± std | 51.43% ± 12.78pp | **51.43% ± 12.78pp** (identical) |
| Iter ever past M4 (score > 4) | 0 of 5 | 0 of 5 |
| Viridian entered | 0 of 5 | 0 of 5 |
| 28.57% dip iter | 4 (OaksLab loop) | 3 (PalletTown wander) |

## Trajectory introspection — what actually happened

### 1. Iter-1 M4 speedup is real but localized to one map transition

| map | Stage L iter-1 dwell | Stage M iter-1 dwell |
|---|---:|---:|
| RedsHouse2f | 21 | 19 |
| RedsHouse1f | 15 | 14 |
| PalletTown | 161 | **115** (−46) |
| OaksLab | 102 | **76** (−26) |
| Route1 | 1 | **76** (+75) |

Stage M escaped PalletTown 46 steps faster and OaksLab 26 steps faster on the *first* pass. Both maps were entered at the same step (~34 / ~46), but the agent moved through them more decisively. This is the 92-step M4 vs Stage L's 259-step M4.

**Note:** iter 1 has no cumulative memory to inherit. Neither the state-delta nor the logprob signal had observations to score against. The novelty θ-bump fired 0 times during this iter (see §4). So the iter-1 speedup is **not** attributable to any of the three new signals. Most likely sampling variance from the LLM's stochastic subtask planner; n=1 vs n=1 comparison.

### 2. The iter-1 advantage does not compound — and we now know why

M4 banking trajectory:

| iter | Stage L steps to M4 | Stage M steps to M4 |
|---:|---:|---:|
| 1 | 259 | **92** |
| 2 | 229 (−30) | 140 (+48) |
| 3 | 172 (−87) | n/a (dip) |
| 4 | n/a (dip) | 122 (+30) |
| 5 | 140 (−119) | 191 (+99) |

Stage L gets steadily faster (259 → 140). Stage M is non-monotonic and slower by iter 5 (191 > 140).

**Cause:** the procedure cache is not growing where it would need to.

| iter | L procs learned | L successful execs | L procs refined | M procs learned | M successful execs | M procs refined |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 5 | 1 | 0 | 0 |
| 2 | 2 | 2 | 44 | 2 | 0 | 0 |
| 3 | 3 | 3 | 121 | 2 | 0 | 0 |
| 4 | 3 | 3 | 211 | 3 | 0 | 0 |
| 5 | 4 | **4** | **289** | 4 | **1** | **13** |

Stage M had **22× less procedure refinement than Stage L (13 vs 289)** and only **1 successful execution vs Stage L's 4**. The cumulative cache is functionally cold in Stage M — procedures exist in memory but they rarely fire and rarely refine.

At the same time, the selector firing rates are similar:

| | Stage L | Stage M |
|---|---:|---:|
| Selector events | 507 | 420 |
| Fired (eu ≥ θ) | 18 (3.6%) | 10 (2.4%) |
| Mean EU | 0.0119 | 0.0185 |
| Max EU | 0.270 | 0.131 |

Stage M's mean EU is *higher* (0.0185 vs 0.0119), so the multiplicative damping doesn't broadly suppress the cache. It is the **max EU** that collapses (0.131 vs 0.270): the highest-confidence procedure in Stage L pulled an EU of 0.27, but in Stage M no single procedure ever crossed 0.13. The damping flattens the top of the distribution, robbing the cache of its highest-confidence wins.

### 3. The iter-3 dip is a different failure mode than Stage L's iter-4 dip

| | Stage L iter 4 (28.57%) | Stage M iter 3 (28.57%) |
|---|---:|---:|
| OaksLab dwell | **171** | 63 |
| PalletTown dwell | 25 | **162** |
| BluesHouse dwell | 0 | **37** |
| RedsHouse1f dwell | 82 | 31 |
| Reached Route1 | no | no |
| Dominant action | `continue_dialog` x lots | `interact_with_object` to NPCs + signs |

Stage L's dip = stuck inside OaksLab doing dialog loops. Stage M's dip = wandered into PalletTown side-NPCs (8× SCIENTIST, 8× FISHER, 7× DAISY, 6× MOM, 6× POKEDEX, 6× SIGN, 5× RIVALSHOUSE_SIGN) and the unrelated `BluesHouse` side-room. This **looks like** the novelty effect we wanted — except it never fired through the selector (§4), so the exploration came from the LLM planner itself.

### 4. The novelty θ-bump never fired across 420 selector events

`grep "new_map=True" logs/stage_m_v2_*.log` → **0 hits.** `new_map=False` → **420 hits.**

Why: the procedure cache is dominated by procs whose preconditions are `map=OaksLab, position=(6,4), facing=up` — the Charmander nickname dialog. When the agent is *in* OaksLab, the cache fires (or fails to fire); when the agent is in PalletTown / Route1 / RedsHouse, `select_procedure` finds zero candidates matching the current map+position and returns early, never reaching the new-map check. By the time the agent reaches a new map with candidate procs, the new map is OaksLab — already visited on iter 1.

The novelty bonus was designed to suppress *cached procs* in new maps. There are no cached procs in new maps to suppress. The bonus is effectively dead code.

## Why this is FLAT-on-score with a structural finding

The score numbers say zero lift. The trajectory numbers say something more important:

**MACLA's procedural memory is functionally limited to one OaksLab tile.** Stage K (PR #75/#81), Stage L (PR #85), and Stage M (this PR) have all tuned the selector signal on a 1–4 procedure cache that lives entirely in OaksLab's nickname dialog. The map-aware key (L) made the OaksLab procedures fire more cleanly, which delivered monotonic M4 banking. The multi-signal damping (M) flattened the EU distribution and lost that gain. But **neither stage has produced procedural memory for any map past OaksLab**.

The 57.14% ceiling and the 28.57% periodic dip are both **agent-side behaviour outside the procedure cache**:

- **Ceiling at M4:** the LLM doesn't reliably take the Viridian entry tile. This is a planner / observation / battle-policy issue, not a procedure-selection issue. Three procedure-quality redesigns in a row have not moved this number.
- **Periodic 28.57% dip:** consistent ~20%-per-iter probability of stuck-state, but the stuck-state varies (OaksLab dialog in L, PalletTown NPC tour in M). Likely planner-side stochasticity, not cache-side.

## Implications for Stage N

**Stop tuning the selector.** Three stages of selector redesigns have produced identical means (51.43%) and identical ceilings (57.14% on 4 of 5 iters). The bottleneck is upstream.

Two distinct Stage N candidates emerge from the trajectory data:

1. **Procedure acquisition** — why is the cache stuck at 4 procs all at one tile? Possibilities: (a) the success-criterion for "procedure captured" is too strict and only fires for the deterministic dialog loop; (b) the map-aware key partitioned the cache so finely that no two trajectories share preconditions outside OaksLab; (c) the schema-based subtask planner doesn't produce stable repeatable action sequences that MACLA recognises as procedure candidates. This is the **substantive** finding from Stage M — fix this and procedure-quality signals start to matter.

2. **M5 gate replay** — independent of the cache work, replay the iters that reached the Route1/Viridian boundary (L iter 3: 114 Route1 steps; L iter 5: 12; M iter 1: 76; M iter 5: 53). Instrument the boundary tile transition. Ask: what does the LLM see at the boundary, and why does it never cross? This may be a planner / observation / battle-policy fix that is orthogonal to procedure learning.

Both are generalisable. (1) is the higher-leverage Stage N because it would unlock procedure-quality signals for any of the four games. (2) is targeted to pokemon_red but would directly move the score ceiling.

## Closed follow-ups from this PR

- **Multi-signal procedure-quality hypothesis** — falsified at the score level. Per-iter inspection shows the signals fire correctly but on too small a cache to matter.
- **Map-novelty θ-bump** — falsified as designed. 0 firings across 420 selector events. Procs don't exist in new maps to be suppressed; the bonus would need to apply at the LLM-planner level, not the procedure-selector level.
- **#46 Launch Stage M n=5** — closed by this writeup (verdict cron failed to fire post-compaction; landed manually).

## Out-of-scope for this PR

- **Investigate why procedural memory plateaus at 4 procs in OaksLab** (Stage N candidate 1, higher leverage).
- **Replay the four iters that touched Route1/Viridian and instrument the boundary tile** (Stage N candidate 2, score-focused).
- **`autoresearch.py` is not wired here** — Stage K/L/M have all been ad-hoc launchers. Worth migrating to a schedule yaml for the Stage N sweep so subsequent iters get a sweep tracker and PR updater for free (per CLAUDE.md ML sweep orchestration contract).
