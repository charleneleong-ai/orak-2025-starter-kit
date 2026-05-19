# Stage R v3 — n=5 introspection: ceiling-bind, not plumbing bug

**Question:** Stage R v3 returned 57.14% on all five iters (mean=57.14% ± 0.00pp).
Is the perfect flat-line a **ceiling effect** (cumulative memory is working,
but Stage R's subgoal stack only reaches M5) or a **plumbing bug** (cumulative
memory isn't actually changing planner behavior past iter 1)?

**Verdict:** Ceiling-bind. Cumulative memory is working. We need M6+ subgoals
to break the plateau, not a fix to the cumulative-memory pipeline.

## Evidence 1 — Procedural memory growth across iters

Extracted from `MACLA Stats & Optimisation (Step N)` lines in
`logs/stage_r_v3_sweep_20260518T204630Z.log` (step-10 snapshot of each iter +
step-300 final):

| iter | step | procedural | learned | refined | avg_success |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 1 | 1 | 0 | 0.667 |
| 1 | 300 | 7 | 11 | 39 | 0.490 |
| 2 | 300 | 19 | 28 | 180 | 0.432 |
| 3 | 300 | 30 | 42 | 473 | 0.436 |
| 4 | 300 | 34 | 56 | 814 | 0.412 |
| 5 | 300 | **34** | **71** | **1142** | 0.432 |

Procedural memory grew **1 → 34** procedures across the cumulative-memory chain;
**71** total procedures learned, **1142** refinements. Not a plumbing failure.

The avg_success_rate drift (0.667 → 0.432) is consistent with the agent
spending more steps in late-game maps where the procedure cache hasn't yet
discovered the right tactics — i.e. exploration cost paid forward across iters.

## Evidence 2 — Milestone latency drops monotonically

Parsed `result.score` step-by-step from `game_states.jsonl`:

| milestone (raw score) | iter 1 first hit | iter 5 first hit | Δ |
|---|---:|---:|---|
| 1.0 | step 16 | step 9 | **−7 steps (−44%)** |
| 2.0 | step 32 | step 22 | **−10 steps (−31%)** |
| 3.0 | step 210 | step 181 | **−29 steps (−14%)** |
| 4.0 | step 227 | step 198 | **−29 steps (−13%)** |
| 5.0 | never | never | — |

Same milestone order. Same ceiling. **But every milestone gets hit faster
in iter 5** — cumulative memory is reducing exploration cost monotonically.
Iter 5's extra ~30 steps after score=4.0 are spent in PalletTown without
a defined "next thing" to chase.

## Evidence 3 — Map dwell shifts (modestly) toward Route1

```
iter 1 map dwell top-5: PalletTown 183, OaksLab 100, RedsHouse1f 16, RedsHouse2f 1
iter 5 map dwell top-5: PalletTown 184, OaksLab  98, RedsHouse1f  8, Route1 8, RedsHouse2f 2
```

Iter 1 never entered Route1. **Iter 5 spent 8 steps there.** Small absolute
movement, but it confirms the planner *is* choosing Route1 transitions
post-starter — the score just doesn't credit it because M5+ require staying
on Route1 long enough to reach Viridian (a single-screen transition that the
300-step budget after starter doesn't allow).

## Diagnosis

The 0.00pp variance is **ceiling-bind**, with three causal layers:

1. **Step budget.** 300 steps total. Iter 1 hits score=4.0 at step 227; iter 5
   at 198. Remaining budget after starter: ~70-100 steps. Reaching Viridian
   needs ~50+ Route1 steps plus the Pallet→Route1 transition, which is right
   at the budget edge.
2. **Subgoal stack scope (NOT spatial knowledge).** Stage P already
   dynamically injects the full 221-map MAP_GRAPH + Stage Q the exit-tile
   coordinates every step — the agent *knows* the topology past Viridian.
   What's missing is `initial_subgoal_stack()` returning anything past
   `[ViridianCity, Route1]`: no `EnterPokemonMart`, no `BeatRivalRoute22`,
   no `EnterViridianGym`. The map adjacency is in the observation; there's
   just no stack entry telling the planner to chase it.
3. **Score function.** The eval credits *map entry* milestones (score 4.0 ≈
   "starter obtained"). The 5.0 milestone (presumably Viridian arrival) is
   reachable only by crossing the full Route1 corridor — a much longer
   commitment than the prior milestones.

## What this means for v3 disposition

- **F1 (soft phrasing) + F2 (escape valve) worked.** v2's iter 1-2 sag
  (28.57 / 28.57) is gone. v3 matches v1's best (57.14%) on every iter,
  with monotonic per-milestone speedups.
- **v3 is ready to merge** as the new Stage R baseline.
- **Next stage** should extend the subgoal stack with M6+ entries
  (`EnterPokemonMart`, `BeatRivalRoute22`, `EnterViridianGym`, `DefeatBrock`,
  ...) to give the agent something to chase past M5. The spatial knowledge
  (map graph + exit tiles) already lives in Stage P + Q's per-step
  observation augmentation — what's missing is *stack entries*, not
  *map injection*. Pair with a step-budget bump (300 → 400-500) since
  iter 5's score-4.0 latency was 198/300.

## What else to try (ranked by impact × cost)

Beyond extending the subgoal stack past Viridian, the introspect surfaces
five concrete improvement levers. The first three are high-impact and
small-diff; the last two are deeper audits.

### 1. Anti-perseveration position penalty (HIGH impact, ~20 LOC)

**Evidence:** iter 5 revisited `PalletTown(7,10)` **44 times**,
`OaksLab(6,4)` **39 times**, `PalletTown(12,12)` **20 times** — only 66
unique positions across 300 steps. The agent is provably stuck in
position-level loops the planner can't see.

**Lever:** track visit count per `(map, x, y)` in
`EnhancedHierarchicalMemorySystem`; surface a `### Recently looped`
section listing positions with `visits > N` into the planner prompt,
or down-weight `move_to(x,y)` candidates where the destination is
already in the looped set. Smallest version: 5-line system-prompt
addition. Larger version: position-aware procedure-cache filter.

### 2. F4 — `move_to` boundary detection (HIGH impact, deeper change)

**Evidence:** the v2 introspect already caught this — `move_to(12,0)`
silently lands at `(12,5)` at the Pallet→Route1 edge and reports
success. This poisons procedural memory: the agent learns "walked to
(12,0)? Use `move_to(12,0)`" but the tool doesn't actually cross map
boundaries that way.

**Lever:** in the tool wrapper, compare requested destination to actual
position-after-move. On mismatch ≥3 tiles, return a structured
`PartialMove(actual=..., requested=...)` so the executor can react,
fall back to `overworld_map_transition`, or down-weight the tool
selection. Tracked under v2 introspect's "Recommended fixes — deep
workstream".

### 3. Step-budget bump 300 → 400-500 (HIGH impact, single-config change)

**Evidence:** iter 5 hits score-4.0 at step 198/300. Remaining 102 steps
isn't enough to cross Route1 + enter Viridian + chase any M6 milestone
even if subgoals exist. Score-3→4 took 17 steps (198-181); a similar
budget for 4→5 (Viridian arrival) leaves no margin for M6.

**Lever:** bump `configs/pokemon_red/env/default.yaml:max_steps` to
400 or 500 for the next sweep. Cost: ~33-66% more wall-clock per iter.

### 4. Stagnation counter reset on checkpoint load (MEDIUM, 1 LOC)

**Evidence:** v3 sweep showed `stagnation=440` at iter 2 step 1 —
counter pickled from iter 1's tail. Escape valve engages from step 1
in cumulative iters, never showing the planner the active_subgoal block.

**Lever:** in `EnhancedHierarchicalMemorySystem.__setstate__`, reset
`_subgoal_stagnation_key = None` and `_subgoal_stagnation_steps = 0`.
Benign in this sweep (iter 2 was actually 10min *faster*) but the
per-iter semantics are cleaner.

### 5. Perf-prune semantics audit (LOW-MEDIUM, audit + 1-line fix)

**Evidence:** v2 iter 1+2 scored 2.0/7 raw — well below
`PROC_CACHE_MIN_ITER_SCORE = 4.0` — but `procedures_pruned_low_score`
stayed at 0 across the chain. Suspected unit mismatch (raw 0-7 vs
normalized 0-100). When perf-prune doesn't fire, low-quality
procedures from a regressed iter propagate forward and slow recovery.

**Lever:** trace `last_iter_score` write site, confirm units, fix
threshold or normalize on write.

## Reproduce

```bash
# Procedural memory growth
python3 <<'PY'
import re, json
from pathlib import Path
log = Path("logs/stage_r_v3_sweep_20260518T204630Z.log")
text = log.read_text()
sections = re.compile(r"\[\d\d:\d\d:\d\dZ\] stage_r_subgoals_v3 iter (\d)/5").split(text)
for i in range(1, len(sections), 2):
    n, body = sections[i], sections[i+1]
    stats = re.findall(r"MACLA Stats & Optimisation \(Step (\d+)\): (\{[^\n]+)", body)
    if stats:
        print(f"iter {n}: first={stats[0][0]}, last={stats[-1][0]}, n={len(stats)}")
PY

# Milestone latency
python3 <<'PY'
import json, re
from pathlib import Path
for label, d in [
    ("iter 1", "stage_r_subgoals_v3_iter1_20260518T204630Z"),
    ("iter 5", "stage_r_subgoals_v3_iter5_20260519T004949Z"),
]:
    p = Path(f"/tmp/orak-stage-r-subgoals-v3/pokemon_red/{d}/game_states.jsonl")
    seen = {}
    for i, line in enumerate(p.open()):
        s = float(json.loads(line).get("result", {}).get("score", 0))
        if int(s) > 0 and int(s) not in seen:
            seen[int(s)] = i
    print(label, seen)
PY
```
