# Stage K — Cumulative Cross-Episode Memory

**Status:** completed n=5 — **REGRESS verdict** (learning_delta = −14.28pp)  •  **Branch:** `feat/cumulative-memory`  •  **PR:** [#75](https://github.com/charleneleong84/orak-2025-starter-kit/pull/75)

## Result (2026-05-14)

| Metric | Value |
|---|---|
| Per-iter scores | `[57.14, 57.14, 42.86, 28.57, 57.14]` |
| Mean ± std | **48.57% ± 12.78pp** |
| Min / Max | 28.57% / 57.14% |
| Early mean (iters 1-2) | 57.14% |
| Late mean (iters 4-5) | 42.86% |
| **learning_delta** | **−14.28pp** → **REGRESS** |
| vs Stage D 57.14% pure | **−8.57pp** |
| vs Stage H 57.14% Qwen | **−8.57pp** |

**Reading:** iters 3 and 4 collapsed below the universal 57.14% ceiling — the inherited memory actively *hurt* mid-sweep. Iter 5 recovered to ceiling but did not exceed it; no iter broke 57.14%. The 4/7 milestones still bank from in-town actions, and cumulative memory carryover did not unlock milestone 5+.

**Falsified hypothesis:** "Each iter inheriting `EnhancedHierarchicalMemorySystem` lifts late iters above the 57.14% ceiling." Memory carryover at the agent state level (procedures + atomic + vmem pickled iter→iter) is not the missing piece.

**Most likely interpretation:** chained checkpoint loading compounds noisy/wrong procedures faster than it compounds useful ones — the regress signature (`[57, 57, 42, 28, 57]`) is consistent with bad-habit accumulation rather than i.i.d. variance. Iter 5's recovery to 57.14% suggests the agent occasionally "shakes off" the inherited state but doesn't reliably build on it.

### Post-hoc introspection (game_states.jsonl): the map is fine, the *procedural memory* is the problem

Comparison of milestone-bank steps and dominant action across the n=5 sweep:

| iter | final | M1 banked | M2 banked | M3 banked | M4 banked | top action (×count) | dominant map | inherited |
|---|---|---|---|---|---|---|---|---|
| 1 | 57.14% | step 30 (Pallet sign) | step 42 (OaksLab) | step 177 (OaksLab) | step 196 (OaksLab) | `continue_dialog` ×24 | OaksLab (176/300) | NONE |
| 2 | 57.14% | step 21 | step 28 | step 158 | step 173 | `continue_dialog` (most) | OaksLab (113/300) | iter1 |
| 3 | 42.86% | step 52 | step 56 | **step 297** (3 steps before timeout!) | never | — | **PalletTown (180/300)** | iter2 |
| 4 | **28.57%** | step 38 | step 55 | **never** | never | **`interact_with_object(SIGN_PALLETTOWN_SIGN)` ×19** | **PalletTown (171/300) + BluesHouse (31)** | iter3 |
| 5 | 57.14% | step 21 | step 40 | step 149 | step 164 | `continue_dialog` ×19 | OaksLab (126/300) | iter4 |

Three things this table reveals:

1. **The map and game adapter are *not* the issue.** Iters 1, 2, 5 visit the same OaksLab-dominant map sequence as past Stage A→H runs and hit 57.14% on the same dialogue chain. Iter 4 visits one new map (`BluesHouse`, 31 steps) that no previous iter touched, but `BluesHouse` is empty for the Stage-D milestone set — it's a *distraction*, not a broken map.

2. **The inherited procedural memory induced perseveration on the wrong action.** In iter 4, `interact_with_object("SIGN_PALLETTOWN_SIGN")` fires **19 times** — by far the top action. In iter 1 (no inheritance) the top action is `continue_dialog` ×24 (the M3/M4-triggering interaction with Oak's dialogue chain). The town sign banks M1 *once* in iter 1; iter 4's inherited "sign interaction was useful" procedure caused the agent to keep interacting with the sign at the expense of returning to OaksLab to bank M3/M4.

3. **Spatial drift away from the milestone-rich map.** Iters 3 and 4 spent ~60% of steps in PalletTown vs. ~28-42% for successful iters; iter 4 specifically left OaksLab after banking M2 at step 76, bounced Oak ↔ Pallet from step 76 → 250, then ventured into BluesHouse. The inherited memory weakened the "stay in OaksLab and complete the dialogue" prior.

**Conclusion:** the failure mode is **bad-habit compounding in the inherited `EnhancedHierarchicalMemorySystem`**, not anything wrong with the map / ROM / game adapter. The inherited procedures over-weight one-shot milestone triggers (sign interactions, building warps) and under-weight long-form dialogue chains — exactly the wrong bias for the OaksLab Pokedex/party sequence that gates milestones 3-4.

This points the next ablation toward **inherit-vmem-only** or **inherit-with-confidence-thresholded-procedures**, not toward debugging the map system.

**What this does NOT rule out:**
- Different memory subsets (vmem-only, procedures-only, atomic-only — the current Stage K inherits all three)
- Curation between iters (filter out low-confidence procedures before passing forward)
- Cross-game memory transfer
- Combining cumulative memory with Stage J thinking-mode (Stage J also REGRESS'd standalone at 28.57%, so this is a long shot)



## Hypothesis

Stages A→H ([diagnosis doc](gemma/cross-stage-diagnosis.md)) converge on **57.14%** (4/7 milestones) for pokemon at 300 steps across:
- 6 Gemma variants (Stage A→G action/procedure-layer interventions)
- 1 Qwen 3.5 35B-A3B-Int4 (Stage H), σ=0 across n=3 iters

Trajectory introspection ([`scripts/introspect_trajectory.py`](../../scripts/introspect_trajectory.py)) shows:
- The 4/7 milestones bank from **in-town** actions (starter, Pokedex, Mom dialogue, etc. — all reachable from Pallet Town + OaksLab)
- Milestone 5+ requires **leaving Pallet Town** → Viridian → Forest → Pewter Gym (Brock fight)
- Stage H iter 2 literally **never left OaksLab** in 226/300 steps, but still scored 4/7
- Self-reflection at step 149 of every iter says *"You are stuck in a movement loop..."* — **the agent diagnoses correctly but the action layer doesn't change strategy**

**Common feature of all Stage A→H attempts:** each iter starts **fresh**. No memory of previous attempts. Even though `EnhancedHierarchicalMemorySystem` (procedural + atomic memory) is built up within an episode, it's discarded at the start of the next iter.

Stage K tests one specific intervention: **inherit each iter's learned memory into the next iter**. The checkpoint system already supports this; the launcher just needs to wire it up.

## Mechanism

`run.py` already has the flags (used by autoresearch sweeps but not by Stage A–H launchers):

```bash
--load-checkpoint               # load latest checkpoint from <run_id>'s checkpoint dir
--prev-run-id <id>              # load from a DIFFERENT run's checkpoint dir instead
```

The checkpoint pickle contains:
```
agent_state:
  macla_memory:    EnhancedHierarchicalMemorySystem  ← procedures + atomic + vmem
  macla_stats:     dict
  step_count, last_score, prev_state_str, last_action
```

The Stage K launcher chains them: iter N inherits iter N-1's checkpoint:

```bash
prev_run_id=""
for iter in 1..5:
    cmd=(uv run python run.py -c gemma_26b ... --run-id $run_id)
    if [[ -n $prev_run_id ]]; then
        cmd+=(--load-checkpoint --prev-run-id $prev_run_id)
    fi
    "${cmd[@]}"
    prev_run_id=$run_id   # next iter inherits this one
```

## Falsification criteria (n=5 learning curve)

| Per-iter scores | Reading |
|---|---|
| `[57, 57, 57, 57, 57]` flat | Memory carryover doesn't help — procedures don't capture useful generalisation |
| `[57, 57, 71, 71, 86]` rising | **Cumulative memory IS the missing piece** — late iters break the ceiling |
| `[57, 71, 86, 86, 86]` quick saturation | Memory helps in 1-2 iters then saturates at higher ceiling |
| `[57, 28, 14, 0, 0]` crashing | Memory captures BAD habits — anti-learning |

The "learning delta" metric in the launcher = `mean(iters 4-5) - mean(iters 1-2)`. A value > +7pp = LIFT, ±7pp = FLAT, < -7pp = REGRESS.

## Why this is *general* (not pokemon-specific)

The mechanism is at the framework level:
- Game adapter only provides `evaluation_score` (already done for all 3 games)
- `EnhancedHierarchicalMemorySystem` is game-agnostic (lives in `agents/macla/macla_lib.py`)
- The checkpoint format is shared across games
- Same approach works on mario/2048 with no code changes — just point `--prev-run-id` at the previous mario iter

If Gemma pokemon shows lift → next: cross-game cumulative (mario inherits from previous mario, 2048 inherits from previous 2048).
If Gemma pokemon shows lift AND we test cross-game memory → ultimate test: does *pokemon* memory help on mario? (probably not, but a cleaner null result than today's literature has.)

## Comparison to Stages A–H

| Stage | What changed | Pokemon n=3 result |
|---|---|---|
| D pure (PR #31) | Stage D stack (vmem + planner + procedures) | 57.14% (n=1) |
| B' (PR #69) | Procedures OFF | 42.86% ± 14.29pp |
| G (PR #70) | Procedure-escape | 47.62% ± 16.49pp |
| H (this PR not landed) | Qwen 3.5 35B-A3B-Int4 | 57.14% × n=3 (σ=0) |
| **K (this writeup)** | **Cumulative memory iter→iter (n=5)** | **48.57% ± 12.78pp** (REGRESS, −8.57pp vs D) |

Stage K shares Stage D's exact agent config — only the launcher differs (iter N inherits iter N-1's checkpoint).

## Run

After Stage H iter 3 finishes (~18:40Z), swap vLLM back to Gemma:

```bash
pkill -f 'vllm.entrypoints.openai.api_server'
nohup ./serving/gemma_serve.sh cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  >/tmp/gemma_serve.log 2>&1 & disown
until curl -s http://localhost:8000/v1/models | grep -qi 'gemma-4-26B-A4B'; do sleep 5; done

# Launch Stage K n=5
nohup bash experiments/stage_k_cumulative_memory/run_pokemon_cumulative.sh \
  >/tmp/stage_j_pokemon_n5.log 2>&1 & disown
```

Wall-clock estimate: ~50 min × 5 iters = **~4h 10min total**.

## Out of scope (for now)

- Cross-game memory transfer (pokemon procedures applied to mario/2048)
- Long-term persistent memory (nightly snapshot of best checkpoint to a versioned KB)
- Combined with Stage I (plateau deliberation) — natural next experiment if Stage K lifts
- Qwen-Hermes variant — defer to a follow-up if Gemma cumulative shows positive learning delta

## Follow-ups (post-REGRESS)

The full inherit-everything design REGRESS'd. Before abandoning cumulative memory entirely, the natural ablations are:

1. **Inherit vmem only** (skip procedures + atomic) — tests whether procedures specifically carry bad habits
2. **Inherit procedures only** (skip vmem + atomic) — opposite ablation
3. **Curated inherit** (drop procedures with confidence below threshold before passing forward) — the simplest mitigation if the regress signal is "bad procedures compound"
4. **Re-baseline at n=5** — current Stage D/H baselines are n=1 and n=3; Stage K is the first n=5 we have, and the 28.57% in iter 4 may partly be noise the smaller samples couldn't see

These are not queued; they need an explicit decision about whether to keep mining the cumulative-memory direction or pivot.
