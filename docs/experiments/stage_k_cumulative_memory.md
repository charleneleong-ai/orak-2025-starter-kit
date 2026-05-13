# Stage J — Cumulative Cross-Episode Memory

**Status:** scaffolded, queued after Stage H finishes  •  **Branch:** `feat/stage-j-cumulative-memory`

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

Stage J tests one specific intervention: **inherit each iter's learned memory into the next iter**. The checkpoint system already supports this; the launcher just needs to wire it up.

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

The Stage J launcher chains them: iter N inherits iter N-1's checkpoint:

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
| **J (this writeup)** | **Cumulative memory iter→iter** | _pending_ |

Stage J shares Stage D's exact agent config — only the launcher differs (iter N inherits iter N-1's checkpoint).

## Run

After Stage H iter 3 finishes (~18:40Z), swap vLLM back to Gemma:

```bash
pkill -f 'vllm.entrypoints.openai.api_server'
nohup ./serving/gemma_serve.sh cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  >/tmp/gemma_serve.log 2>&1 & disown
until curl -s http://localhost:8000/v1/models | grep -qi 'gemma-4-26B-A4B'; do sleep 5; done

# Launch Stage J n=5
nohup bash experiments/stage_j_cumulative_memory/run_pokemon_cumulative.sh \
  >/tmp/stage_j_pokemon_n5.log 2>&1 & disown
```

Wall-clock estimate: ~50 min × 5 iters = **~4h 10min total**.

## Out of scope (for now)

- Cross-game memory transfer (pokemon procedures applied to mario/2048)
- Long-term persistent memory (nightly snapshot of best checkpoint to a versioned KB)
- Combined with Stage I (plateau deliberation) — natural next experiment if Stage J lifts
- Qwen-Hermes variant — defer to a follow-up if Gemma cumulative shows positive learning delta
