# Generalized Agent Harness — MVA (Memory4 + Reflector)

> **MVA** = **Minimum Viable Agent** — by analogy to MVP (Minimum Viable Product). The smallest agent architecture that can grow into the full long-horizon, self-evolving, embodied-reasoning agent envisioned at the top of this doc. Bootstraps from pokemon's existing scaffolds as data + extends to mario, 2048, and any future env/task via the layered contracts below.

**Last updated:** 2026-05-24

A successor architecture to the per-game MACLA scaffold. North star: a single agent that **excels at embodied reasoning for any task, plans long-horizon, executes, and self-evolves through interaction with its environment** — without hand-curated per-game milestone libraries, map graphs, or hint injectors.

This doc captures the **have vs need** audit and the staged build plan. Sister doc to [`architecture.md`](architecture.md) (which describes the current per-game architecture as-shipped).

> **Naming caveat:** while this work lives in the orak repo we keep current names (`GameAdapter`, `MilestoneSpec`, `_POKEMON_MILESTONE_LIBRARY`, `procedure`). Renaming happens at port time to the new repo at https://github.com/charleneleong-ai/tgaer. See [Naming convention](#naming-convention-orak-vs-tgaer) below.

---

## The vision in one paragraph

The current orak architecture is a per-game scaffold dressed up as a general framework: pokemon's `_POKEMON_MILESTONE_LIBRARY`, `NavigateToMap` bridges, exit-tile hints, and `map_graph_hint` are hand-curated for one game. Mario has none, 2048 has none, and any future task would need fresh scaffolding. The MVA replaces this with a **layered, self-extending agent** built on four contracts: a universal environment adapter, a four-store memory (`Memory4`), a planner with universal pathology guards, and a reflector that writes new skills/rules/prompts after every episode. The agent **bootstraps from data, not code** — pokemon's existing milestone library becomes one game's *initial skill set*, not the architectural unit.

---

## Architecture — five layers

```
┌─────────────────────────────────────────────────────────┐
│  L5  Reflector                                          │
│      Post-episode + post-stagnation LLM self-critique   │
│      → updates planner prompt, prunes/extends Memory4   │
├─────────────────────────────────────────────────────────┤
│  L4  Planner (LLM)                                      │
│      goal + obs + Memory4 retrievals → action / subgoal │
├─────────────────────────────────────────────────────────┤
│  L3  Universal pathology guards                         │
│      futile, loop, stagnation, regression detectors     │
│      → mutate planner prompt; emit events to Reflector  │
├─────────────────────────────────────────────────────────┤
│  L2  Memory4                                            │
│      ┌─ Episodic  — raw traces, retrievable by obs-sim  │
│      ├─ Procedural — skills, gated by success-rate      │
│      ├─ Semantic   — rules / invariants from reflector  │
│      └─ Self-model — capability map, updated per task   │
├─────────────────────────────────────────────────────────┤
│  L1  GameAdapter (will become EnvAdapter + Task)        │
│      obs(), actions(), step(), score(), goal_string()   │
└─────────────────────────────────────────────────────────┘
```

The two layers that matter most for the vision — **long-horizon planning** (L3 + L4 + L2.Procedural + L2.Semantic) and **self-evolution** (L5 + writes to all of L2) — are stacked: every layer feeds the one above and is updated by L5.

---

## Have vs need — by layer

The orak repo already ships ~70% of the *infrastructure* for the MVA — what's missing is composition, the failure-mode guards, and the L5 reflector. The table below maps every existing component to a layer and flags the gap.

### Layer 1 — GameAdapter (universal interface)

| component | have today | location | what's missing |
|---|---|---|---|
| Per-game env servers | ✅ | `evaluation_utils/mcp_game_servers/{pokemon_red,super_mario,twenty_fourty_eight,star_craft}` | unified `GameAdapter` Protocol — each game speaks a slightly different interface |
| `obs()` via gRPC | ✅ | `evaluation_utils/runner.py` | standardize obs shape across games |
| `step(action)` | ✅ | same | — |
| `score()` | ✅ | `evaluation_summary.json` | expose live, not just at episode-end |
| `is_done()` | ✅ | runner | — |
| `goal_string()` nat-lang | ❌ partial (in configs as `task_description`) | configs | **NEW**: structured method accessible to agent |
| `available_actions()` query | ❌ implicit per-game | adapter modules | **NEW**: discoverable action space |

### Layer 2 — Memory4

| store | have today | location | what's missing |
|---|---|---|---|
| **Episodic** | partial: `game_states.jsonl` written but not retrievable | rollout dirs | **NEW**: embedding index + retrieve-by-obs-similarity |
| **Procedural** (skills) | ✅ MACLA's `EnhancedHierarchicalMemorySystem` | [`agents/macla/macla_lib.py:325`](../agents/macla/macla_lib.py#L325) | **ADD**: success-rate floor gate, stagnation→demote hook, episode-end pruning |
| **Semantic (rules)** | ❌ absent | — | **NEW**: rule store ("when X, then Y") written by Reflector |
| **Self-model** | ❌ absent | — | **NEW**: `capability_id → competence` dict, updated by Reflector |

### Layer 3 — Pathology guards (cross-cutting)

| feature | have today | location | what's missing |
|---|---|---|---|
| `map_graph_hint` | ✅ pokemon-only | [`unified.py:622-630`](../agents/macla/unified.py#L622) | **GENERALIZE** as universal loop detector via obs-hash |
| `looped_positions_hint` | ✅ pokemon-only | [`unified.py:640-645`](../agents/macla/unified.py#L640) | **GENERALIZE** to any obs with positional or stateful signal |
| `subgoal escape valve` | ✅ pokemon-only | [`unified.py:687-690`](../agents/macla/unified.py#L687) | **GENERALIZE** as universal stagnation handler |
| **futile-action guard** | ❌ absent | — | **NEW (PR 1, in flight)**: obs-hash equality check, universal across all games |
| pathology events → trigger reflection | ❌ events logged but don't feed back | — | **NEW**: events → Reflector queue |

### Layer 4 — Planner

| feature | have today | location | what's missing |
|---|---|---|---|
| LLM-based planner | ✅ | `unified.py:_base_fallback` | — |
| subgoal decomposition | ✅ via `MilestoneSpec` + `requires_location` | `agents/macla/macla_lib.py` | **REFACTOR**: parameterize the `requires_location` field name (today pokemon-only) |
| Memory4 retrieval injection | partial (vector memory only) | `_base_fallback` | **EXTEND** to retrieve from Episodic + Semantic + Self-model stores |

### Layer 5 — Reflector (mostly missing)

| feature | have today | location | what's missing |
|---|---|---|---|
| episode-end hook | ✅ stub | [`unified.py:775` `record_episode_end`](../agents/macla/unified.py#L775) | **REWRITE**: currently only updates score; needs to invoke Reflector |
| trajectory storage | ✅ | `game_states.jsonl` | — |
| **post-episode LLM critique** | ❌ absent | — | **NEW**: LLM reads trace + outcome, emits rules/skills/prompt edits |
| **skill authoring** | ❌ absent (skills are hardcoded `MilestoneSpec`) | — | **NEW**: LLM proposes new `SkillSpec`, sandbox-tested before adoption |
| **prompt evolution** | ❌ absent (planner prompt is static) | — | **NEW**: Reflector rewrites system prompt based on failure patterns |
| **GEPA hook point** | ❌ absent | — | becomes a component inside Reflector |

### Cross-cutting infrastructure

| feature | have today | location | what's missing |
|---|---|---|---|
| per-rollout telemetry | ✅ wandb + weave + jsonl | rollout dirs + wandb projects | **EXTEND** with `agent_events.jsonl` (pathology fires, memory writes, reflector emits) |
| cross-rollout analysis | ❌ each rollout siloed | — | **NEW**: memory persists across rollouts (currently dropped) |
| checkpoint save/load | ✅ procedural only | [`agents/macla/base.py` `load_state/save_state`](../agents/macla/base.py) | **EXTEND** to all 4 Memory4 stores |

---

## Net new code estimate

If we list everything that doesn't exist today and ship it as a connected MVA:

1. `GameAdapter` Protocol — interface contract (~50 lines)
2. `SkillLibrary` with success-rate gate, stagnation demotion, ep-end pruning (~200 lines refactor of `EnhancedHierarchicalMemorySystem`)
3. Episodic memory with embedding retrieval (~150 lines)
4. Semantic rule store (~100 lines + retrieval LLM call)
5. Self-model (~50 lines)
6. Universal pathology detectors — futile, loop, stagnation, regression (~150 lines, generalizes existing pokemon hints)
7. Reflector — LLM-driven critique → memory updates (~300 lines)
8. Skill authoring sandbox — LLM writes `SkillSpec`, validate, adopt (~200 lines)
9. Cross-rollout memory persistence — load/save all 4 stores (~100 lines)
10. `agent_events.jsonl` telemetry stream (~50 lines)

**Total: ~1500 lines new + ~500 lines refactor.** Three weeks of focused work for the full MVA.

---

## What we keep from current orak work

- `evaluation_utils/mcp_game_servers/` per-game server infra — **unchanged**
- MACLA's procedural cache data model (`ProceduralMemoryEntry`) — **kept, with gating added**
- Pokemon's `_POKEMON_MILESTONE_LIBRARY` — becomes the **bootstrap state** for pokemon's L2 Procedural store, proves the interface
- All existing wandb/weave/jsonl logging — **extended, not replaced**
- Adaptive theta + EU selector — kept as the within-`SkillLibrary` selection mechanism
- Stage S work (`NavigateToMap` bridges, `requires_location` field) — **kept** as proof-of-concept of subgoal composition

## What we delete

- The pokemon-specific hints in `unified.py:_base_fallback` (`map_graph_hint`, `looped_positions_hint`, `subgoal_escape_valve`) — replaced by their universal equivalents reading from L2 memory
- The hardcoded coupling between `MilestoneSpec` and pokemon-specific fields like `requires_location` (parameterize the field name)
- `_POKEMON_MILESTONE_LIBRARY` as the *primary* abstraction — it becomes one game's *data*, not the architectural unit

---

## Staged build — ship in PRs

Building all five layers at once is suicide; stage by ROI:

| stage | scope | size | status (2026-05-24) | regression test |
|---|---|---|---|---|
| **PR 1** | Futile-action detector (universal pathology) | ~60 lines + 11 tests | ✅ **committed** @ [`176f68c`](https://github.com/charleneleong-ai/orak-2025-starter-kit/commit/176f68c), **3 rollouts queued** | mario @ 1000 — expect death-loops to drop, mean to lift from 9.04% |
| **PR 2** | Per-skill success-rate floor in selection | ~80 lines | pending | mario re-run — kill-on-spawn proc gated by ~ep 8 |
| **PR 3** | Stagnation → skill demotion | ~60 lines | pending | pokemon @ 1200 — expect stagnation=1081 to clear earlier |
| **PR 4** | `agent_events.jsonl` telemetry | ~80 lines | pending | any run — verify event log contents |
| **PR 5** | Episode-end skill pruning | ~60 lines | pending | 2048 @ 1000 — expect lift above 64% peak |
| **PR 6** | Episodic store + retrieval | ~200 lines (needs embeddings) | pending | pokemon — does retrieval-augmented planning help? |
| **PR 7** | Reflector post-episode critique | ~250 lines | pending | mario — does reflection rewrite bad procs? |
| **PR 8** | Self-model | ~100 lines | pending | depends on PR 7 |

PRs 1-5 don't need embedding infra or extra LLM calls — cheap, ship in a week. PRs 6-8 need more setup.

### PR 1 — futile-action detector (in flight, results pending)

**Branch:** `feat/futile-action-detector` (worktree `/workspace/orak-futile-detector`)
**Commit:** [`176f68c`](https://github.com/charleneleong-ai/orak-2025-starter-kit/commit/176f68c) — `feat(macla): universal futile-action detector (PR 1 of MVA harness)`

Implementation: agent-side hook in [`unified.py:_base_fallback`](../agents/macla/unified.py#L549).
- New constant `FUTILE_ACTION_WINDOW = 3` (top of `unified.py`)
- New method `_detect_futile_action(observation: str) → str | None` — hashes the planner-visible obs, tracks the last K hashes in a `deque(maxlen=K)`, fires when all K entries match → returns a one-line "your last actions did nothing" hint
- Wired into `_base_fallback` before per-game hint injection so hint-suffix changes don't artificially break the streak
- Reset in `record_episode_end` so short-episodic games (mario, 2048) don't carry the previous terminal frame into a new episode
- Streak-logging gate: one `[MACLA] futile_action_hint fired (...)` log line per consecutive futile streak, not once per step

Tests: 11/11 in `tests/test_futile_action_detector.py` — game-agnostic parametrization (pokemon obs, 2048 board, mario state string) plus state-machine sanity (window init, streak break, streak persist, post-clear reset, lazy init, log flag toggle).

**Live regression rollouts (launched 2026-05-24 01:37 UTC):**

| game | PID | run_id | budget | ETA | baseline to beat |
|---|---|---|---|---|---|
| pokemon | 1990348 | `futile_detector_pokemon_1200_20260524T013749Z` | 1200 steps | ~04:38 UTC | v3 = 6/7 |
| mario | 1991018 | `futile_detector_mario_1000_20260524T013809Z` | 1000 steps | ~01:51 UTC | 21.85% best / 9.04% mean |
| 2048 | 1992409 | `futile_detector_2048_1000_20260524T013826Z` | 1000 steps | ~04:08 UTC | 64% best / 45% mean |

All three share the vLLM server at port 8000 via continuous batching. Pokemon v4 (no detector, 2000 steps) still in flight under a separate PID — its outcome provides additional baseline context for the futile-detector pokemon run.

**Expected signal per game:**
- **mario** (largest expected delta): 60% of baseline episodes died from futile-loop game-overs. Detector should turn those into productive re-prompts → mean lifts toward the 21.85% peak.
- **2048**: episode 8 baseline died after 4 consecutive `down` actions on a `down`-blocked board. Detector should re-prompt at step 3, agent picks `left`/`right`/`up`, ceiling tile rises above 128.
- **pokemon**: subtler — stagnation=1081 in the ViridianCity loop. Detector fires when the agent literally hashes-equal observations (walking into a wall), not when it merely fails to make progress (legal-but-circular movement). Expect smaller delta unless the M6 wall is mostly wall-bumping.

---

## Regression test corpus

Live wandb runs that serve as MVA regression baselines:

| game | run | duration | best | mean | wandb |
|---|---|---|---|---|---|
| pokemon (v3 1200) | `step_budget_1200_baseline_20260523T171829Z` | 177 min | 6/7 (0.86) | — | `chaleong/orak-pokemon-red` |
| pokemon (v4 2000, in flight) | `step_budget_2000_baseline_20260523T210201Z` | ~5h | TBD | — | same |
| mario (1000) | `stage_s_super_mario_1000_20260523T210441Z` | ~13 min | 21.85% | 9.04% | `chaleong/orak-super-mario` |
| 2048 (1000) | `stage_s_2048_1000_20260523T210447Z` | ~2.4h | 63.64% | 44.92% | `chaleong/orak-2048` |

Each MVA PR posts a fresh run for the affected game(s), diff'd vs these baselines in the PR body.

---

## Naming convention (orak vs tgaer)

Until we port to the new repo, **keep all current orak names** so renames don't churn pokemon Stage S work. At port time, do a single find-replace pass:

| current (orak) | future (tgaer) | reason |
|---|---|---|
| `GameAdapter` | `EnvAdapter` + `Task` | split env-dynamics from task-objective; same env can host many tasks |
| `_POKEMON_MILESTONE_LIBRARY` | `SkillLibrary[task_id]` | per-task data, not per-game hardcode |
| `MilestoneSpec` | `SkillSpec` | "milestone" implies linear progression; skills compose |
| `Procedure` / `procedure_learned` | `Skill` / `skill_invoked` | "procedure" is RL-jargon |
| `EnhancedHierarchicalMemorySystem` | `SkillLibrary` | the L2 procedural store |
| `evaluation_utils/mcp_game_servers/` | `envs/` | clearer when we add non-game envs |
| `games:` config key | `tasks:` | tasks are what users select |

New code in orak should use **neutral names** that translate cleanly:
- `agents/macla/skill_library.py` (not "ProcedureCache") — wraps existing `EnhancedHierarchicalMemorySystem`
- `agents/macla/memory4.py` — Episodic + Semantic + Self-model stubs
- `agents/macla/reflector.py` — post-episode hook
- `agents/macla/pathology.py` — futile/loop/stagnation detectors

---

## Cross-refs

- Sister architecture doc: [`architecture.md`](architecture.md) (current per-game architecture as-shipped)
- Pokemon stage history: [`experiments/gemma/cross-stage-diagnosis.md`](experiments/gemma/cross-stage-diagnosis.md)
- 3-game MACLA findings: [`experiments/gemma/macla_findings.md`](experiments/gemma/macla_findings.md)
- Stage S openevolve writeup: [`experiments/openevolve_milestones/v1.md`](experiments/openevolve_milestones/v1.md) (in `feat/openevolve-milestones-spike` branch)
- Future repo target: https://github.com/charleneleong-ai/tgaer
