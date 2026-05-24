# Generalized Agent Harness — MVA (Memory4 + Reflector)

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

| stage | scope | size | regression test |
|---|---|---|---|
| **PR 1** ✅ | Futile-action detector (universal pathology) | ~60 lines + 11 tests | mario @ 1000 — expect death-loops to drop, mean to lift from 9.04% |
| **PR 2** | Per-skill success-rate floor in selection | ~80 lines | mario re-run — kill-on-spawn proc gated by ~ep 8 |
| **PR 3** | Stagnation → skill demotion | ~60 lines | pokemon @ 1200 — expect stagnation=1081 to clear earlier |
| **PR 4** | `agent_events.jsonl` telemetry | ~80 lines | any run — verify event log contents |
| **PR 5** | Episode-end skill pruning | ~60 lines | 2048 @ 1000 — expect lift above 64% peak |
| **PR 6** | Episodic store + retrieval | ~200 lines (needs embeddings) | pokemon — does retrieval-augmented planning help? |
| **PR 7** | Reflector post-episode critique | ~250 lines | mario — does reflection rewrite bad procs? |
| **PR 8** | Self-model | ~100 lines | depends on PR 7 |

PRs 1-5 don't need embedding infra or extra LLM calls — cheap, ship in a week. PRs 6-8 need more setup.

### PR 1 in flight (futile-action detector)

Branch: `feat/futile-action-detector` (worktree `/workspace/orak-futile-detector`).
Implementation: agent-side hook in [`unified.py:_base_fallback`](../agents/macla/unified.py#L549). Hashes obs each step, fires when last K=3 consecutive obs are byte-identical, injects a one-line "your last K actions did nothing" hint into the planner prompt. Tests: 11/11 pass.

Test queue (after pokemon v4 finishes ~02:00 UTC 2026-05-24):
- pokemon @ 1200 → vs v3 baseline 6/7
- mario @ 1000 → vs baseline 9.04% mean / 21.85% best
- 2048 @ 1000 → vs baseline 45% mean / 64% best

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
