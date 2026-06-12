# TGAER — Toward General-Purpose Abstraction & Embodied Reasoning

> **TGAER** = **T**oward **G**eneral-Purpose **A**bstraction & **E**mbodied **R**easoning. A layered agent architecture for long-horizon, self-evolving, cross-medium reasoning — captured here as the in-build version that will land in the standalone repo at https://github.com/charleneleong-ai/tgaer. The two halves of the name map to the two halves of the stack: **General-Purpose Abstraction** = L2 Memory4 + L5 Reflector (what the agent *learns and stores*); **Embodied Reasoning** = L1 EnvAdapter + L3 Pathology guards + L4 Planner (what the agent *does in the world*). Bootstraps from pokemon's existing scaffolds as data + extends to mario, 2048, and any future env/task via the layered contracts below.

**Last updated:** 2026-06-12

> **Status update 2026-06-12 — L3 PRs 1–3 landed.** The "PR 1 (futile-action detector) in flight" status throughout this doc is stale: PR 1 merged as [#107](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/107) (2026-05-26, ships as a safety floor — no cross-game lift), the action-side repeated-plan sibling as [#109](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/109), and the progress-stagnation detector + procedure-aware hint enrichment as [#110](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/110) (both 2026-05-27). Episode-end retrospective credit assignment landed as [#114](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/114). Separately, the pokemon M5 ceiling was broken — M6 (OAK's PARCEL) banked live via the escape-valve exemption ([#122](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/122)); see the [Update 2026-06-12](architecture.md#update-2026-06-12--m5-ceiling-broken-l3-guards-landed) banner in `architecture.md`. Read the per-PR `in flight` statuses below against this note.

A successor architecture to the per-game MACLA scaffold. North star: a single agent that **excels at embodied reasoning for any task, plans long-horizon, executes, and self-evolves through interaction with its environment** — without hand-curated per-game milestone libraries, map graphs, or hint injectors.

This doc captures the **have vs need** audit and the staged build plan. Sister doc to [`architecture.md`](architecture.md) (which describes the current per-game architecture as-shipped).

> **Naming caveat:** while this work lives in the orak repo we keep current names (`GameAdapter`, `MilestoneSpec`, `_POKEMON_MILESTONE_LIBRARY`, `procedure`). Renaming happens at port time to the new repo at https://github.com/charleneleong-ai/tgaer. See [Naming convention](#naming-convention-orak-vs-tgaer) below.

> **Scope + scale (read this before any claim):** this work targets **Small Language Models (SLMs) that fit on a single A100 40GB** — currently Gemma-4 26B-A4B-it (4B active, 26B total MoE) served via vLLM, with Qwen 7-14B / 30-35B-quantized as alternatives. All findings here may differ from frontier-API-model results (Claude Opus, GPT-5.2, Gemini 3) — relevant prior work tested at frontier scale is flagged explicitly in [Prior work — credibility-weighted](#prior-work--credibility-weighted-code--stars--scale). The primary benchmark surface is **[Orak (KRAFTON, 2025)](https://github.com/krafton-ai/Orak)** (144★) — `orak-2025-starter-kit` is the official starter kit. This repo is a leaderboard submission, not a benchmark invention.

> **Relationship to upstream Orak:** the cross-game agent framing (pokemon + super_mario + 2048 + starcraft) is the [Orak benchmark](https://huggingface.co/papers/2506.03610)'s design, not ours — published June 2025 by KRAFTON + Seoul National U + NVIDIA + UW-Madison. Our contribution is the agent stack ON Orak, not the benchmark itself.

---

## The vision in one paragraph

The current orak architecture is a per-game scaffold dressed up as a general framework: pokemon's `_POKEMON_MILESTONE_LIBRARY`, `NavigateToMap` bridges, exit-tile hints, and `map_graph_hint` are hand-curated for one game. Mario has none, 2048 has none, and any future task would need fresh scaffolding. The TGAER replaces this with a **layered, self-extending agent** built on four contracts: a universal environment adapter, a four-store memory (`Memory4`), a planner with universal pathology guards, and a reflector that writes new skills/rules/prompts after every episode. The agent **bootstraps from data, not code** — pokemon's existing milestone library becomes one game's *initial skill set*, not the architectural unit.

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

The orak repo already ships ~70% of the *infrastructure* for the TGAER — what's missing is composition, the failure-mode guards, and the L5 reflector. The table below maps every existing component to a layer and flags the gap.

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

If we list everything that doesn't exist today and ship it as a connected TGAER:

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

**Total: ~1500 lines new + ~500 lines refactor.** Three weeks of focused work for the full TGAER.

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
| **PR 1** | Futile-action detector (universal pathology) | ~60 lines + 11 tests | ✅ **committed** @ [`eda10f0`](https://github.com/charleneleong-ai/orak-2025-starter-kit/commit/eda10f0), **3 rollouts queued** | mario @ 1000 — expect death-loops to drop, mean to lift from 9.04% |
| **PR 2** | Per-skill success-rate floor in selection | ~80 lines | pending | mario re-run — kill-on-spawn proc gated by ~ep 8 |
| **PR 3** | Stagnation → skill demotion | ~60 lines | pending | pokemon @ 1200 — expect stagnation=1081 to clear earlier |
| **PR 4** | `agent_events.jsonl` telemetry | ~80 lines | pending | any run — verify event log contents |
| **PR 5** | Episode-end skill pruning | ~60 lines | pending | 2048 @ 1000 — expect lift above 64% peak |
| **PR 6** | Episodic store + retrieval | ~200 lines (needs embeddings) | pending | pokemon — does retrieval-augmented planning help? |
| **PR 7** | Reflector post-episode critique | ~250 lines | pending | mario — does reflection rewrite bad procs? |
| **PR 8** | Self-model | ~100 lines | pending | depends on PR 7 |

PRs 1-5 don't need embedding infra or extra LLM calls — cheap, ship in a week. PRs 6-8 need more setup.

> **PRs are how components land into the arch, not optional add-ons.** Each PR in the table below adds (or upgrades) a specific layer component listed in the "Have vs need" tables above. Once PR 1 merges, the futile-action detector is a permanent L3 component — the "PR" framing reflects shipping cadence, not provisional status. The regression rollouts are the merge gate, not a "maybe we'll add this if it works" filter.

### PR 1 — futile-action detector (component of L3, in flight)

**Branch:** `feat/futile-action-detector` (worktree `/workspace/orak-futile-detector`)
**Commit:** [`eda10f0`](https://github.com/charleneleong-ai/orak-2025-starter-kit/commit/eda10f0) — `feat(macla): universal futile-action detector (PR 1 of TGAER harness)`

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

Live wandb runs that serve as TGAER regression baselines:

| game | run | duration | best | mean | wandb |
|---|---|---|---|---|---|
| pokemon (v3 1200) | `step_budget_1200_baseline_20260523T171829Z` | 177 min | 6/7 (0.86) | — | `chaleong/orak-pokemon-red` |
| pokemon (v4 2000, in flight) | `step_budget_2000_baseline_20260523T210201Z` | ~5h | TBD | — | same |
| mario (1000) | `stage_s_super_mario_1000_20260523T210441Z` | ~13 min | 21.85% | 9.04% | `chaleong/orak-super-mario` |
| 2048 (1000) | `stage_s_2048_1000_20260523T210447Z` | ~2.4h | 63.64% | 44.92% | `chaleong/orak-2048` |

Each TGAER PR posts a fresh run for the affected game(s), diff'd vs these baselines in the PR body.

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

## References & influences

The TGAER is not novel architecture — it's a specific stack of patterns that have proven out in the academic and industry agent literature, applied to orak's cross-game test bed. Each layer maps to prior work we're building on. Where an orak experiment validated (or refuted) a pattern, that's cross-referenced too.

### Layer 1 — Env + Task split

- **[Gymnasium](https://gymnasium.farama.org/)** (Farama Foundation, 2022 — fork/successor of OpenAI Gym, Brockman et al. 2016) — the canonical `reset / step / observation_space / action_space` interface. Our `GameAdapter` is gymnasium-shaped.
- **[DeepMind dm_env](https://github.com/google-deepmind/dm_env)** (Muldal et al., 2019) — TimeStep abstraction that decouples env transitions from agent control loop. Influences our `StepResult` shape.
- **[MetaWorld](https://meta-world.github.io/)** (Yu et al. 2019, *CoRL*) — pioneered the env-vs-task separation: one robot arm env hosts 50 manipulation tasks. Our `EnvAdapter` + `Task` split is the same.
- **[BabyAI](https://github.com/mila-iqia/babyai)** (Chevalier-Boisvert et al. 2019, *ICLR*) — language-conditioned task hierarchy with `Mission` strings. Inspired `goal_string()` as a first-class adapter method.

### Layer 2 — Memory4

The four-store split (episodic / procedural / semantic / self-model) follows the cognitive architecture tradition explicitly:

- **[Generative Agents](https://arxiv.org/abs/2304.03442)** (Park et al. 2023, *UIST*) — three-store memory (observation stream, reflection tree, plan tree) with embedding-indexed retrieval and importance scoring. The Episodic + Semantic split mirrors this.
- **[Voyager](https://voyager.minedojo.org/)** (Wang et al. 2023, *NeurIPS*) — auto-extending **skill library** for Minecraft; LLM writes new skills, tests them via the env, adds to the library. The L2.Procedural store + skill-authoring sandbox (PR 8) are the Voyager pattern.
- **[ExpeL](https://arxiv.org/abs/2308.10144)** (Zhao et al. 2023, *AAAI*) — Experiential Learning: agent extracts cross-trial **insights** (semantic rules) from trajectory comparisons. Directly inspires L2.Semantic.
- **[CLIN](https://allenai.github.io/clin/)** (Majumder et al. 2023) — continually learning language agent with a structured **causal memory** ("X may be necessary for Y"). Another L2.Semantic precedent.
- **[STELLA](https://arxiv.org/abs/2404.01270)** (Liu et al. 2024) — self-evolving LLM agent with a dynamic tool/skill repository; influences the SkillLibrary success-rate gate (PR 2).

### Layer 3 — Universal pathology guards

- **[ReAct](https://arxiv.org/abs/2210.03629)** (Yao et al. 2022, *ICLR*) — interleaved thought-action-observation loop where the agent observes its own action effects. The futile-action detector is a deterministic version of ReAct's "if observation didn't change, reconsider" pattern.
- **[Reflexion](https://arxiv.org/abs/2303.11366)** (Shinn et al. 2023, *NeurIPS*) — verbal critique injected into next attempt. Our stagnation→reflector pipeline (PRs 3 + 7) is the Reflexion shape.
- **[LATS](https://arxiv.org/abs/2310.04406)** (Zhou et al. 2023) — Language Agent Tree Search; pathology guards as MCTS pruning signals.
- **[BALROG](https://arxiv.org/abs/2411.13543)** (Paglieri et al. 2024) — benchmark for long-horizon LM agents; their pathology taxonomy (loops, stagnation, regression) is the basis for our four guards (futile / loop / stagnation / regression).
- **Orak experiments** that motivated this layer's universality: 2026-05-23 cross-game baselines showed (a) mario 58 instant-death episodes after ep4 (procedure poisoning), (b) 2048 episode 8 dying after 4 consecutive `down` actions on a `down`-blocked board, (c) pokemon ViridianCity loop with stagnation=1290 — three games, three different shapes of the same universal pathology.

### Layer 4 — Planner

- **[ReAct](https://arxiv.org/abs/2210.03629)** (Yao et al. 2022) — base loop shape (referenced above)
- **[Plan-and-Solve](https://arxiv.org/abs/2305.04091)** (Wang et al. 2023, *ACL*) — decompose first, then act. Influences the subgoal decomposition currently in `MilestoneSpec`.
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)** (Yao et al. 2023, *NeurIPS*) — lookahead via LLM-imagined branches. Maps to optional L4 enhancement once Memory4 is stable.
- **MACLA** (orak, 2026 — Memory-Augmented Contextual Learning Agent) — orak's existing `UnifiedMaclaAgent` in [`agents/macla/unified.py`](../agents/macla/unified.py) IS the L4 planner, kept and refactored rather than replaced.

### Layer 5 — Reflector

- **[Reflexion](https://arxiv.org/abs/2303.11366)** (Shinn et al. 2023) — post-episode verbal critique. The canonical pattern for L5.
- **[GEPA](https://arxiv.org/abs/2507.19457)** (Agrawal et al. 2025) — Genetic-Pareto reflective prompt optimization; the right tool when the prompt itself is the surface to evolve. Slots into L5 as the prompt-evolution component.
- **[Self-Refine](https://arxiv.org/abs/2303.17651)** (Madaan et al. 2023, *NeurIPS*) — iterative critique-refine loop; precursor to Reflexion.
- **[STaR](https://arxiv.org/abs/2203.14465)** (Zelikman et al. 2022, *NeurIPS*) — self-taught reasoner; on-policy improvement via rationale generation. Conceptual basis for "reflector writes new skills/rules from successful trajectories."
- **[ExpeL](https://arxiv.org/abs/2308.10144)** (Zhao et al. 2023) — its insight-extraction step IS Reflector → L2.Semantic.

### Cross-cutting

- **Auto-skill writing**: [Voyager](https://voyager.minedojo.org/) (Wang et al. 2023), [Eureka](https://eureka-research.github.io/) (Ma et al. 2023, *ICLR* — LLM as reward designer/skill author), [SWE-agent](https://swe-agent.com/) (Yang et al. 2024, *NeurIPS*)
- **Telemetry / event streams**: [LangSmith](https://docs.smith.langchain.com/) (LangChain, 2023), [Weave](https://weave-docs.wandb.ai/) (W&B, 2024) — orak already uses both
- **Sweep orchestration / regression corpus**: [`experiments/autoresearch.py`](../experiments/autoresearch.py) (orak's own, 2026) — kept as the harness around the new arch

### What we are NOT claiming as novel (expanded after 2026-05-24 literature audit)

After a credibility-weighted literature scan (code + GitHub stars + scale alignment) on 2026-05-24:

- **Memory4 itself** — Generative Agents had three stores; we add Self-model. [G-Memory](https://github.com/bingreeky/GMemory) (234★) ships hierarchical multi-agent memory; [GAM](https://github.com/VectorSpaceLab/general-agentic-memory) (849★) ships JIT-compiled agent memory; [HiAgent](https://huggingface.co/papers/2408.09559) ships hierarchical working memory
- **Skill library auto-extension** — [Voyager](https://voyager.minedojo.org/), and at SLM scale [AgentEvolver](https://github.com/modelscope/AgentEvolver) (1,440★), [Agent0](https://github.com/aiming-lab/Agent0) (1,196★), [SAGE](https://github.com/amazon-science/SAGE) (13★) all do this with RL
- **Episodic retrieval** — RAG + Generative Agents + [Letta](https://www.letta.com/) + [mem0](https://mem0.ai/)
- **Verbal self-critique** — Reflexion + Self-Refine
- **Prompt evolution** — GEPA + OPRO
- **Cross-game LLM agent benchmark** — [Orak](https://github.com/krafton-ai/Orak) (144★) is published, has the 4 games, has a leaderboard — this is exactly the surface we're submitting to, not inventing
- **Skill efficacy measurement at frontier scale** — [SkillsBench](https://github.com/benchflow-ai/skillsbench) (1,209★) measured curated vs self-generated skills across 84 tasks × 11 domains at frontier-model scale (Claude Opus 4.6 / GPT-5.2 / Gemini 3 Flash); their headline finding "self-generated skills give −1.3pp" is the prior art for our claim #1
- **Memory operations via RL** — [Memory-R1](https://huggingface.co/papers/2508.19828) (paper-only, no code released) trains ADD/UPDATE/DELETE/NOOP via PPO/GRPO
- **Hierarchical memory for multi-agent** — [G-Memory](https://github.com/bingreeky/GMemory) (234★), [JoyAgent](https://huggingface.co/papers/2510.00510), [GAM](https://github.com/VectorSpaceLab/general-agentic-memory) (849★)
- **Self-evolution with RL** — [AgentEvolver](https://github.com/modelscope/AgentEvolver) (1,440★) + [Agent0](https://github.com/aiming-lab/Agent0) (1,196★) + [SAGE](https://github.com/amazon-science/SAGE) — combined >2,600 stars at SLM scale

### What IS (potentially) novel — credibility-weighted

After the audit, the original 8-claim list reduces to **1 dead, 2 narrower-defensible, 5 defensible** with the prior-work caveats below. Each claim is hedged against the highest-credibility collision found.

> **Methodology:** "novel" here means "no A-tier collision found" where A-tier = paper with released code + ≥100 GitHub stars + comparable scale (SLM-class for our work). Paper-only collisions (no code) are flagged as concurrent paper-only work, not as settled prior art.

1. **Bootstrap-vs-auto-emergence at SLM scale.** [SkillsBench](https://github.com/benchflow-ai/skillsbench) (1,209★) measured this for **frontier API models** on terminal/CS tasks and found self-generated skills don't help (−1.3pp). **Our narrower differentiator:** does that finding hold for SLMs (Gemma 26B-A4B-it / Qwen 7-14B class) on multi-step game environments where the skill compounds across episodes? Untested.

2. **Universal pathology event protocol.** [Diagnosing Failure Root Causes](https://huggingface.co/papers/2509.23735) covers a failure-root-cause taxonomy, but has no released code / minimal community signal. The unified L3 framing (futile + loop + stagnation + regression as a typed event stream routed to Reflector → Memory4) appears genuinely undone in the public-code agent literature. **Defensible.**

3. ~~**Game-shape-invariant Memory4.**~~ **DROPPED.** [Orak benchmark](https://github.com/krafton-ai/Orak) (144★, KRAFTON+SNU+NVIDIA+UW-Madison) is published exactly for cross-game LLM agent evaluation with 12 games incl. our 4. We're a submission, not the benchmark author.

4. **Procedure-cache hygiene as a phase transition.** [AgentPoison](https://huggingface.co/papers/2407.12784) (51 upvotes) and [SkillTrojan](https://huggingface.co/papers/2604.06811) cover **adversarial poisoning** (red-team threat model). Our observation is **benign degradation** during normal operation — different phenomenon, different threat model. **Defensible.**

5. **The dominance-lock-in metric.** No collision found at any credibility tier. The Gini-style portable diagnostic on procedure-selection distribution remains unmeasured. **Defensible.**

6. **Cross-game semantic transfer.** [MCMA](https://huggingface.co/papers/2601.07470) (Jan 2026, paper-only, no code) tests learnable memory abstraction with DPO on ALFWorld + ScienceWorld + BabyAI (not games). Differentiator survives: cross-game-class transfer via L2.Semantic on Orak game classes. **Defensible.**

7. **Self-evolution without weight updates, with budget accounting — at SLM scale.** [AgentEvolver](https://github.com/modelscope/AgentEvolver) (1,440★, Qwen 7-14B), [Agent0](https://github.com/aiming-lab/Agent0) (1,196★, Qwen3-8B), [SAGE](https://github.com/amazon-science/SAGE) (13★, Qwen 32B) all do self-evolution at SLM scale but all use RL/GRPO. **The frozen-model + pure-memory + budget-accounted control arm against those baselines is exactly the question they can't easily answer** (they bake RL in). Narrowed but defensible.

8. **Reflector → Memory4 write-schema ablation.** [Memory-R1](https://huggingface.co/papers/2508.19828) (paper-only, no released code, 152 training pairs) trains ADD/UPDATE/DELETE/NOOP via PPO/GRPO. Different study design (RL-training of memory operations vs ablation of which write-type matters most). **Defensible.**

### Engineering contributions

Separate from the research claims above, these are useful artifacts even if every research result above turns out negative:

- **Naming + repo split** — orak (validated patterns) → tgaer (clean abstractions): the architecture becomes portable infrastructure that other agent-research projects can adopt, not a one-off research artifact
- **Cross-game regression corpus on wandb** — pokemon (1200 / 2000) + mario (1000) + 2048 (1000) baselines as fixed reference points; any future agent change can be diff'd against these in one click
- **The pathology-event JSONL stream** (PR 4) — a stable telemetry format that survives architecture rewrites; lets others post-hoc analyze trajectories without re-running the LLM

## Benchmark landscape (audited 2026-05-24)

Star-weighted view of the public agent-evaluation surfaces. Used to pick comparison surfaces and Tier 4 targets above. Stars current as of 2026-05-24.

### Tier S — universal reference (1k+ stars or top HF adoption)

| benchmark | repo | stars | what |
|---|---|---|---|
| SWE-bench | [`SWE-bench/SWE-bench`](https://github.com/SWE-bench/SWE-bench) | **5,000★** | THE canonical coding-agent benchmark |
| AgentBench | [`THUDM/AgentBench`](https://github.com/THUDM/AgentBench) | **3,446★** | 8 env cross-task generalisation |
| OSWorld | [`xlang-ai/OSWorld`](https://github.com/xlang-ai/OSWorld) | **2,867★** | desktop / OS-level multimodal |
| WebArena | [`web-arena-x/webarena`](https://github.com/web-arena-x/webarena) | **1,480★** | browser tasks |
| τ-bench v1 | [`sierra-research/tau-bench`](https://github.com/sierra-research/tau-bench) | **1,239★** | tool-agent-user customer service |
| τ²-bench v2 | [`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench) | **1,219★** | telecom-domain extension of τ-bench |
| **GAIA** | [`huggingface.co/datasets/gaia-benchmark/GAIA`](https://huggingface.co/datasets/gaia-benchmark/GAIA) | **670 HF likes, 50K downloads/month** | general AI assistant — multimodality + tool use; Meta FAIR + HF + AutoGPT + GenAI |

### Tier A — strong adoption (250-1k stars)

| benchmark | repo | stars | what |
|---|---|---|---|
| Meta ARE / Gaia2 | [`facebookresearch/meta-agents-research-environments`](https://github.com/facebookresearch/meta-agents-research-environments) | **498★** | follow-up to GAIA, Meta FAIR backed |
| VisualWebArena | [`web-arena-x/visualwebarena`](https://github.com/web-arena-x/visualwebarena) | **476★** | multimodal browser tasks |
| ScienceWorld | [`allenai/ScienceWorld`](https://github.com/allenai/ScienceWorld) | **359★** | text-based science curriculum exploration |
| MemoryAgentBench | [`HUST-AI-HYZ/MemoryAgentBench`](https://github.com/HUST-AI-HYZ/MemoryAgentBench) | **341★** | ICLR 2026; 4-competency memory eval |
| BALROG | [`balrog-ai/BALROG`](https://github.com/balrog-ai/BALROG) | **254★** | cross-game LLM/VLM reasoning (NetHack/Crafter/etc.) |
| ARC-AGI-3 agent SDK | [`arcprize/ARC-AGI-3-Agents`](https://github.com/arcprize/ARC-AGI-3-Agents) | **247★** | Tier 4a target — $850K prize, SLM-forced |

### Tier B — niche / recent (50-250 stars)

| benchmark | repo | stars | what |
|---|---|---|---|
| **Orak** | [`krafton-ai/Orak`](https://github.com/krafton-ai/Orak) | **144★** | **PRIMARY — `orak-2025-starter-kit` is the official starter kit; we're on this** |

### Tier C — early-stage or paper-only

| benchmark | repo | stars | what |
|---|---|---|---|
| ARC-AGI toolkit | [`arcprize/arc-agi`](https://github.com/arcprize/arc-agi) | **48★** | ARC-AGI-1/2/3 unified Python interface |
| ARC-AGI-3 benchmarking | [`arcprize/arc-agi-3-benchmarking`](https://github.com/arcprize/arc-agi-3-benchmarking) | **18★** | benchmark agents helper |
| MemoryArena | [memoryarena.github.io](https://memoryarena.github.io/) | — | paper-only Feb 2026, no released code |

### ARC-AGI-3 — Tier S+ for our specific narrative

Not a star-based ranking — strategic alignment. The competition is the highest-leverage public artifact this work can produce:

| dimension | why it's the right target |
|---|---|
| Dated milestones | June 30 / Sept 30 / Nov 2 / Dec 4 (2026) — forces shipping cadence |
| SLM-forced by sandbox rules | No internet/API → most frontier-lab teams have to use their non-frontier models |
| Open-source mandatory | Public Kaggle leaderboard + your code visible |
| $700K grand prize + $150K milestones | Real competitive participation, not just academic |
| Returns to original TGAER framing | The April 2026 realignment had ARC-AGI-3 as the original target |

### Eval-rigor references (for honest variance reporting)

Cited for the methodology around our pokemon v3 (6/7 @ 1200 steps) vs v4 (5/7 @ 2000 steps) variance band:

- **[Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://huggingface.co/papers/2507.02825)** (Jul 2025) — Agentic Benchmark Checklist (ABC)
- **[On Randomness in Agentic Evals](https://huggingface.co/papers/2602.07150)** (Feb 2026) — pass^k metric for evaluation variance; pokemon v3/v4 variance is exactly the issue this paper warns about
- **[Claw-Eval](https://huggingface.co/papers/2604.06132)** (Apr 2026) — trajectory-aware grading + Pass@k / Pass^k
- **[Survey on Evaluation of LLM-based Agents](https://huggingface.co/papers/2503.16416)** (Mar 2025) — 97 upvotes; foundational survey

### Frontier-lab references (no public code but field-defining)

Frontier labs rarely release code for core research, but their public work defines the field. Cited as context regardless of code availability:

- **[Anthropic Claude plays Pokemon](https://www.anthropic.com/news/claude-plays-pokemon)** — publicly-streamed pokemon agent with memory tool + heavy scaffolding (Tier 4b informal target)
- **[Anthropic Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use)** (Oct 2024) — OS-level agent action protocol
- **[Anthropic Skills](https://www.anthropic.com/news/agent-skills)** — the public "skills" concept (curated procedural folders, exactly what SkillsBench tests)
- **[DeepMind SIMA](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/)** (2024) — generalist agent for 3D environments
- **[DeepMind Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)** (2024) — generative world model
- **[DeepMind AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)** (2024) — theorem proving via gradient + search hybrid
- **[OpenAI Operator](https://openai.com/index/introducing-operator/)** (2025) — computer-use agent
- **[Meta FAIR CICERO](https://ai.meta.com/research/cicero/)** (2022) — Diplomacy agent

## Research roadmap — extensions beyond the TGAER

The 8-PR TGAER is *necessary* groundwork. By itself it's a proof of architecture; it doesn't yet test the hardest open questions in the field. This section maps tiered extensions — what each tier adds technically, what published precedent it tests against, and what scope each tier represents.

### Field-level open problems the TGAER framework can address

After the 2026-05-24 literature audit (see [Prior work — credibility-weighted](#what-we-are-not-claiming-as-novel-expanded-after-2026-05-24-literature-audit)), the open problems where TGAER is well-positioned to contribute — **at SLM scale specifically**:

| open problem | why still open after the audit |
|---|---|
| **Skill efficacy at SLM scale on multi-step games** | [SkillsBench](https://github.com/benchflow-ai/skillsbench) (1,209★) answered "curated skills help, self-generated skills don't" — but **for frontier-API models on terminal/CS tasks**. The SLM (7-30B) answer on multi-step games where the skill compounds across episodes is open. |
| **Unified pathology taxonomy + handler** | Reflexion handles one type, LATS another, Self-Refine another. [Diagnosing Failure Root Causes](https://huggingface.co/papers/2509.23735) covers a taxonomy but minimal community signal; no released unification + handler. |
| **Pure-memory baseline against RL-trained self-evolution** | [AgentEvolver](https://github.com/modelscope/AgentEvolver) (1,440★) + [Agent0](https://github.com/aiming-lab/Agent0) (1,196★) + [SAGE](https://github.com/amazon-science/SAGE) all do self-evolution with GRPO at SLM scale. **The frozen-model + pure-memory control arm at the same scale isn't reported** — they bake RL in, so they can't easily isolate it. |
| **Cross-game-class transfer via memory at SLM scale** | [MCMA](https://huggingface.co/papers/2601.07470) tests cross-task on ALFWorld/ScienceWorld/BabyAI (not games, paper-only no code). Cross-game-class memory transfer at SLM scale on Orak's 12-game span is unmeasured. |
| **Dominance-lock-in as portable diagnostic** | No collision found at any credibility tier. The Gini-style coefficient on procedure-selection distribution as a portable diagnostic remains open. |
| **Self-Evolution Curve protocol** | All current agent benchmarks are one-shot. BALROG measures plateau, not learning rate. [On Randomness in Agentic Evals](https://huggingface.co/papers/2602.07150) (Feb 2026) proposes pass^k — moves in the right direction but not learning-rate-specific. |
| **Reflector write-schema ablation** | [Memory-R1](https://huggingface.co/papers/2508.19828) trains memory ops via RL but doesn't ablate which write-type contributes most lift. Paper-only, no released code. |

What the audit closed off (no longer open problems):

- ~~Cross-game LLM agent benchmark~~ — [Orak](https://github.com/krafton-ai/Orak) (144★, KRAFTON+SNU+NVIDIA+UW) ships this
- ~~Skill efficacy at frontier scale~~ — SkillsBench answered (curated +16pp, self-generated −1pp)
- ~~Memory operations as structured RL targets~~ — Memory-R1 covered the design even if no code released

### Tier 1 — TGAER + cross-game ablation (6-8 weeks)

Already planned in PRs 1-8 above. Tests the architecture on pokemon + mario + 2048.

### Tier 2 — extend to non-game tasks (4-6 additional weeks)

The TGAER framework only matters if it generalizes beyond games. Add `EnvAdapter` implementations for:

- **[WebArena](https://webarena.dev/)** (Zhou et al. 2024, *ICLR*) — browser tasks
- **[SWE-bench Verified](https://www.swebench.com/)** (Jimenez et al. 2024, *NeurIPS*) — code fix tasks
- **[OSWorld](https://os-world.github.io/)** (Xie et al. 2024, *NeurIPS*) — desktop / OS-level tasks

If the SAME Memory4 + Reflector reaches competitive scores on browser + code + games with NO task-specific scaffolds, that's the cross-task generalization claim. The contrast with task-specific SOTA agents (Devin for SWE-bench, Operator for web) makes the comparison concrete.

This is also where the **Self-Evolution Curve methodology** (open problem #1) becomes the publishable artifact. We define: run agent for N trials with no human reset, measure score-vs-trial. Publish leaderboards for every task using this protocol. Establish it as the standard for measuring agent *learning rate* rather than peak capability.

### Tier 3 — the agentic-RL × memory comparison (8-12 weeks, GPU-heavy)

The current TGAER is pure inference (frozen Gemma 26B served via vLLM). The natural alternative is **agentic RL / post-training**:

- **[GRPO](https://arxiv.org/abs/2402.03300)** (Shao et al. 2024 — DeepSeekMath / DeepSeek-R1) — group-relative policy optimization, no value model needed
- **[DPO](https://arxiv.org/abs/2305.18290)** (Rafailov et al. 2023, *NeurIPS*) and **[SimPO](https://arxiv.org/abs/2405.14734)** (Meng et al. 2024, *NeurIPS*) — preference pair learning from trajectories
- **[Process Reward Models](https://arxiv.org/abs/2305.20050)** (Lightman et al. 2024, *ICLR*) — step-level rewards
- **[Tülu 3](https://arxiv.org/abs/2411.15124)** (Lambert et al. 2024) — open-source RL post-training recipe

The repo already contains most of the GSPO infrastructure (offline GSPO gradient loop in `train.py` via Unsloth, gspo_group.json sidecars, paired-rollouts harness). The missing piece is wiring it to TGAER's collected trajectories.

The **four-way comparison** at a fixed compute budget:

| arm | model | memory | description |
|---|---|---|---|
| A. Pure inference | Gemma 26B frozen | none (vanilla MACLA) | current orak baseline |
| B. TGAER only | Gemma 26B frozen | full Memory4 + Reflector | what PRs 1-8 produce |
| C. GRPO only | Gemma 26B fine-tuned on trajectories | none | classic post-training |
| D. TGAER + GRPO | Gemma 26B fine-tuned | full Memory4 + Reflector | does memory + RL compound, substitute, or interfere? |

The "memory vs RL" debate is folklore-level today — Letta claims memory is enough, DeepSeek-R1 says RL is enough. A clean comparison with budget accounting (LLM calls, GPU hours, $ cost) settles whether they substitute, compound, or interfere.

### Tier 4 — ARC-AGI-3 competition submission (highest-leverage public artifact)

The **[ARC Prize 2026 — ARC-AGI-3 competition](https://arcprize.org/competitions/2026/arc-agi-3)** (launched March 25, 2026) is the right Tier 4 target — dated, public, prize-backed, and **SLM-forced by sandbox rules** (no internet, no API calls). The original TGAER framing per the April 2026 realignment was for ARC-AGI-3 anyway (see [`hermes/life/REALIGNMENT-2026-04-19.md`](file:///root/dotfiles/hermes/life/REALIGNMENT-2026-04-19.md)); the Orak cross-game work generalises that architecture, and Tier 4 takes it back to ARC.

| dimension | value |
|---|---|
| Total prizes | **$850K** ($700K grand prize + $150K milestone prizes) |
| Sandbox constraint | **No internet, no API calls** — forces local-model-only (perfect SLM fit) |
| Open-source requirement | Mandatory (CC0 or MIT-0) before prize evaluation |
| Milestone #1 deadline | **June 30, 2026** ($25K / $10K / $2.5K for top-3) |
| Milestone #2 deadline | September 30, 2026 (same prize tier) |
| Final deadline | November 2, 2026 |
| Results announced | December 4, 2026 |
| Public artifacts | [arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) (247★) — agent SDK, [arcprize/arc-agi-3-benchmarking](https://github.com/arcprize/arc-agi-3-benchmarking) (18★) — benchmark helper |
| Why uniquely good for SLMs | Most frontier-lab teams have to use their non-frontier models (no Claude/GPT/Gemini API available in sandbox) — removes their normal advantage |

**Strategic upside:** ARC-AGI-3 milestone #1 (June 30) is the earliest credible RE-portfolio public artifact this work can produce. Even a competitive partial submission gets a public Kaggle rank and forces the SLM architecture decisions into the open.

### Tier 4b — Anthropic Pokemon as informal-but-strategic surface

Separate from ARC-AGI-3, **Anthropic's [Claude-plays-Pokemon](https://www.anthropic.com/news/claude-plays-pokemon) stream remains the single most public agentic-LLM project** and is openly used by Anthropic as a recruiting funnel. Not a formal benchmark (no leaderboard, no prize, no SLM constraint), but strategically valuable for Anthropic-specific RE-track signalling.

| dimension | value |
|---|---|
| Format | publicly-streamed Claude rollouts with custom memory tool + heavy scaffolding |
| Best score (as of ~2026) | Cerulean City (M11+) via Claude 3.7 Sonnet + custom memory tool |
| Our angle | Match or beat with **26B SLM + auto-emerging skills** (no hand-curated milestone library beyond M5) → direct comparison: "How much of the gap was the model vs the scaffold?" |
| Recruiting upside | Anthropic openly uses the thread as a hiring signal; pings on the thread are visible to the agentic team |
| Risk | Anthropic keeps moving the goalposts (new scaffolding, new memory tools); chasing their public results is a moving target |

**Strategic call:** pursue ARC-AGI-3 (Tier 4a) as the *competitive* artifact (dated, prize-backed, SLM-forced, your rank is public) and Anthropic Pokemon (Tier 4b) as the *narrative* artifact (frame the 26B vs Claude comparison as a blog post + Twitter thread; tag the Anthropic team). They're complementary, not substitutes.

### Tier 4c — secondary public benchmark targets

Beyond ARC-AGI-3 and Anthropic Pokemon, these are cleaner cross-task surfaces but with less competitive urgency:

- **[SWE-bench Verified](https://github.com/SWE-bench/SWE-bench)** (5,000★) — the canonical coding-agent benchmark; current SOTA ~60-65% via Claude + custom agents. TGAER + Reflector on code-edit traces is a clean test.
- **[WebArena](https://github.com/web-arena-x/webarena)** (1,480★) — current SOTA ~35-45% success. Cross-task transfer surface.
- **[OSWorld](https://github.com/xlang-ai/OSWorld)** (2,867★) — desktop / OS-level tasks.
- **[τ²-bench](https://github.com/sierra-research/tau2-bench)** (1,219★) — customer-service / tool-use; applied-AI angle for Sierra/Decagon-style roles.
- **[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)** (670 HF likes, 50K downloads/month) — general AI assistant benchmark with multimodality + tool use.
- **[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench)** (341★, ICLR 2026) — direct surface for claim #6 (cross-game/cross-session memory).
- **[BALROG](https://github.com/balrog-ai/BALROG)** (254★) — agentic LLM/VLM reasoning on diverse games (NetHack, Crafter, BabaIsAI, etc.); cross-game pathology comparison.

### Tier 5 — tgaer as research infrastructure

The orak → tgaer repo split (see [naming convention](#naming-convention-orak-vs-tgaer) above) is operationalised by making tgaer the **reference implementation** of the TGAER:

- Pluggable model backends (vLLM, Anthropic API, OpenAI API, Gemini, local)
- Pluggable memory backends (in-process, Letta, mem0, custom)
- Pluggable env adapters (gymnasium-compatible)
- Documentation + examples to onboarding-in-a-day quality
- One canonical benchmark suite (Self-Evolution Curve protocol applied to 6+ envs)

## Future angles to explore (post-Tier-5)

Beyond the 8-PR TGAER + Tiers 1-5, exploratory directions worth flagging for future iteration. Ranked by strategic leverage (compound with current work × signal for RE-track roles × novelty after the 2026-05-24 audit).

### Top 8 directions

| # | angle | why it could matter |
|---|---|---|
| 1 | **MCP-native memory agent** | [Anthropic Skills](https://www.anthropic.com/news/agent-skills) + [Model Context Protocol](https://modelcontextprotocol.io/) are converging into the standard agent surface. TGAER could become the **reference open-source MCP-consumer with Memory4** — no clear claimant yet. Aligns 1:1 with Anthropic FDE role descriptions. |
| 2 | **Cost/latency Pareto study at SLM scale** | Extend the budget-accounting claim from `$`-only to `(score, $, latency, GPU-hours)`. Frontier-lab eng teams care obsessively; few public studies cover this for agents. Pure measurement, no new architecture. |
| 3 | **Adversarial robustness of skill libraries** | Reframe TGAER's success-rate floor (PR 2) + dominance-lock-in detection (claim #5) as **defenses** against [AgentPoison](https://huggingface.co/papers/2407.12784) / [SkillTrojan](https://huggingface.co/papers/2604.06811). Lands at safety-flavoured RE roles (Apollo, METR, AISI). |
| 4 | **Continual learning across weeks/months on Orak** | Most agent benchmarks are one-shot or one-session. Run TGAER on Orak across N sessions over weeks — does the agent measurably improve? This is **the Self-Evolution Curve methodology** as a standalone paper-shaped output. |
| 5 | **Vision-language TGAER** | Most agent work is text-only. Add VLM support (Pixtral, Llama-4-Scout, Qwen-VL) — directly applies to ARC-AGI-3 grids, [BALROG](https://github.com/balrog-ai/BALROG) VLM track, [OSWorld](https://github.com/xlang-ai/OSWorld) vision mode. Big surface, unclaimed at SLM scale. |
| 6 | **Composable LLM routing (SLM ↔ frontier)** | TGAER as the orchestrator — uses SLM by default, escalates to frontier API model when needed. Cost-optimal hybrid is genuinely novel as a measured study. |
| 7 | **Knowledge distillation from frontier traces** | Use Claude/GPT/Gemini to play Orak games via API, distill trajectories, train SLM on them. Hybrid of memory (their experience) + weight updates (your SFT). Connects RL track (Tier 3) to distillation literature. |
| 8 | **TGAER for AI4Sci automation** | [Anthropic Asta](https://www.anthropic.com/news/anthropic-deepens-research-collaborations), [DeepMind GNoME](https://deepmind.google/discover/blog/millions-of-new-materials-discovered-with-deep-learning/), [Coscientist](https://www.nature.com/articles/s41586-023-06792-0). Long-horizon + checkable success = TGAER fit. Bigger pivot but very high TAM. |

### Quick-fire honourable mentions

- **Empirical study of pathology event ordering** — does futile-first vs loop-first detection matter? Small ablation if claim #2 survives
- **Multi-agent TGAER extensions** — co-op or competitive play; connects to [CoMAS](https://huggingface.co/papers/2510.08529), [G-Memory](https://github.com/bingreeky/GMemory)
- **Skill provenance + auditability** — trace each skill back to creating episodes; alignment angle
- **Curriculum learning across games** — 2048 → mario → pokemon → starcraft, measure transfer
- **Pure-LLM "from-scratch" baseline (no memory, no scaffold)** — absolute floor; small experiment, big rhetorical value

### Recommended next-3 (if a fork in the road appears)

Picked to compound with current work + maximise RE-track signalling + maintain SLM/A100-40GB constraint:

1. **#1 MCP-native memory agent** — directly maps to Anthropic FDE postings, low extra code
2. **#4 Continual learning across sessions on Orak** — gives the Self-Evolution Curve a real data anchor, no new infra
3. **#3 Adversarial robustness reframing** — opens AISI/Apollo/METR roles (UK/remote-friendly per geo constraint)

## Efficient frontier — what helps with what?

The deeper question behind the TGAER work: **for generalizable self-evolving agents across many mediums (games, web, code, science, robotics), what's the most compute- and effort-efficient capability-lift mechanism at each task class?** The dominant lever shifts as you move along the task spectrum. This section maps the landscape honestly.

### Capability-lift mechanisms — ordered by cost and ceiling

Eleven mechanisms in current use, with their cost/ceiling/iteration-speed profile:

| # | mechanism | cost (per unit lift) | ceiling | iteration speed | example |
|---|---|---|---|---|---|
| 1 | Prompt engineering | ~$0 | low | seconds | system prompt rewrites |
| 2 | In-context few-shot | ~$0 (token cost) | low-mid | seconds | demo examples in prompt |
| 3 | Retrieval-augmented generation (RAG) | low | mid | minutes | embedding-search over docs |
| 4 | Tool-use scaffolding | low | mid-high | hours | function-calling APIs |
| 5 | Episodic memory | low | mid-high | hours | [Letta](https://www.letta.com/), [mem0](https://mem0.ai/), [MemGPT](https://memgpt.readthedocs.io/) |
| 6 | Skill library auto-extension | medium (LLM calls) | high | days | [Voyager](https://voyager.minedojo.org/) |
| 7 | Verbal self-critique (Reflexion) | medium | mid | days | [Reflexion](https://arxiv.org/abs/2303.11366) |
| 8 | Prompt evolution (GEPA-class) | medium | high | days-weeks | [GEPA](https://arxiv.org/abs/2507.19457), [OPRO](https://arxiv.org/abs/2309.03409) |
| 9 | LoRA fine-tuning | high (GPU $$$) | high | weeks | [PEFT](https://github.com/huggingface/peft), [Unsloth](https://github.com/unslothai/unsloth) |
| 10 | Full SFT / DPO post-training | very high | high | weeks-months | [TRL](https://github.com/huggingface/trl), [Tülu 3](https://arxiv.org/abs/2411.15124) |
| 11 | RL on agent trajectories (GRPO / PPO / etc) | very high | highest for some tasks | months | [GRPO](https://arxiv.org/abs/2402.03300) (DeepSeek-R1), agentic RL papers |

The TGAER stack uses mechanisms **3-8** as a unified architecture; mechanisms **9-11** are the orthogonal post-training axis that Tier 3 of the roadmap tests.

### What dominates per task class (honest take)

Different task classes have different bottlenecks. The dominant lever depends on **whether the base model already has the underlying skill** (then you need scaffolding to compose / remember / orchestrate) vs **lacks the capability fundamentally** (then post-training is necessary).

| task class | dominant lever | secondary | why | published evidence |
|---|---|---|---|---|
| **Math reasoning** (AIME, MATH, IMO) | post-training (RL on CoT) | none | base model lacks deliberate-reasoning patterns; need new behaviour via gradient updates | DeepSeek-R1 (GRPO from base), o1/o3 (post-trained reasoning) |
| **Long-horizon agents / tool use** | harness + Memory4 | post-training adds ~10-20% | model knows the tools; bottleneck is execution discipline + memory | Voyager (Minecraft), Reflexion (AlfWorld), Anthropic Pokemon |
| **Code (SWE-bench class)** | harness wins | post-training helps marginally | model can write code; bottleneck is context retrieval + multi-file navigation | SWE-agent, Cursor, Devin — all heavily scaffold-based |
| **Browser / OS automation** | harness wins | post-training too noisy | env is messy; memory of session state is critical; gradients hurt due to noise | Operator, Manus, Anthropic Computer Use |
| **Robotics (low-level control)** | post-training (RL) | harness for high-level | physical control needs gradient learning of motor policies | RT-2, SIMA, Physical Intelligence π₀ |
| **Robotics (high-level planning)** | harness | post-training for skill primitives | hybrid: gradients for skills, scaffolding for sequencing | SIMA hierarchical work |
| **Science research (AI4Sci)** | harness wins decisively | post-training rarely needed | workflow is inherently checkable + long; memory of failed paths dominates | Coscientist, ChemCrow, Asta |
| **Theorem proving** | both compound | needs both | gradient-trained tactic predictor + harness for proof tree search | AlphaProof (gradient + search hybrid) |
| **Conversational AI / personalization** | both | base from post-training, lift from memory | post-training for persona; harness for per-user context | ChatGPT memory feature, Letta, Inflection Pi |
| **Creative writing / generation** | post-training (RLHF) | harness rarely helps | task is fully novel per instance; no procedural compounding | RLHF on preferences |
| **High-frequency control / real-time** | post-training (small specialised model) | harness too slow | LLM-in-loop too high-latency | RL specialists or distilled small models |

### The decision rule

**Does the base model have the underlying capability in single-shot?**

```
                    ┌─────────────────────────────────┐
                    │  Does the base model have       │
                    │  the underlying capability      │
                    │  in single-shot?                │
                    └────────────┬────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
              YES                                NO
                │                                  │
                ▼                                  ▼
    ┌───────────────────────┐         ┌────────────────────────┐
    │  Bottleneck is        │         │  Bottleneck is the     │
    │  execution / memory / │         │  base capability       │
    │  long-horizon         │         │  itself                │
    └────────────┬──────────┘         └───────────┬────────────┘
                 │                                 │
                 ▼                                 ▼
    HARNESS + Memory4 wins              POST-TRAINING needed
    (mechanisms 3-8)                    (mechanisms 9-11)
    Examples: agents, code,             Examples: math reasoning,
    browser, science, planning          motor control, novel
                                        domains the model has
                                        no prior on
```

### When to compound — the hybrid sweet spot

For task classes where **the base model is borderline-capable** (e.g., long agentic tasks with mixed reasoning + execution), the hybrid wins:

1. **Use harness + Memory4 to discover what works** (cheap iteration, no GPU cost, fast)
2. **Use collected successful trajectories as SFT/DPO data** (post-train the model on its own best outputs)
3. **Re-deploy the post-trained model under the same harness** (compounding)

This is what Tier 3 of the roadmap tests rigorously — and what almost no published work has done with budget accounting.

### Mediums × mechanisms — generalisation matrix

Which mechanism transfers across mediums vs is medium-specific:

| mechanism | games | web | code | science | robotics | conversational | transfer story |
|---|---|---|---|---|---|---|---|
| Prompt engineering | ✓ | ✓ | ✓ | ✓ | partial | ✓ | universal (low ceiling) |
| RAG / retrieval | ✓ | ✓ | ✓ | ✓✓ | ✓ | ✓ | universal |
| Episodic memory | ✓ | ✓ | ✓ | ✓✓ | ✓ | ✓✓ | universal |
| Skill library | ✓✓ | ✓ | ✓✓ | ✓ | ✓✓ | partial | universal IF env exposes invokable skills |
| Verbal self-critique | ✓ | ✓ | ✓ | ✓✓ | partial | ✓ | universal where LLM can reason about own trace |
| Prompt evolution | ✓ | ✓ | ✓ | ✓ | partial | ✓ | universal where prompts are the surface |
| LoRA fine-tuning | partial | partial | ✓ | partial | ✓ | ✓ | per-medium (gradients don't transfer well across) |
| RL on traces (GRPO) | ✓ | ✓ | ✓ | rare | ✓✓ | ✓ | per-medium (reward signal medium-specific) |

The **TGAER stack (mechanisms 3-8)** is mostly universal-transfer. The **post-training stack (9-11)** is mostly per-medium. **That's the case for harness-first as the generalisation strategy** — gradients don't transfer across mediums, but memory + critique do.

### Open empirical questions (where Tier 3 + Tier 4 of the roadmap deliver evidence)

1. **At what task difficulty does harness alone plateau?** (We'll have data points on pokemon M5/M6/M7, mario, 2048 after PRs 1-8.)
2. **Does post-training compound with memory, substitute for it, or interfere?** (Tier 3's four-way ablation.)
3. **Does memory learned on one medium transfer to another via Semantic store?** (Tier 2 cross-task experiments.)
4. **What's the parameter-efficiency curve?** (4B-active MoE + heavy harness vs 70B + thin harness — pareto frontier on cost.)
5. **Can the Reflector itself be self-improving?** (Reflector-on-Reflector — does meta-reflection help?)

### My current best guess (subject to revision after Tier 1-3 evidence)

For **generalisable self-evolving agents across many mediums** specifically, the efficient frontier looks like:

- **Default to harness + Memory4** for any task class where the base model is single-shot capable. This is most agentic work today. The 8-PR TGAER is the right architecture.
- **Add LoRA post-training (mechanism 9)** only when harness clearly plateaus and the trajectory data quality justifies the GPU spend — usually after thousands of validated harness episodes.
- **Reserve full GRPO/PPO (mechanism 11)** for narrow tasks where the base model lacks the fundamental skill (math, motor control, novel domains). It's the highest-ceiling mechanism but the worst transfer story across mediums.
- **The harness wins for cross-medium transfer**; post-training wins for raw ceiling on a single medium. Most real applications are cross-medium, which biases toward harness as the foundation.

This is the working hypothesis the Tier 1-3 roadmap is designed to test. The honest answer to "what helps with what" today is *we have priors but not measurements* — and producing those measurements rigorously **is the contribution**.

## Applications — where TGAER architecture matters

The TGAER design is well-suited to **any task where success is checkable, the trajectory is long-horizon, and continual improvement matters more than peak per-instance capability**. Eight high-value application domains where this profile fits, ranked by current ecosystem signal:

### AI4Science (highest leverage)

The TGAER framework is structurally a good fit for scientific automation: long horizons, sparse rewards, the success criterion is independently verifiable (the molecule synthesizes / the experiment reproduces / the theorem checks), and accumulated semantic memory (failed reaction → ruled-out pathway) compounds across trials.

| domain | task class | published precedent | why TGAER fits |
|---|---|---|---|
| **Autonomous chemistry** | Multi-step retrosynthesis planning + lab execution | [Coscientist](https://www.nature.com/articles/s41586-023-06792-0) (Boiko et al. 2023, *Nature*), [ChemCrow](https://arxiv.org/abs/2304.05376) (Bran et al. 2023) | 10-30 step plans, failed routes should self-prune (Procedural + Semantic stores) |
| **Drug discovery** | Lead optimisation, target deconvolution | [Recursion's Phenomap](https://www.recursion.com/), [Insilico Medicine Pharma.AI](https://insilico.com/) | Memory of which scaffolds failed for which targets is exactly L2.Episodic + L2.Semantic |
| **Materials science** | Crystal structure search, property prediction | [GNoME](https://deepmind.google/discover/blog/millions-of-new-materials-discovered-with-deep-learning/) (DeepMind 2024), [MatterGen](https://arxiv.org/abs/2312.03687) (Microsoft 2024) | Multi-step exploration with checkable simulations |
| **Autonomous biology** | Protein engineering loops, gene editing experimental design | [Asta](https://www.anthropic.com/news/anthropic-deepens-research-collaborations) (Anthropic 2025 — automated science assistant for AllenAI) | Multi-day experiment cycles where memory of failed perturbations dominates |
| **Theorem proving / formal math** | Lean / Coq proof search | [AlphaProof](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) (DeepMind 2024), [Lean Copilot](https://github.com/lean-dojo/LeanCopilot) | Long proof trees with verifiable sub-goals — Reflector writes "this lemma always helps when goal contains X" |
| **Scientific literature synthesis** | Multi-paper claim verification, hypothesis generation | [Elicit](https://elicit.com/), [Consensus](https://consensus.app/), [STORM](https://storm.genie.stanford.edu/) (Stanford) | Long retrieval chains where Episodic memory of "papers already searched" prevents redundant work |

**Why now (2026 climate):** every major lab now has an AI-for-science arm (DeepMind GNoME / AlphaFold-3, Anthropic Asta, OpenAI partnerships with Scale + scientific orgs, NVIDIA BioNeMo). The bottleneck has shifted from model capability to **agent reliability over long horizons** — which is exactly the TGAER framing.

### Coding agents (largest economic prize)

| sub-domain | published precedent | TGAER fit |
|---|---|---|
| **SWE-bench class** (multi-file repo fixes) | [SWE-agent](https://swe-agent.com/) (Yang et al. 2024), [Devin](https://www.cognition.ai/blog/introducing-devin) (Cognition), [Cursor agent mode](https://www.cursor.com/) | Multi-step file navigation, failed-attempt memory critical |
| **API integration / scripting** | [Continue.dev](https://continue.dev/), [Aider](https://aider.chat/) | Episodic recall of past similar refactors |
| **Continuous test/CI debugging** | [Sourcegraph Cody](https://sourcegraph.com/), [GitHub Copilot Workspace](https://githubnext.com/projects/copilot-workspace) | Cross-PR semantic rules ("this lint pattern means X") |

Devin / Cognition / Cursor are billion-dollar bets on exactly the long-horizon-coding problem.

### Continual personalization (memory-native space)

| use case | precedent | TGAER fit |
|---|---|---|
| **Personal AI assistants** with months of context | [Letta](https://www.letta.com/), [mem0](https://mem0.ai/), [Limitless](https://www.limitless.ai/) | L2.Episodic + Self-model layer specifically designed for this |
| **Enterprise CSM / support agents** with customer history | [Sierra.ai](https://sierra.ai/), [Decagon](https://decagon.ai/), [Ada](https://www.ada.cx/) | Memory of past tickets + Reflector for policy updates |
| **Education tutors** | [Khanmigo](https://www.khanacademy.org/khan-labs), [MagicSchool](https://www.magicschool.ai/) | Per-student Self-model, per-curriculum Semantic store |

### Robotics / embodied AI

| use case | precedent | TGAER fit |
|---|---|---|
| **Robot manipulation policies** | [SIMA](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/) (DeepMind 2024), [RT-2](https://robotics-transformer2.github.io/) | Long-horizon task plans, skill library of motion primitives |
| **Autonomous vehicles** (decision layer) | [Wayve](https://wayve.ai/), [Comma.ai](https://comma.ai/) | Episodic memory of unusual scenarios |
| **Warehouse / industrial agents** | [Physical Intelligence](https://www.physicalintelligence.company/), [Skild AI](https://www.skild.ai/) | Cross-task skill transfer |

### Browser / OS automation

| use case | precedent | TGAER fit |
|---|---|---|
| **General browser agents** | [Operator](https://openai.com/index/introducing-operator/) (OpenAI), [Manus](https://manus.im/), [Multion](https://www.multion.ai/) | Multi-page workflows, semantic rules about UI patterns |
| **Desktop automation** | [Anthropic Computer Use](https://www.anthropic.com/news/3-5-models-and-computer-use), [Open Interpreter](https://www.openinterpreter.com/) | OS-level long horizon, memory of past app interactions |
| **RPA 2.0** (intelligent process automation) | [UiPath GenAI agents](https://www.uipath.com/), [Automation Anywhere AI Agent Studio](https://www.automationanywhere.com/) | Enterprise long-horizon workflows |

### Other high-signal application domains

| domain | precedent |
|---|---|
| **Defense / dual-use** | [Anduril Lattice](https://www.anduril.com/lattice/), [Palantir Maven AI](https://www.palantir.com/) |
| **Healthcare diagnostics** | [Hippocratic AI](https://www.hippocraticai.com/), [Glass Health](https://glass.health/) |
| **Legal research / contract review** | [Harvey](https://www.harvey.ai/), [Ironclad AI Assist](https://ironcladapp.com/) |
| **Financial analysis** | most quant funds + Bloomberg GPT successors; few public products |
| **Climate / energy grid optimisation** | [DeepMind energy](https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/) (extended), Tapestry / Climavision |

### Where TGAER does NOT fit

Honest delimitation:

- **One-shot tasks** (single-turn Q&A, summarisation, image generation) — no long horizon, no continual learning to do
- **Pure perception** (image classification, OCR, speech recognition) — no decision-making
- **Real-time low-latency control** (high-frequency trading, robotics control loops <10ms) — LLM-in-the-loop too slow
- **Fully novel-per-instance tasks** (creative writing, original research questions) — Procedural cache and Semantic memory degrade to per-instance state with no compounding

## Cross-refs

- Sister architecture doc: [`architecture.md`](architecture.md) (current per-game architecture as-shipped)
- Pokemon stage history: [`experiments/gemma/cross-stage-diagnosis.md`](experiments/gemma/cross-stage-diagnosis.md)
- 3-game MACLA findings: [`experiments/gemma/macla_findings.md`](experiments/gemma/macla_findings.md)
- Stage S openevolve writeup: [`experiments/openevolve_milestones/v1.md`](experiments/openevolve_milestones/v1.md) (in `feat/openevolve-milestones-spike` branch)
- Future repo target: https://github.com/charleneleong-ai/tgaer
- Frontier-lab references: [Anthropic Claude plays Pokemon](https://www.anthropic.com/news/claude-plays-pokemon), [DeepMind SIMA](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/), [OpenAI Operator](https://openai.com/index/introducing-operator/), [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

## Cross-refs

- Sister architecture doc: [`architecture.md`](architecture.md) (current per-game architecture as-shipped)
- Pokemon stage history: [`experiments/gemma/cross-stage-diagnosis.md`](experiments/gemma/cross-stage-diagnosis.md)
- 3-game MACLA findings: [`experiments/gemma/macla_findings.md`](experiments/gemma/macla_findings.md)
- Stage S openevolve writeup: [`experiments/openevolve_milestones/v1.md`](experiments/openevolve_milestones/v1.md) (in `feat/openevolve-milestones-spike` branch)
- Future repo target: https://github.com/charleneleong-ai/tgaer
- Frontier-lab references: [Anthropic Claude plays Pokemon](https://www.anthropic.com/news/claude-plays-pokemon), [DeepMind SIMA](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/), [OpenAI Operator](https://openai.com/index/introducing-operator/), [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
