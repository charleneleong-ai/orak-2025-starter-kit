# Orak 2025 — Architecture & Roadmap

**Last updated:** 2026-05-19 (added "Generalization beyond games" section; experimental snapshot frozen at post-asm-fix Stage K — see [`cross-stage-diagnosis.md`](experiments/gemma/cross-stage-diagnosis.md) for Stage L → Q progression)

One-page snapshot of the cognitive architecture, what we've proven useful, what we've ruled out, and what's still open. For depth see [`docs/experiments/gemma/cross-stage-diagnosis.md`](experiments/gemma/cross-stage-diagnosis.md) and [`docs/experiments/gemma/macla_findings.md`](experiments/gemma/macla_findings.md).

> **Successor architecture in progress:** [`generalized-agent-mva.md`](generalized-agent-mva.md) — Memory4 + Reflector MVA aimed at a single agent that handles any embodied task without per-game scaffolds. PR 1 (futile-action detector) in flight on `feat/futile-action-detector`.

> ⚠️ **Read this before scanning pokemon results:** every pokemon experiment row dated **2026-03-28 → 2026-05-13** ran with `pokered/data/maps/objects/*.asm` empty. The harness emitted placeholder `OBJ_n_n` tokens instead of real `SPRITE_*` names, so 74–78% of reasoning chains were anchored on placeholders. PR #80 hard-fails on the missing dir; PR #81 reran Stage D + Stage H (n=3 each) under the fix and collapsed the bimodal `[57.14, 57.14, 28.57]` distribution to `[57.14, 57.14, 57.14]` zero-variance. Pokemon rows in this doc are tagged **`PRE-ASM-FIX`** where the placeholder caveat applies. See [`docs/experiments/pokemon-asm-gap.md`](experiments/pokemon-asm-gap.md) for the full list of affected experiments.

---

## Architecture in use (Stage D = the baseline)

```
Observation  ──>  Subtask planner (D)  ──>  Vector memory (C)  ──>  Procedure cache (B)
   ▲                       │                       │                          │
   │                       ▼                       ▼                          ▼
   │              Subgoal sequence         Top-K retrieval (MMR)      Tool-call decision
   │                       │                  + temporal decay                 │
   │                       └──────────────────────┬────────────────────────────┘
   │                                              ▼
   │                                       LLM (Gemma 26B / Qwen 35B)
   │                                              │
   │                                              ▼
   │                                  Per-game adapter (PR #64)
   │                                              │
   │                                              ▼
   └──────────────────────────────────  Game env (PyBoy + pokered)
```

| Layer | Source | Role | Status |
|---|---|---|---|
| Agent shell | `agents/macla/unified.py` (`UnifiedMaclaAgent`) | Orchestrates planner + memory + procedures + reflection | ✅ Net-positive across all 3 games |
| Procedure cache (B) | `agents/macla/procedures/` | Bayesian acquisition of named tool sequences from successful trajectories | ✅ −14.29pp on pokemon when removed (PR #69) |
| Vector memory (C) | `agents/macla/vector_memory/` | MMR retrieval + temporal decay over past observations | ✅ Cheap signal, kept |
| Subtask planner (D) | `agents/macla/planner/` | Emits subgoal sequence, replans every K steps | ✅ Default on pokemon + mario; off on 2048 |
| Self-reflection | `agents/{pokemon_red,super_mario,twenty_fourty_eight}/game_adapter.py` | Per-game critique schedule (PR #64) | ✅ Improves critique quality, no harm |
| LoopDetector | `agents/macla/loop_detector.py` + obs preprocessor (PR #50/#51) | State-hash + action-class + oscillation signals → surfaced into obs prompt | ✅ Wired into stuck-state recovery |
| Model serving | `serving/{gemma,qwen}_serve.sh`, vLLM @ `:8000` | Single A100 40GB; swap models between sweeps | ✅ Hot-swap pattern works |
| Observation | `evaluation_utils/mcp_game_servers/pokemon_red/...pyboy_runner.py` + `pokered/data/maps/objects/*.asm` | Real `SPRITE_*` tokens (was `OBJ_n_n` placeholders pre-#80) | ✅ Hardened, hard-fails on missing asm (#80) |
| Reward | `evaluation_utils/.../pokemon_red_env.py:277-304` | 7 milestones, score = (n/7)·100 | — |
| Sweep orchestration | `experiments/autoresearch.py` + companion daemons | Schedule-driven sweep, results.jsonl, live PR narrative, PPID=1 daemons | ✅ Used by every Stage A→K |

## Models tested

| Model | Quant | Params | Tool parser | Stage | Result (pokemon n=3) |
|---|---|---|---|---|---|
| Gemma-4 26B-A4B-it | AWQ-Int4 | 4B active / 26B total (MoE) | pythonic | A → G, D-rerun | **57.14% × 3, σ=0** (post-fix) |
| Qwen 3.5 35B-A3B-GPTQ | Int4 | 3B active / 35B total (MoE) | hermes | H, H-rerun | **57.14% × 3, σ=0** (post-fix) |
| Qwen 3-30B-A3B-Thinking-2507 | AWQ-Int4 | 3B active / 30B total (MoE) | hermes + `--reasoning-parser qwen3` | J (pre-fix) | 28.57% × 3 — **rerun deferred** |

## Verdict from 11 stages (A → K)

**The 57.14% pokemon ceiling lives at M5 (`'Viridian' in map_name`), not in reasoning, model lineage, or cognitive scaffolding.** Pre-asm-fix bimodal `[57.14, 57.14, 28.57]` results in Stages G/H/B' were placeholder-reasoning artifacts during the scripted M1-M4 phase; they collapse to zero variance under #80. M1-M4 are scripted progressions (cutscene + name screen + forced rival battle). M5 is the first milestone that demands real navigation: cross Route1 from south to north, ~25 tiles. **0 / 6 post-fix runs ever set foot in Viridian.** ([PR #81](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/81), [PR #82](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/82))

---

## Component decision log

The architecture didn't spring up whole. Each component below was added in response to a specific failure mode in an earlier sweep; the table records what was added, why, what it bought, and where the verdict lives.

### Core cognitive components (Stages A → D)

| Component | Introduced in | Hypothesis | Ablation result | Verdict |
|---|---|---|---|---|
| MACLA core (Bayesian procedure selector + contrastive refinement) | [#5](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/5), [#14](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/14) | Memory-augmented procedure learning generalises across games | Mario 100% W1-1 at iter 5 (#22); 2048 8.40% best; pokemon flat at 14.29% pre-substrate | ✅ Core kept. "Unified" claim qualified: works for dense-reward / repeating-context games. |
| Triage + autoresearch loop | [#14](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/14), [#20](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/20) | Schedule-driven sweeps with active early-kill keep GPU productive | Caught mario's 77% breakthrough that an earlier triage would have killed | ✅ Default. Schema in `experiments/autoresearch.py`. |
| Checkpoint carry-over (`--prev-run-id`) | [#22](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/22) | MACLA procedures persisting iter→iter compound knowledge | Mario 35→44→52→100% across 5 KEEP iters; 2048 8.40% (vs 6.02% prior); pokemon flat | ✅ Default. Strongest single uplift in the project. |
| Reward shaping + state abstraction (2048) | [#23](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/23) | Corner-anchor reward + `StrategicGridExtractor` keys make procedures fire on novel boards | 2048 cold-start 4.88 → 7.04 (+44%) but ceiling regressed because param-search bounds inherited from prior keying | ✅ Cold-start lift real; param search needs retuning. Open. |
| Vector memory provider (Stage C) | [#26](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/26) | Composable memory primitive separable from MACLA | 2048 +48% (Stage A→C); mario 0% null; pokemon 0% null (PR #28) | ✅ Kept; helps 2048 specifically. |
| Subtask planner (Stage D) | [#31](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/31) | Sub-goal decomposition + exploration heuristics + outcome-tagged history | Mario +25% (C→D); 2048 −23% (C→D); pokemon Stage D=57.14% pre-fix (originally 0% pre-#46/#52, see below) | ✅ Default on pokemon + mario; off on 2048 (PR #28 verdict applied to configs). |
| Stage B' — `use_procedure_layer` master switch | [#69](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/69) | Removing the procedure cache should be neutral if procedures don't generalise | Pokemon −14.29pp, mario −7.72pp, 2048 −3.03pp (n=3 each) | ✅ Procedures **net-positive on all 3 games**. Toggle kept as ablation knob; defaults preserve Stage D. |

### Action-layer interventions (closed — none lifted the pokemon ceiling)

| Stage | PR | What was added | Pokemon n=1 result | Verdict |
|---|---|---|---|---|
| D + reflect | [#62](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/62), [#64](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/64) | LLMSelfReflector at `every=10` | 57.14% **PRE-ASM-FIX** | Same ceiling. Self-reflection emits the correct diagnosis ("stuck in movement loop") but does not change behaviour. |
| E — verify_action | [#66](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/66) (CLOSED) | LangGraph + Reflexion-style action verification | 57.14% **PRE-ASM-FIX**, 91% revision rate | Rewriting actions doesn't lift the ceiling. Closed. |
| F — plan-do-check | [#67](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/67) (CLOSED) | `ToolGateValidator` + `LLMPlanValidator` + retry | **28.57% PRE-ASM-FIX** — REGRESSED | Validator over-rejected legitimate actions (warps, signs during stuck-recovery). Closed. |
| G — procedure-escape | [#70](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/70) (CLOSED) | Failure-streak retire K=5 + force-LLM-on-stuck N=50 | 47.62% ± 16.49pp **PRE-ASM-FIX** | Force-LLM fired 100+ times per iter; LLM fallback itself couldn't break M4. Closed. |

### Observation-layer hardening

| Component | Introduced in | What it fixed | Verdict |
|---|---|---|---|
| `RegexSpatialExtractor` / `DictFieldExtractor` / `StrategicGridExtractor` | [#23](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/23) | Game-specific procedure keys so cached procedures actually fire on next observation | ✅ Kept per-game; non-trivial researcher knowledge encoded here. |
| MMR rerank + repetition decay (vmem) | [#60](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/60) | Break stuck-state retrieval loops where vmem kept returning the same K observations | ✅ Default. Avoids reinforcement-trap on dense-context games. |
| Pokemon obs preprocessor wiring | [#61](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/61) | `[Full Map]` in prompt + map memory accumulating across steps | ✅ Necessary not sufficient (Stage A obs-fix still wedged at y<5 without exploration push). |
| Warp-destination rendering | [#44](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/44), [#47](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/47) | Render `wWarpEntries` into obs as `Warp→<destination>` labels | ✅ Lifted "what does this stairwell do?" ambiguity. |
| Milestone 1→2 cutscene-aware reward | [#45](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/45) | Pokemon eval recognises cutscene-triggered transitions | ✅ Pokemon scoring no longer drops M1→M2. |
| 2048 progress normalisation | [#46](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/46) | `log2(max_tile)` units instead of raw score | ✅ All post-#46 2048 numbers in this doc are on the new scale. |
| `LoopDetector` (state-hash + action-class + oscillation) | [#50](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/50), [#51](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/51) | Surface oscillation signal into obs prompt | ✅ Kept; wired through `UnifiedMaclaAgent`. |
| Exhaust-interactables checklist prompt | [#55](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/55) | Stuck-state recovery prompt nudges agent to enumerate untried objects | ✅ Default. |
| Case-insensitive asm-file resolver | [#52](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/52) | `pret/pokered` uses `RedsHouse1F.asm`; runtime `map_names.json` says `RedsHouse1f` → 76/248 maps unresolvable on Linux | ✅ Fixes the casing trap **once asm files exist**. Pre-#80 the dir was empty so this fix did nothing. |
| Hard-fail when `pokered/*.asm` missing | [#80](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/80) | Detect the empty-dir failure mode at `PyBoyRunner.__init__` | ✅ Root-cause fix for the pre-#80 placeholder caveat. Self-fixing error message (exact `git clone` command + doc pointer). |

### Per-game self-reflection (PR #64)

Per-game `RECOMMENDED_USE_SELF_REFLECTION` + `RECOMMENDED_REFLECTION_EVERY` constants on `agents/{pokemon_red,super_mario,twenty_fourty_eight}/game_adapter.py`. `UnifiedMaclaAgent` precedence: YAML > adapter > False. Even though self-reflection doesn't lift the pokemon ceiling, it improves critique quality cross-game and is a no-harm opt-in.

### Model serving + sweep infrastructure

| Component | Introduced in | Purpose |
|---|---|---|
| Local vLLM/Ollama/MLX serving | [#15](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/15) | Decouple from API providers; runs on single A100 40GB |
| Gemma-4 26B serving + agent configs | [#16](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/16) | Default model. AWQ-Int4 quantisation, pythonic tool parser. |
| Hermes pattern lift (caching, retry, trajectory) | [#25](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/25) | Pull in upstream Hermes harness primitives |
| Autoresearch package adoption | [#37](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/37), [#43](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/43), [#58](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/58) | Bump to v0.19.0+ → re-exports from shared package; `format_recent_history`; drop local `current_run_updater` |
| WandB Artifact auto-archive | [#72](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/72), [#77](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/77) | Curated `game_logs/<game>/` upload as wandb Artifact per game |
| Loop-detector score-grace period | [#56](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/56) | Suppress LoopDetector noise after recent progress |
| Ruff lint baseline + CI | [#59](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/59) | Style consistency |

---

## Per-game scoreboard

| Game | Best stage (post-#46 units) | Best score | n | Source |
|---|---|---|---|---|
| **pokemon_red** | Stage D | **57.14% × 6** (σ=0) | 6 | post-asm-fix re-runs ([PR #81](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/81)) |
| **super_mario** | Stage D + carry-over | **100.0%** (W1-1 complete, iter 5) | 1 | `macla_procedure_carryover/` ([PR #22](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/22)) |
| **twenty_fourty_eight** | Stage D | **63.64%** | 3 ep | `pr31_2048_rerun/` ([PR #31](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/31)) |

### Cross-game scoreboard (PR #28 cross-game ablation)

The Stage C/D substrate **does not generalise uniformly** — each game has a different optimum and the substrate's two halves help different games:

| | vmem (Stage A → C) | planner (Stage C → D) |
|---|---|---|
| **2048** | **+48% HELPS** | −23% REGRESSES |
| **mario** | 0% null | **+25% HELPS** |
| **pokemon** | 0% null | **−100% REGRESSES** (pre-#46/#52 confound, see below) |

| Game | Best stage @ PR #28 | Best stage @ PR #31 (post-#46) | Reason |
|---|---|---|---|
| 2048 | Stage C (vmem only) | Stage D (53→63%) — units changed post-#46 | Stage C still best by inference-cost / lift ratio. |
| mario | Stage D | Stage D (+25%) | Planner does the work; vmem dead weight. |
| pokemon | Stage A (no substrate, 14.29%) | Stage D (57.14%) | Pre-#46/#52 confounds inverted the verdict. |

Plot: [`docs/experiments/gemma/plots/stage_d_cross_game.png`](experiments/gemma/plots/stage_d_cross_game.png) · [`docs/experiments/gemma/plots/pr31_ablation_26b.png`](experiments/gemma/plots/pr31_ablation_26b.png) · [`docs/experiments/gemma/plots/pr64_v3_crossgame.png`](experiments/gemma/plots/pr64_v3_crossgame.png)

### Pokemon Red — the M5 navigation gate

Stage D + procedures + vmem + planner = **57.14% × 6, σ=0** post-asm-fix. 0/6 runs ever set foot in Viridian City. M5 is the first milestone that requires real (non-scripted) navigation: cross Route1 south-to-north, ~25 tiles. None of Stage A–H operated on the navigation layer; they all touched planner prompts, action validation, procedure caches, self-reflection, or model lineage. Full narrative: [`cross-stage-diagnosis.md`](experiments/gemma/cross-stage-diagnosis.md).

**Pokemon-stage timeline** (pre-asm-fix rows tagged):

| Stage | PR | Score | Tag |
|---|---|---|---|
| Stage A | #31 | 28.57% | PRE-ASM-FIX |
| Stage B | #31 | 57.14% | PRE-ASM-FIX |
| Stage C | #31 | 0.00% | PRE-ASM-FIX |
| Stage D | #31 | 57.14% | PRE-ASM-FIX |
| Stage D++ (600 steps) | #31 | 71.43% | PRE-ASM-FIX, n=1 |
| Stage D + reflect | #62, #64 | 57.14% | PRE-ASM-FIX |
| Stage E (verify_action) | #66 | 57.14% | PRE-ASM-FIX, CLOSED |
| Stage F (plan-do-check) | #67 | 28.57% | PRE-ASM-FIX, CLOSED |
| Stage B' (no procedures n=3) | #69 | 42.86% ± 14.29pp | PRE-ASM-FIX |
| Stage G (procedure-escape n=3) | #70 | 47.62% ± 16.49pp | PRE-ASM-FIX, CLOSED |
| Stage H (Qwen 3.5 n=3) | #71 | 47.62% ± 16.49pp | PRE-ASM-FIX, CLOSED |
| Stage J (Qwen3-Thinking n=3) | #76 | 28.57% × 3 | PRE-ASM-FIX, rerun deferred |
| **Stage D-rerun (Gemma n=3)** | #81 | **57.14% ± 0pp** | POST-FIX |
| **Stage H-rerun (Qwen n=3)** | #81 | **57.14% ± 0pp** | POST-FIX |

### 2048 — the carryover + state-abstraction story

The most informative game for the procedure-carryover hypothesis. Across 33 iters in `macla_procedure_carryover/` the best 2048 score was **8.40%** at iter 5 — a real lift over PR #22's 6.02% prior. PR #23 added strategic-feature keying (`StrategicGridExtractor`) and a corner-anchor reward; cold-start improved **4.88 → 7.04 (+44%)** because procedures actually fired across boards, but the autoresearch param-search bounds were inherited from the literal-grid keying and never found the new working region.

Post-#46 normalisation: 2048 numbers are now in `log2(max_tile)·100` units. Stage A baseline = 54.55, Stage C = 54.55, Stage D = **63.64** (3 ep, KEEP), Stage B (planner only) = 63.64. PR #28's "Stage C wins" verdict holds: planner adds 1.2× inference cost without lift; bottleneck is forward search, not decomposition.

| Sweep | Best iter | Score | PR |
|---|---|---|---|
| `macla_procedure_carryover/` | iter 5 | 8.40% (pre-#46 units) | [#22](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/22) |
| `macla_state_abstraction/` | iter 0 | 7.04% (+44% cold-start vs prior) | [#23](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/23) |
| `pr31_2048_rerun/` Stage A | — | 54.55% (post-#46) | [#31](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/31) |
| `pr31_2048_rerun/` Stage D | — | **63.64%** (post-#46, 3 ep, KEEP) | [#31](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/31) |
| Stage B' (no procedures) | n=3 | 60.61% (−3.03pp vs Stage D) | [#69](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/69) |

Plots: [`docs/experiments/gemma/plots/cognitive_check_2048.png`](experiments/gemma/plots/cognitive_check_2048.png) · [`docs/experiments/gemma/plots/stage_a_vs_c_2048.png`](experiments/gemma/plots/stage_a_vs_c_2048.png) · [`docs/experiments/gemma/plots/cross_game_scoreboard.png`](experiments/gemma/plots/cross_game_scoreboard.png).

### Super Mario — the strong-fit poster child

Mario is the cleanest demonstration that procedure carryover compounds: `macla_procedure_carryover/` produced three monotonic KEEP iters (35→44→52) followed by a 100% W1-1 completion at iter 5. Each iter loaded the previous iter's MACLA checkpoint, so the 51.55% start of iter 2 carried iter 1's procedures.

Local visual contexts (Goomba ahead, gap, ledge) recur across the level → `RegexSpatialExtractor` keys procedures by entity-relative-to-player tokens (e.g. `goomba_ahead_near`) which are short, repeating, semantically meaningful. Dense per-step reward (x_pos delta) keeps the autoresearch param-search well-conditioned.

| Sweep | Best iter | Score | PR |
|---|---|---|---|
| `macla_procedure_carryover/` | iter 5 | **100.0%** (W1-1 complete) | [#22](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/22) |
| `pr31_mario_rerun/` Stage A | — | 44.50% | [#31](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/31) |
| `pr31_mario_rerun/` Stage B (planner only) | — | **61.26%** | [#31](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/31) |
| `pr31_mario_rerun/` Stage C (vmem only) | — | 35.18% | [#31](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/31) |
| `pr31_mario_rerun/` Stage D | — | 35.21% | [#31](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/31) |
| Stage B' (no procedures) | n=3 | 27.49% (−7.72pp vs Stage D) | [#69](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/69) |

The PR #28 verdict (mario → Stage D) is the production config; the iter-5 100% number is from the unified-macla carry-over sweep where procedures compounded across iterations.

---

## What we've ruled out

| Lever | Verdict | PR |
|---|---|---|
| Action validation (verify_action / 91% revision rate) | Same ceiling | #66 (closed) |
| Planning retries (ToolGateValidator + LLMPlanValidator) | Regressed to 28.57% — validator over-rejected | #67 (closed) |
| No procedures (B' ablation) | −14.29pp pokemon / −7.72pp mario / −3.03pp 2048 — procedures are net-positive | #69 (merged) |
| Procedure-layer escape (failure-streak + force-LLM-on-stuck) | 47.62% bimodal pre-fix; no lift | #70 (closed) |
| Self-reflection schedule density (every=10 vs every=30) | Ties baseline | #62, #64 |
| Step budget (600 steps) | Stuck at M4 for 461 frames | — |
| Model lineage (Gemma → Qwen 3.5) | Same upper bound | #71 (closed) → #81 (merged) |

## What's still in flight or open

| Lever | Status | Notes |
|---|---|---|
| **Stage K — cumulative cross-episode memory** (n=5, Gemma 26B, `--load-checkpoint --prev-run-id` chaining) | ✅ Done @ 10:09Z 2026-05-15 | Will land on [PR #75](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/75). Pre-fix was REGRESS −14.28pp learning_delta. **Post-fix final: `[57.14] × 5`, σ=0.00pp, `learning_delta=+0.00pp`** — floor-stable but no compounding. Introspection across iters 1-3 showed negative transfer: iter 2 took **+91 steps** to bank M4 (220 vs iter 1's 129), never reached Route1, spent 160 steps stuck in OaksLab. Iter 3 partially recovered Route1 (70 steps) but worst perseveration at 22.0%. Asm fix prevents catastrophic *floor* regression but doesn't fix underlying context-blind procedure keying. **See `Cumulative-memory mechanics` section below** for the redesign — implemented as **Stage L ([PR #85](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/85), in flight)**. |
| **Stage L — map-aware procedure cache** (n=5, Gemma 26B, modified MACLA) | Running (iter 1 launched 12:34Z 2026-05-15, ETA ~16:45Z) | [PR #85](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/85). `feat/macla-map-aware-procedures` branch. Procedures keyed on `(map_name, hash(steps))`; retrieval filters wrong-map matches; `prune_stale_procedures(max_age=2)` on checkpoint load. Minimum bar: `late_mean ≥ early_mean` (no negative transfer). Lift bar: iter-over-iter steps-to-M4 decreasing OR any iter reaches M5+ (Viridian). |
| **Map-exit callouts in observation** | Untested, recommended next | Surface unvisited adjacent maps + their exit tile coordinates in every observation. Afternoon's work. |
| **Visited-maps sticky note** | Untested | Append `(map_name, first_step, last_step)` tuples to the observation. Cheap. |
| **Mini-map / map-graph view** | Untested | Compact rendering of connected maps reachable from current location, unvisited highlighted. |
| **Subtask injection at stuck-state** | Untested | When self-reflection emits *"stuck in movement loop"*, inject `"goal: leave the current map. try moving to each edge."` |
| **Stage J — Qwen3-Thinking** (always-on reasoning budget) | Scaffold in commit `e431e30`, **rerun deferred** | Pre-fix scored 28.57% × 3 — failed M3 (no starter). Post-fix re-run not queued; would only test reasoning budget, not the navigation gate. |
| **2048 state-abstraction sweep retune** | Open since PR #23 | Cold-start showed +44% but param-search bounds still calibrated for prior keying. Two sweeps with widened bounds would close the loop. |
| **Paired rollouts + adapter / logprobs passthrough** | [PR #79](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/79) | Harness scaffold for future learning experiments. |

---

## PR timeline — what changed when

| PR | Merged | Theme | Why it mattered |
|---|---|---|---|
| #5, #14 | early | MACLA core + autoresearch framework | Foundation: Bayesian procedure selector + triage |
| #15, #16 | early | Local model serving (vLLM) + Gemma-4 26B | A100-40GB self-hosted inference |
| #20 | early | Live sweep tracking + UX | First usable progress dashboard |
| #22 | 2026-04-28 | Checkpoint carry-over | Mario 35→100%; strongest single uplift |
| #23 | 2026-04-28 | Reward shaping + state abstraction | 2048 cold-start +44% |
| #26 | 2026-05-02 | `VectorMemoryProvider` (Stage C) | Composable memory primitive |
| #28 | merged | Cross-game ablation sweep | Disproved uniform substrate; per-game configs |
| #30 | 2026-05-?? | Port orak autoresearch to SweepRunner | Shared package adoption |
| #31 | 2026-05-11 | Planner exploration heuristics + outcome-tagged history | Stage D defaults; post-#46 unit baseline |
| #44 | merged | Render warp destinations from `wWarpEntries` | Disambiguated stairwells |
| #45 | merged | M1→M2 cutscene-aware reward | Pokemon scoring correctness |
| #46 | merged | 2048 progress normalisation (`log2(max_tile)`) | All post-#46 2048 numbers on new scale |
| #47 | merged | Accept enriched `Warp→X` tile labels (#44 regression fix) | — |
| #49 | merged | Surface LLM token usage into `raw_requests.jsonl` | Cost transparency |
| #50, #51 | 2026-05-06 | LoopDetector + obs prompt wiring | Stuck-state recovery signal |
| #52 | merged | Case-insensitive asm-file resolver | 76/248 maps were unresolvable on Linux due to `F`/`f` casing |
| #55 | merged | Exhaust-interactables checklist prompt | Stuck-state recovery |
| #56 | merged | LoopDetector score-grace period | Suppress noise after recent progress |
| #60 | 2026-05-10 | Vmem MMR rerank + temporal decay | Break stuck-state retrieval loops |
| #61 | merged | Pokemon obs preprocessor wiring | `[Full Map]` accumulating across steps |
| #62, #64 | 2026-05-13 | LLMSelfReflector + per-game schedules | Self-reflection per-game adapter |
| #66 | CLOSED | Stage E — LangGraph + verify_action | Same ceiling, dead code |
| #67 | CLOSED | Stage F — plan-do-check | Regressed (28.57%) |
| #69 | 2026-05-13 | `use_procedure_layer` toggle + Stage B' cross-game baseline | Procedures net-positive on all 3 games |
| #70 | CLOSED | Stage G — procedure-escape | Bimodal pre-fix; doesn't lift ceiling |
| #71 | CLOSED → #81 | Stage H — Qwen 3.5 35B-A3B ceiling-check | Superseded by post-fix rerun |
| #72, #77 | merged | WandB Artifact auto-archive | Per-game game_logs upload |
| #75 | OPEN | Stage K — cumulative cross-episode memory | In flight |
| #76 | OPEN | Stage J — Qwen3-Thinking | Pre-fix only; rerun deferred |
| #80 | 2026-05-14 | **Hard-fail when `pokered/*.asm` missing** | Root-cause fix for placeholder-anchored reasoning |
| #81 | 2026-05-15 | **Stage D + Stage H n=3 reruns under asm fix** | 57.14% × 6 zero-variance ceiling; M5 navigation gate identified |
| #82 | 2026-05-15 | Cross-stage diagnosis post-asm-fix update | Supersedes Stage H bimodal interpretation |

---

## Cumulative-memory mechanics — TTL/decay + map-aware procedure keys

**Finding (2026-05-15, Stage K rerun iters 1-3 at 57.14% × 3 zero-variance):** even when cumulative memory doesn't change the *score* it can hurt *efficiency*. Inherited procedures from earlier iters dead-end the agent in the scripted M1-M4 phase: iter 2 took 91 more steps than iter 1 to bank M4 and never reached Route1, because procedures captured against transient OaksLab states kept the agent in OaksLab instead of progressing. This is **negative transfer** in the procedure cache — the same pathology that caused the pre-fix Stage K REGRESS (now a floor-stable but efficiency-degraded version).

The current procedure cache has two design gaps that need to close before cumulative memory becomes net-positive:

| Gap | Why it bites | Proposed fix |
|---|---|---|
| **Procedure key is not map-aware** | A procedure like `interact_with_object(SPRITE_OAK)` is keyed on the action + sprite identity. Applied later in PalletTown or Route1 where there's no Oak, the procedure still fires because the cache match doesn't condition on `map_name`. Inherited procedures end up firing in the wrong map. | Key procedures on `(map_name, sprite/object_id, action)` tuple. A procedure captured in OaksLab can only match when `map_info.map_name == "OaksLab"`. |
| **No "forget irrelevant" mechanism** | Bayesian acquisition adds procedures but never retires them based on usage staleness or context-mismatch. Procedures that succeeded once at step 50 of iter 1 stay top-K-retrievable through iter 5. | Add a **TTL or confidence decay** in `agents/macla/procedures/`: each procedure carries a `last_used_iter` + `success_count_since_last_used`; procedures unused for ≥ 1 full iter, or whose recent-use success-rate drops below a threshold, decay out of top-K. Complements the existing Stage G failure-streak retire (which retires *currently* failing procedures, not stale ones). |

Both fixes are local to `agents/macla/procedures/` and don't require a new sweep harness — they're a Stage L-style intervention. Either one alone is testable (n=3 cumulative-memory variant); both together is the more conservative bet. The current Stage K REGRESS-but-floor-stable result establishes the baseline to beat: any redesign needs to deliver `late_mean ≥ early_mean` (no negative transfer) at minimum, ideally improving steps-to-M4 iter-over-iter.

This is the next move for the procedure layer regardless of the navigation gate. The M5 navigation interventions (map-exit callouts, visited-maps sticky note, mini-map) attack the ceiling; the procedure-cache redesign attacks the *floor stability* of any future cumulative-memory sweep.

---

## Evaluation policy: when to run cross-game

**Convention (2026-05-15):** cognitive-harness interventions are validated on **pokemon-only first** (cheapest single-game cost, hardest ceiling). Cross-game reruns happen as a **follow-up sweep** if and only if the pokemon variant **LIFTs** above the 57.14% ceiling.

| Stage | Lever | Cross-game scope | Why |
|---|---|---|---|
| B' | no procedures | all 3 games at n=3 (PR #69) | Ablation testing whether procedures generalize — they do (−14.29 / −7.72 / −3.03pp) |
| D baseline | full Stage D stack | all 3 games (PR #31) | Sets the per-game baseline scoreboard |
| H | model lineage (Qwen vs Gemma) | **pokemon-only** | Specific ceiling check on pokemon's 57.14%; mario/2048 aren't gated the same way |
| J | thinking-mode reasoning budget | **pokemon-only** | Defer cross-game until pokemon shows signal |
| K | cumulative cross-episode memory | **pokemon-only** | Defer cross-game until pokemon shows signal |

Rule of thumb:
- **Architectural ablations** (B', D) → cross-game from day one, because the question is "does this component generalize."
- **Pokemon-specific ceiling pokes** (H, J, K, future navigation interventions) → pokemon-only first. If pokemon LIFTs (mean > 60% OR max ≥ 71%), queue a cross-game rerun as a follow-up PR.
- **Game-specific tuning** (2048 state-abstraction, mario carry-over) → that game only.

GPU cost rationale: cross-game is roughly 3× per sweep on a single A100 (300 steps × 3 games). For interventions that haven't shown signal on pokemon, 3× cost is wasted on confirming the obvious (no change).

---

## Generalization beyond games — what would port to real-world tasks

The architecture has explicit task-agnostic seams; what's game-shaped is a specific *task class* (long-horizon observe → act loops with sub-structure), not most components. The MaCLA core never branches on game — game-specific bits live behind the `GAME_ADAPTERS` registry in [`agents/macla/unified.py`](../agents/macla/unified.py#L30-L44) and have already proven adapter-portable across 3 games.

### What's already task-agnostic

| Component | Why it generalizes | Caveat |
|---|---|---|
| **Adapter seam** ([`unified.py:30-44`](../agents/macla/unified.py#L30-L44)) | `UnifiedMaclaAgent` is game-blind. Game-specific action schema + success/progress patterns export from `agents/{game}/game_adapter.py`. Already proven across pokemon_red, super_mario, twenty_fourty_eight. | Adapter contract assumes a *structured action schema* (pydantic). Open-ended action spaces (free-text shell) need a different contract. |
| **`autoresearch`** ([`experiments/autoresearch.py`](../experiments/autoresearch.py)) | Schedule-driven sweep orchestrator, milestone bars, kill triage, per-iter `metric_scores`. Already a separate package; zero game-specific code. | Drop-in. |
| **Procedure cache + score-gated prune** (Stage B / Stage Q2) | Caches successful sub-policies, prunes by outcome quality. Pattern applies to any task with reusable substructure. | Assumes "useful procedures recur." Tasks where each instance is fully novel (creative writing) gain nothing. |
| **Milestone-based eval + per-iter scatter** | `Milestone.metric_scores` is task-shape-agnostic — SWE-bench partial credit, WebArena sub-goals all fit. | Needs a checkpointable progress signal. |
| **Vector memory + MMR rerank + temporal decay** | No game knowledge; only operates on embedded observation text. | Cheap to keep; lift outside dense-context games is unproven. |

### What's game-shaped (would need rework)

| Component | Why it's game-shaped | What changes for real-world tasks |
|---|---|---|
| Tight observe → act loop | Assumes discrete steps with structured per-step observations. | Browser / OS / robotics fit. Long-form generation, dialogue, research don't — no step boundary. |
| `graph_hint` + map extraction ([`pokered_map_extractor.py`](../agents/macla/pokered_map_extractor.py)) | Curated structural prior from `pokered/*.asm`. The *pattern* (inject domain priors into obs) generalizes; the extractor doesn't. | Replace with task-domain prior: sitemap for browsers, repo tree for SWE-bench, scene graph for robotics. |
| Self-reflection cadence | Per-game `RECOMMENDED_REFLECTION_EVERY`. Tuned to game pacing. | Re-tune per task class. Mechanism is generic; cadence is not. |
| LoopDetector action taxonomy | Game-defined action classes (movement / interact / menu). | Re-define per task. Detector itself is task-agnostic. |

### Task classes likely to port cleanly

| Class | Fit | Why |
|---|---|---|
| Browser automation (BrowserGym, WebArena, VisualWebArena) | ✅ Strong | Same observe/act shape; sitemap as `graph_hint` analogue; recurring sub-procedures (login, search, form fill). |
| OS / terminal agents (OSWorld, SWE-bench Verified) | ✅ Strong | Adapter pattern fits; procedure cache for repo-nav primitives; milestones → partial credit. |
| Robotics sim (ManiSkill, Habitat, RoboArena) | ✅ Likely | Discrete-ish obs/action; scene graph as structural prior; sub-skills as procedures. |
| Embodied AI (AI2THOR, Habitat) | ✅ Likely | Map-aware procedure keying ([Stage L](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/85)) is already the right shape. |
| Long-form generation, dialogue, research writing | ❌ Doesn't fit | No observe/act loop, no procedure reuse, no checkpointable milestones. |
| One-shot novel-instance tasks | ❌ Doesn't fit | Procedure cache and vmem degrade to per-instance state. |

The sweet spot is **long-horizon agentic tasks with sub-structure**: multiple steps, checkpointable progress, sub-procedures that recur within and across instances. Pokemon's M5 navigation gate (cross-map traversal under sparse signal) is structurally analogous to "navigate a multi-page checkout flow" or "find the failing test in this repo." If map-aware procedures + observation priors break M5 (the Stage L → Q working theory), the same shape should drop into the agentic-benchmark family.

---

## Current work (2026-05-24) — generalized agent harness (MVA)

The per-game scaffold work (Stage S openevolve, milestone library) plateaued at pokemon 6/7 @ 1200 steps, and surfaced shared failure modes across mario + 2048 that are NOT pokemon-specific: procedure-cache poisoning (mario locking into kill-on-spawn after ep4), MACLA dominance lock-in (2048 selecting one of 4 procs 84% of decisions), and futile-action loops in all three games. Diagnosing these as universal architectural gaps shifted the priority from "more pokemon scaffolding" to a **task-agnostic Memory4 + Reflector MVA**: see [`generalized-agent-mva.md`](generalized-agent-mva.md) for the full 5-layer design, have-vs-need audit by layer, and 8-PR staged plan.

**Cross-game baselines captured 2026-05-23** (wandb under `chaleong`):

| game | run | budget | best | mean | notes |
|---|---|---|---|---|---|
| pokemon (v3) | `step_budget_1200_baseline_20260523T171829Z` | 1200 | 6/7 (0.86) | — | reached ViridianMart, got Oak's Parcel |
| pokemon (v4) | `step_budget_2000_baseline_20260523T210201Z` | 2000 | in-flight | — | stuck in ViridianCity loop (stagnation=1290 at step 1821) |
| mario | `stage_s_super_mario_1000_20260523T210441Z` | 1000 | 21.85% | 9.04% | 58 consecutive 8.20% deaths after ep4 — procedure poisoning |
| 2048 | `stage_s_2048_1000_20260523T210447Z` | 1000 | 63.64% | 44.92% | 17 episodes, only 4 unique procs across 999 selections |

**PR 1 in flight — universal futile-action detector** (branch `feat/futile-action-detector` @ `176f68c`):
- Agent-side hook in [`unified.py:_base_fallback`](../agents/macla/unified.py#L549) that hashes the planner-visible obs, fires when K=3 consecutive obs are byte-identical, injects a "your last actions did nothing" hint
- 11/11 tests in [`tests/test_futile_action_detector.py`](../tests/test_futile_action_detector.py) — game-agnostic parametrization + state-machine sanity
- Three regression rollouts launched 2026-05-24 01:37 UTC against the baselines above; cross-game results pending

PRs 2-8 from the MVA plan are queued behind PR 1 validation: per-skill success-rate floor, stagnation→skill demotion, `agent_events.jsonl` telemetry, episode-end pruning, episodic store + retrieval, Reflector post-episode critique, self-model.

## Cross-refs

- Detailed pokemon history: [`docs/experiments/gemma/cross-stage-diagnosis.md`](experiments/gemma/cross-stage-diagnosis.md)
- 3-day MACLA findings (mario / 2048 / pokemon): [`docs/experiments/gemma/macla_findings.md`](experiments/gemma/macla_findings.md)
- Asm-fix root cause: [PR #80](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/80) (hardened `pyboy_runner.py`) + [PR #81](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/81) (Stage D+H reruns)
- Asm gap analysis: [`docs/experiments/pokemon-asm-gap.md`](experiments/pokemon-asm-gap.md)
- Cross-game scoreboard plots: [`docs/experiments/gemma/plots/`](experiments/gemma/plots/) (`stage_d_cross_game.png`, `pr31_ablation_26b.png`, `pr64_v3_crossgame.png`, `cross_game_scoreboard.png`)
- Sweep convention: `~/.claude/CLAUDE.md` (ML sweep orchestration section) + `experiments/autoresearch.py` docstring
