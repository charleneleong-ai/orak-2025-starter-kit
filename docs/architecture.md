# Orak 2025 — Architecture & Roadmap

**Last updated:** 2026-05-15 (post-asm-fix verdict, PR #81 merged)

One-page snapshot of the cognitive architecture, what we've proven useful, what we've ruled out, and what's still open. For depth see [`docs/experiments/gemma/cross-stage-diagnosis.md`](experiments/gemma/cross-stage-diagnosis.md).

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
| Subtask planner (D) | `agents/macla/planner/` | Emits subgoal sequence, replans every K steps | ✅ Default |
| Self-reflection | `agents/{pokemon_red,super_mario,twenty_fourty_eight}/game_adapter.py` | Per-game critique schedule (PR #64) | ✅ Improves critique quality, no harm |
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

## What we've ruled out

| Lever | Verdict | PR |
|---|---|---|
| Action validation (verify_action / 91% revision rate) | Same ceiling | #66 (closed) |
| Planning retries (ToolGateValidator + LLMPlanValidator) | Regressed to 28.57% — validator over-rejected | #67 (closed) |
| No procedures (B' ablation) | −14.29pp — procedures are net-positive | #69 (merged) |
| Procedure-layer escape (failure-streak + force-LLM-on-stuck) | 47.62% bimodal pre-fix; no lift | #70 (closed) |
| Self-reflection schedule density (every=10 vs every=30) | Ties baseline | #62, #64 |
| Step budget (600 steps) | Stuck at M4 for 461 frames | — |
| Model lineage (Gemma → Qwen 3.5) | Same upper bound | #71 (closed) → #81 (merged) |

## What's still in flight or open

| Lever | Status | Notes |
|---|---|---|
| **Stage K — cumulative cross-episode memory** (n=5, Gemma 26B, `--load-checkpoint --prev-run-id` chaining) | Running on GPU, ETA ~10:20Z | Will land on [PR #75](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/75). Pre-fix was REGRESS −14.28pp learning_delta. Post-fix prediction: floor holds at 57.14% but ceiling does not lift unless iter 1 stumbles into Viridian and captures a navigation procedure. |
| **Map-exit callouts in observation** | Untested, recommended next | Surface unvisited adjacent maps + their exit tile coordinates in every observation. Afternoon's work. |
| **Visited-maps sticky note** | Untested | Append `(map_name, first_step, last_step)` tuples to the observation. Cheap. |
| **Mini-map / map-graph view** | Untested | Compact rendering of connected maps reachable from current location, unvisited highlighted. |
| **Subtask injection at stuck-state** | Untested | When self-reflection emits *"stuck in movement loop"*, inject `"goal: leave the current map. try moving to each edge."` |
| **Stage J — Qwen3-Thinking** (always-on reasoning budget) | Scaffold in commit `e431e30`, **rerun deferred** | Pre-fix scored 28.57% × 3 — failed M3 (no starter). Post-fix re-run not queued; would only test reasoning budget, not the navigation gate. |

## Next move

Options 2 (visited-maps sticky note) or 1 (map-exit callouts) are the cheapest and most direct attack on the M5 navigation gate. Stage K (cumulative memory) is the in-flight check on whether learned procedures can break it. None of these need a new model or new training — they're observation-layer additions.

## Cross-refs

- Detailed history: [`docs/experiments/gemma/cross-stage-diagnosis.md`](experiments/gemma/cross-stage-diagnosis.md)
- Asm-fix root cause: [PR #80](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/80) (hardened `pyboy_runner.py`) + [PR #81](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/81) (Stage D+H reruns)
- Asm gap analysis: `docs/experiments/pokemon-asm-gap.md`
- Sweep convention: `~/.claude/CLAUDE.md` (ML sweep orchestration section) + `experiments/autoresearch.py` docstring
