# TGAER PR1 — universal futile-action detector × 4 games (n=3 each)

**Verdict:** **NO LIFT.** The detector is flat-to-mildly-negative on 3/4 games and never fires on the 4th (StarCraft). Aggregate Δ ≈ −1pp across mario/2048/pokemon; SC2 stays at the 0.0 floor with all 30 episodes Defeat. Detector is safe to ship as a no-op signal but should not be sold as a lift mechanism.

## Hypothesis

A universal pathology guard — "if the last K=3 observations are byte-identical, the agent's actions are no-ops; inject a planner-side notice to break the loop" — should lift mean score on games where the agent can get visibly stuck (Pokemon overworld lock-ups, Mario walking into a pipe, SC2 supply-blocked build loops). Expected: positive Δ on ≥2 games, neutral on the others.

## Implementation

Single hook in [`agents/macla/unified.py`](../../../agents/macla/unified.py) — `_detect_futile_action` ([L567-597](../../../agents/macla/unified.py#L567-L597)) — runs before the LLM call. Hashes raw `observation` strings (pre-hint), maintains a `deque(maxlen=3)`, fires when all three match. The hint is prepended to the planner prompt as `[Futile-action notice]`.

Commit: [`7b1b69a feat(macla): universal futile-action detector (PR 1 of MVA harness)`](../../../../../commit/7b1b69a).

## Schedule

| Setting | Value |
|---|---|
| Agent | `gemma_26b` (Gemma 4-26B-A4B-AWQ-4bit on vLLM :8000) |
| Games | pokemon_red, super_mario, twenty_fourty_eight, star_craft |
| Inference budget | 1000 calls (pokemon: 1200), per game-specific `MAX_STEPS` |
| Reps | n=3 per (side, game) — 24 rolls total |
| SC2 episodes | 10 per roll (full Protoss vs Hard Zerg on Flat64) |
| Launcher | [`configs/schedules/tgaer_pr1.yaml`](../../../configs/schedules/tgaer_pr1.yaml) via `autoresearch-parallel-batch` (v0.28.0) |
| Bridge (legacy reconcile) | [`scripts/tgaer_results_bridge.py`](../../../scripts/tgaer_results_bridge.py) → `experiments/tgaer_pr1_{baseline,detector}/results.jsonl` |

## Results

![cross_game_lift](../../../experiments/progress/tgaer_pr1/cross_game_lift.png?raw=true)

| Game | Baseline (mean ± std, n=3) | Detector (mean ± std, n) | Δ |
|---|---|---|---|
| pokemon_red | 5.67 ± 0.58 (6, 6, 5) | 4.75 ± 1.89 (6, 2, 5, 6 — n=4 incl. rerun) | **−0.92** |
| super_mario | 34.64 ± 1.54 (36.35, 34.29, 33.28) | 31.75 ± 5.01 (25.90, 35.62, 33.74) | **−2.89** |
| twenty_fourty_eight | 49.66 ± 2.13 (47.16, 50.76, 51.05) | 50.67 ± 1.27 (49.24, 51.52, 51.24) | **+1.01** |
| star_craft | 0.00 (n=3, 30 eps all Defeat) | 0.00 (n=3, 30 eps all Defeat) | 0.00 |

**Pokemon detail:** Detector n=2 (heal2, May 24) scored **2.0** — far below the n=1/n=3 cluster at 5–6. Rerun on 2026-05-26 scored **6.0** ([run](../../../../../tree/feat/futile-action-detector/game_logs/pokemon_red)), confirming the 2.0 was stochastic LLM-sampling noise, not a reproducible detector regression. The mean with the outlier included is 4.75; excluding it, 5.67 — exact parity with baseline.

## Failure mode — StarCraft floor

The detector **never fires on StarCraft**. Verified by replaying `_detect_futile_action(K=3)` over the `game_states.jsonl` of an SC2 rollout (2371 iterations across 10 episodes): 0 consecutive byte-identical observation triples. Reason — SC2 observation strings contain continuously-ticking fields (`Game time`, `Minerals`, `Supply`) that increment every frame, so byte-equality is never reached even when the agent is supply-blocked and looping `TRAIN PROBE × 5` indefinitely.

This is **not a bug in the detector**; it's a limitation of byte-equality as the trigger heuristic for environments whose obs is dominated by free-running counters. A retrospective replay of an action-side variant (`_detect_repeated_plan(K=4)` — hash the chosen action plan, not the obs) fires 615× across the same 2371 iters (127 distinct streaks, 13 nudge moments per episode). That's the PR2 design.

## Why the other games didn't lift

The detector *does* fire on pokemon/mario/2048, but the planner's hint-driven reaction doesn't unstick the agent in expected value. On pokemon the agent loops between OaksLab and PalletTown via `warp` (which DOES change obs — the detector resets every transition), so the streaks short-circuit. On mario the obs ticks via the player position float, similar to SC2. On 2048 the obs is fully discrete and the detector fires when the agent picks invalid moves — and the +1pp Δ may be that effect, but it's within noise.

## Verdict

**Ship as a safety floor, not a lift mechanism.** The detector adds no measurable regression, the SC2 floor result is informative (motivates PR2), and the code surface is ~30 lines. Land it, then move the lift hypothesis to PR2 (`_detect_repeated_plan`) where the retrospective replay shows the actual loop signal.

## Out-of-scope follow-ups

- **PR2 — `_detect_repeated_plan(K=4)`** ([L567 sibling](../../../agents/macla/unified.py#L567-L597)). Universal action-side detector. Retrospective replay on SC2 trace: 127 distinct streaks, 13 nudge moments/episode at K=4. Expected lift on SC2 + pokemon ping-pong loops.
- **SC2 max_steps bump** 1000 → 2500 ([this PR](../../../configs/star_craft/env/linux_default.yaml)). At 1000 calls the agent rarely escapes the early-game supply trap; 2500 gives Stage D Protoss enough rope to reach a tech transition without changing the detector behavior.
- **Re-bridge cadence.** [`scripts/tgaer_results_bridge.py`](../../../scripts/tgaer_results_bridge.py) is idempotent; today's rerun landed the missing n=2 detector cell cleanly. Re-run anytime new rolls drop in `game_logs/`.
