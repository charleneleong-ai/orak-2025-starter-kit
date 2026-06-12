# Qwen3.6-35B-A3B (Reasoning) — cross-game n=3 ceiling check

Sweep config: [`configs/qwen36_a3b_reasoning.yaml`](../../../configs/qwen36_a3b_reasoning.yaml) ·
launcher: [`experiments/qwen36_cross_game_n3/run_sweep.sh`](../../../experiments/qwen36_cross_game_n3/run_sweep.sh) ·
PR [#113](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/113)

## Hypothesis

Qwen3.6 reasoning (AAI 31 → 43 vs Qwen3.5 non-reasoning) is a drop-in upgrade; routing all four games to it should lift or hold every per-game ceiling at n=3 seeds.

## Results (normalised 0–100, mean over eval episodes)

| game | seed1 | seed2 | seed3 | notes |
|---|---|---|---|---|
| pokemon_red | 57.1 | 14.3 | 14.3 | raw milestone 0–7 → `/7×100`; seed1 M4, seed2/3 M1 |
| super_mario | 31.0 | 35.2 | 39.4 | already 0–100 |
| twenty_fourty_eight | 61.6 | 56.8 | 65.5 | already 0–100 |
| star_craft | 80.0 | 81.3 | 83.3 | **retro-rescored** (live metric was bugged → 0); seed3 partial |

## Finding 1 — star_craft live score was a measurement bug, not the model

[`StarCraftEnv.evaluate`](../../../evaluation_utils/mcp_game_servers/star_craft/game/star_craft_env.py#L376) parsed
`str(self.summary)` (underscore-keyed dict-repr) instead of the rendered obs text, so the
[`extract_metrics`](../../../evaluation_utils/mcp_game_servers/star_craft/progress.py#L62) regexes never matched and every
episode collapsed to ~0 (only the Victory rung could credit → 0 or 12.5). Fixed in PR
[#117](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/117) (`evaluate()` now reads `self.obs2text(obs)`).

Retro re-scoring the logged `game_states.jsonl` through the same ladder recovers the real scores: every seed
clears **7 of 8 milestones** (best 87.5, the economic/military rungs), win-rate 0 — the difficulty-4 bot is
effectively unbeatable, so the Victory rung (M8) never credits. star_craft is actually the agent's **strongest**
game, not its weakest.

## Finding 2 — pokemon: a navigation wall, and why the prior gains didn't propagate

All three seeds spent **91–98% of their 1200 steps perseverating in PalletTown**:

| seed | reached | milestone steps | PalletTown dwell | wall |
|---|---|---|---|---|
| 1 | M4 win 1st battle | M1@19 M2@25 M3@77 M4@105 | 1093/1200 (91%) | never reached Viridian (M5) |
| 2 | M1 leave house | M1@20 | 1181/1200 (98%) | never met Oak (M2) |
| 3 | M1 leave house | M1@38 | 1161/1200 (97%) | never met Oak (M2) |

seed1 proves the milestone *reasoning* works — it rattled off M1→M4 inside the first 105 steps, then stalled for
~1095 steps unable to walk north to Viridian. The cross-seed variance is essentially luck of stumbling into Oak's
Lab early. **The bottleneck is spatial traversal, not knowing what to do.**

**The dominant cause: the agent never emits an action.** Correlating the logged actions with the raw LLM
responses, **97% of steps (1167/1200) are `Method: fallback` — "Could not parse response content as the length
limit was reached".** Qwen3.6's reasoning chain runs to the `max_tokens` cap (8192) *before* it closes `<think>`
and emits the action, so the parser finds nothing and the harness defaults to **pressing A**. No directional
presses, almost no `move_to` (33/1200 tool calls). An agent that only presses A literally cannot leave PalletTown.

This reframes the earlier two "nav gains didn't propagate" theories:

1. **Map-graph hint efficacy is untestable, not disproven.** [`graph_hint`](../../../agents/pokemon_red/game_adapter.py#L206)
   *did* fire (502/1200, e.g. `OaksLab: walk to (12, 11)`), but with 97% of steps truncated before any action, the
   model rarely got to act on it. This is NOT the spatial-grounding gap first hypothesised — it's an action-budget
   gap. (For contrast, the Gemma-era stages reliably reached M4/57% and Route 1 with the same hint + `move_to`.)
2. **Stage R v4 loop-counter was genuinely broken** — separate from the truncation. The hint fired **0/1200**
   despite a tile being revisited 980× cumulatively, because [`__setstate__`](../../../agents/macla/macla_lib.py)
   zeroed `_position_visits` on every checkpoint round-trip (the agent is round-tripped far more often than once
   per iter). Fixed in [#118](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/118): the counter now
   survives round-trips and resets once per episode at the agent's `_subgoal_init_done` gate. (Near-moot for
   pokemon until the truncation is fixed — the hint can't help an agent that never emits an action.)

## Verdict

- Qwen3.6 holds 2048, edges super_mario up across seeds, and (post-fix) is strong at star_craft.
- pokemon's apparent navigation collapse is dominated by **reasoning-truncation → press-A fallback**, not a
  spatial-reasoning deficit. The model is competent (seed1 cleared M1→M4 in 105 steps before reasoning ballooned);
  it just runs out of token budget mid-think on 97% of steps.

## Post-fix rerun — pokemon n=3 (truncfix + #118)

Reran pokemon-only n=3 with `max_tokens` 8192→10240, `max_model_len` 12288→16384, and the #118 loop-counter fix.
The truncation fix landed decisively — **0% of completions hit the cap** (mean ~2645 tokens, max 5103 vs the old
8192 cap), and actions flipped from **97% press-A → 94% real `move_to`**. Both nav hints now fire in production:
`map_graph_hint` ~95% of steps and `looped_positions_hint` ~89% (the latter was **0/1200** pre-#118).

### Final ceiling — all three seeds complete (1200/1200)

| seed | reached | norm | milestone steps | top maps (dwell) | move_to |
|---|---|---|---|---|---|
| baseline s1 (bugged) | M4 | 57% | M1@19 … M4@105 | PalletTown 91% | 3% |
| baseline s2/3 (bugged) | M1 | 14% | M1@20 / M1@38 | PalletTown 98% | ~3% |
| **rerun s1** | **M5** | **71%** | M1@12 M2@21 M3@58 M4@74 **M5@172** | ViridianCity 64% · Route1 17% · Pallet 8% | 57% |
| **rerun s2** | **M5** | **71%** | M1@5 M2@11 M3@73 M4@94 **M5@145** | ViridianCity 55% · Route1 20% · Pallet 15% | 60% |
| **rerun s3** | **M5** | **71%** | M1@11 M2@15 M3@62 M4@85 **M5@166** | ViridianCity 84% · Route1 6% · Pallet 1% | 63% |

Unanimous: every seed reaches **Viridian (M5/71%) by ~step 170**, then plateaus for the remaining ~1030 steps.
None breaks **M6 (OAK's PARCEL)**. Two things confirmed:

1. **The truncation diagnosis was right.** PalletTown lock broken (91–98% → 1–15% dwell), actions flipped 97%
   press-A → 57–63% real `move_to`, and the M5 ceiling — which the *bugged* seed1 never reached in 1200 steps,
   the Gemma-era best — is now hit by all three in ~170 steps. pokemon was a reasoning-truncation failure, not a
   spatial-reasoning deficit.
2. **The wall relocated, it didn't vanish.** Dwell moved PalletTown → **ViridianCity** (55–84%). The agent is no
   longer lost *getting* to Viridian; it's pacing the right map unable to **enter the Mart + talk to the clerk**
   for the parcel. That's an *interaction* stall, not a spatial one — every spatial signal reads "all good", so
   `graph_hint`/loop-counter can't fire.

## Next move

The M5→M7 tail is now the sole open ceiling, and it's an **interaction-policy** wall — exactly the failure
mode addressed by the generalised interaction-sweep (milestone-stall detector → graduated hint→override),
[#119](https://github.com/charleneleong-ai/orak-2025-starter-kit/pull/119). The live validation of that
mechanism is a sweep-enabled pokemon rerun checking whether it breaks M6; this n=3 set is its baseline.
