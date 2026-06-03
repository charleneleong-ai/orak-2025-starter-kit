# StarCraft II Progress Metric — Design

**Date:** 2026-06-03
**Branch:** `feat/starcraft-progress-metric` (off `master`)

## Problem

`star_craft_env.evaluate()` returned a **binary** score — `1` on Victory, else `0`. No
LLM wins a full SC2 match, so every model scored a flat `0.0` and the ceiling check had
**zero discriminative power** for starcraft. Yet the agent makes real, measurable progress:
on the Qwen3.6 seed-1 run it built ~31 supply / 12 buildings / 23 workers and engaged the
enemy in 5 of 6 episodes — all invisible to the binary eval.

The per-step progression signal already existed (`StarCraftShaper.extract_metrics`, PR #111)
but only fed the agent's MACLA reward, never the eval.

## Goal

A normalized **0–100 `starcraft_progress`** score, aligned with how the other orak games
score (peak progression → 0–100), discriminating models by economic/military progress.
Keep the binary win-rate as a secondary `star_craft_victory` field for Orak comparability.

## Alignment

Every orak game scores *peak progression normalized to 0–100*: pokemon_red = ordered
milestones (`milestones / N · 100`), 2048 = peak max_tile via `normalize_2048_score`,
mario = level progression. starcraft now mirrors pokemon's **milestone ladder**.

## The metric

Peak-based (step-count-invariant), per episode, mean across episodes (best-episode secondary):

```
M1 built first structure   building_count ≥ 2     M5 tech / production   building_count ≥ 8
M2 expanded supply (Pylon) supply_cap     ≥ 23     M6 larger army         supply_used   ≥ 34
M3 built up economy        worker_supply  ≥ 16     M7 engaged enemy       enemy_units   ≥ 1
M4 trained army            supply_used    ≥ 20     M8 Victory             victory == 1

starcraft_progress(ep) = milestones_reached / 8 · 100 ;  run = mean over episodes
```

Rungs counted independently → monotonic (more peak progress never lowers the score).

### Threshold calibration (frozen)

Calibrated from Qwen3.6 seed-1's 6-episode peak distribution (supply_used 19–38, buildings
4–14, workers 18–28, supply_cap 23–47, enemy contact 5/6) and frozen — all seeds/models on
one ruler.

**Validated on the real seed-1 log:** run mean **75.0%**, per-episode 50 / 75 / 75 / 75 /
87.5 / 87.5, best 87.5%, win-rate 0.0 — vs the flat `0.0` under binary. Typical episodes
clear 6/8 rungs; the two best also clear M6 (supply ≥ 34); the short game drops to 50%. A
weaker model falls to the floor rungs (≤37.5%); a winner reaches 100%.

## Components

- **`star_craft/progress.py`** — pure stdlib module (no SC2 deps), single source
  of truth: `extract_metrics` (obs→metrics regexes), `split_episodes` (game_time reset),
  `episode_peaks`, `milestone_score`, `run_progress`.
- **`StarCraftShaper.extract_metrics`** — refactored to **delegate** to the module parser, so
  the shaper's reward signal and the eval metric cannot drift on extraction.
- **`star_craft_env`** — `evaluate()` accumulates per-episode peak metrics and returns
  `milestone_score`; `reset()` clears them; `get_game_info` exposes `star_craft_victory`.
  `base_game_logic` already averages per-episode final scores → `avg_score` == `run_progress`
  mean, no further runner change.

## Testing

34 tests: rung boundaries + monotonicity, episode splitting (reset guard, empty), peak
extraction (peak-not-terminal) + incremental `merge_peaks`, obs parsing (real field formats,
building exclusion, absent fields), and run aggregation (mean/best/win-rate). The live env
wiring reuses these tested functions; validated end-to-end against the real seed-1 log
(reproducing 75.0% via `run_progress`).

## Out of scope / caveats

- Other games' scoring unchanged.
- `starcraft_progress` is non-Orak-standard; `star_craft_victory` retained for comparison.
  Writeups note cross-game means use `starcraft_progress`.
- The live env change scores future runs natively. Backfilling runs recorded before it (e.g.
  the in-flight qwen36 sweep) is a one-off operational step using the public `run_progress` /
  `extract_metrics` over their `game_states.jsonl` — not a committed script.
