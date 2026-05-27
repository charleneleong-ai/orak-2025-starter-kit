# SC2 intermediate reward shaping — design

**Status:** draft for review
**Author:** charlene
**Date:** 2026-05-27
**Branch:** `feat/sc2-reward-shaping`

## Motivation

The PR3 SC2 smoke (`stagnation_pr3_star_craft_smoke_20260527T094639Z`, 2h 7m, 9 episodes) finished cleanly with **0 victories** and a smoking-gun stats trajectory:

```
Step 10:   procedures_learned=0,  successful_executions=0
Step 200:  procedures_learned=5,  successful_executions=0
Step 2500: procedures_learned=68, successful_executions=0   ← still zero
```

Across 2500 steps the agent registered **zero successful executions**. `avg_procedure_success_rate=0.51` is computed from internal heuristics, not from any signal correlated with winning, so the procedural-memory refinement loop is running on noise. The detector layer (PR1 futile-action / PR2 repeated-plan / PR3 progress-stagnation) works as designed — but each "intervention" just rerolls into the same pool of weak procedures because there is no positive signal to distinguish good procedures from bad.

The proximate cause is that SC2 falls through to `GenericShaper` in [`agents/macla/online_evaluator.py:307-317`](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py#L307-L317), which only emits non-zero reward when `success=True` (i.e. victory) — never the case here. The other three games (`super_mario`, `twenty_fourty_eight`, `pokemon_red`) have rich purpose-built shapers; SC2 is the outlier.

This is also why pokemon Stages L/M/R/S have moved the needle — each iterated the per-game shaping signals. SC2 has never received that treatment.

## Goal

Add a `StarCraftShaper(RewardShaper)` that emits meaningful per-step shaped reward so MACLA's procedural memory has a non-noise signal to refine against. Scope: shaping only. Does not change the detector stack, MACLA agent, or SC2 adapter.

**Success criteria:**
- Re-running the existing 2500-iteration smoke through the new shaper assigns negative reward to the iter-201 failure state (`Mineral: 3980`, `Supply left: -15`, `Pylon count: 1` = floated + supply-blocked) and positive reward to the iter-51 productive state (worker training, mineral spent, building constructing).
- `successful_executions > 0` after a fresh n=1 SC2 smoke with the new shaper.

## Non-goals

- **Episode-end retrospective credit assignment** — lever (2) from the brainstorm. Separate PR.
- **Cross-run warm-start** / **pre-seed procedural memory** — already covered by [`feat(stage_k): cumulative cross-episode memory + trajectory introspection`](../commit/90dfccc) (#75).
- **Per-race specialization** — race-agnostic regex first. Hardcoded Protoss-specific bonuses (4-gate vs FFE etc.) would break Terran/Zerg.
- **Tuning the magnitudes empirically** — first-pass weights; if the replay validation in Section 3 shows weird ratios we tweak before the smoke.
- **Wiring shaped_reward into MACLA's `success_rate` update** — already wired via `_provide_feedback`. The shaper only needs to return non-zero values.

## Existing context

State representation: the SC2 adapter exposes a structured text summary in `obs.obs_str`. Sample from iter 51:

```
Summary 1: At 01:29 game time, our current StarCraft II situation is as follows:

Resources:
- Game time: 01:29
- Worker supply: 20
- Mineral: 515
- Supply left: 2
- Supply cap: 23
- Supply used: 21

Buildings:
- Nexus count: 1
- Pylon count: 1

Units:
- Probe count: 20

In Progress:
Building constructing:
- Constructing gateway count: 1
Unit producing:
- Producing probe count: 1
```

Fields are race-agnostic (Terran/Zerg surface analogues via the same `X count: N` pattern). The summary is emitted by [`evaluation_utils/mcp_game_servers/star_craft/game/star_craft_env.py:282-286`](../tree/feat/sc2-reward-shaping/evaluation_utils/mcp_game_servers/star_craft/game/star_craft_env.py#L282-L286) (`obs2text`).

Shaper contract: [`RewardShaper`](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py#L101-L125) base class — `extract_metrics(state) -> dict`, `compute_reward(prev, cur, success, is_fatal) -> float`, `reset_episode()`, `_clamp(reward)`. Registration in `SHAPERS` dict ([line 322](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py#L322)).

## Design

### 1. Architecture & placement

`StarCraftShaper` lives alongside the existing shapers in [`agents/macla/online_evaluator.py`](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py) and is registered in `SHAPERS`:

```python
SHAPERS = {
    "super_mario": MarioShaper,
    "twenty_fourty_eight": TwentyFortyEightShaper,
    "pokemon_red": PokemonShaper,
    "star_craft": StarCraftShaper,   # ← new
}
```

A `"star_craft"` entry is added to `DEFAULT_SHAPING` so all magic numbers are Hydra-overridable per the existing convention. No changes to `OnlineAgentEvaluator`, MACLA agent, or the SC2 adapter.

### 2. Signal extraction (`extract_metrics`)

Regex over the obs_str text:

| Field | Pattern | Purpose |
|---|---|---|
| `game_time_sec` | `Game time:\s*(\d+):(\d+)` → `mm*60+ss` | Survival baseline |
| `mineral` | `Mineral:\s*(\d+)` | Economy |
| `supply_used` | `Supply used:\s*(\d+)` | Army/worker count proxy |
| `supply_cap` | `Supply cap:\s*(\d+)` | Pylon/tech progression |
| `supply_left` | `Supply left:\s*(-?\d+)` | Supply-block detector |
| `worker_supply` | `Worker supply:\s*(\d+)` | Economy strength |
| `building_count` | sum of `(\w+) count:\s*(\d+)` matches **excluding** `Probe`/`Worker`/`Producing`/`Constructing` | Tech tree progression |
| `enemy_unit_count` | sum of `Enemy unittypeid\.\w+:\s*(\d+)` | Scouting / contact proxy |

Race-agnostic by construction: the regexes don't reference Pylon/SupplyDepot/Overlord specifically — `building_count` sums whatever structures show up.

### 3. Reward formula (`compute_reward`)

Combines per-step deltas + state-based penalties + terminal signals. All weights live in `DEFAULT_SHAPING["star_craft"]`.

| Signal | Default weight | Rationale |
|---|---|---|
| `success=True` (victory) | `+3.0` | Terminal positive — matches other shapers |
| `is_fatal=True` (defeat) | `-2.0` | Terminal negative |
| `Δ supply_used > 0` | `+0.2 × Δ` | Building army/workers |
| `Δ building_count > 0` | `+0.5 × Δ` | Tech tree progression |
| `Δ mineral > 0` AND `Δ supply_used == 0` | `-0.3` | **Idleness / floated-minerals penalty** |
| `supply_left <= 0` | `-0.5` | **Supply-block penalty** |
| first time `enemy_unit_count > 0` | `+0.5` (one-shot) | Scouting / first-contact bonus (mirrors `PokemonShaper` map-discovery) |
| `Δ game_time_sec > 0` baseline | `+0.05` | Survival increment |

Clamped to `[-2.0, +3.0]`. The two boldface penalties are the load-bearing fix for the iter-201 failure mode.

> **Why penalize floated minerals?** The smoke at iter 201 shows `Mineral: 3980`, `Supply left: -15`, `Pylon count: 1` — the agent has 4k minerals and can't spend them because it never built a Pylon. Without the idleness penalty, mineral accumulation alone could earn positive reward via a naive `Δ mineral` term, teaching procedural memory the wrong lesson. This mirrors the [PokemonShaper repeat-visit reward-hack warning](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py#L64-L67) — same class of failure.

Per-step rewards typically land in `[-1, +1]`, terminal in `[-2, +3]`, matching the other shapers' ranges.

### 4. Episode state

Inherits `_stagnation_count` from base. Tracks one extra piece of state: `_seen_enemy_unit: bool` — flipped to `True` the first time `enemy_unit_count > 0` (first scout / first contact), one-shot `+0.5` bonus to incentivize scouting. Mirrors `PokemonShaper._visited_maps`. Reset in `reset_episode()`.

### 5. Error handling

- Missing field → `0` (or `False` for `_seen_enemy_unit`). No exceptions, ever — the regexes can fail silently and the resulting `Δ` will be `0`.
- Negative `Δ` (e.g. mineral spent, supply lost to enemy) → not rewarded (the `> 0` guards prevent it) but also not penalized except for the explicit idleness/supply-block rules.
- Empty `prev` (first step of episode) → `prev = current` effectively, so all deltas are 0.

## Testing strategy

Two layers, matching the [test conventions in CLAUDE.md](#) (one file per area, parametrize aggressively, module-level fixtures):

### Unit tests — `tests/test_online_evaluator_starcraft.py`

| Test class | Cases |
|---|---|
| `TestExtractMetrics` | Parametrized over realistic obs strings lifted from the smoke log: full state, missing-fields, multi-summary, negative supply_left |
| `TestComputeReward` | Idleness penalty fires when mineral grows but supply_used flat; supply-block penalty fires on `supply_left <= 0`; supply-used delta rewards army-building; building-count delta rewards construction; first enemy unit fires one-shot bonus |
| `TestTerminal` | `is_fatal` → -2.0; `success` → +3.0; clamp respected |
| `TestEpisodeReset` | `_seen_enemy_unit` cleared on `reset_episode()`; `_stagnation_count` cleared |

~10-15 tests, all sub-second. Synthetic obs strings live as a module-level fixture (`@pytest.fixture` returning a dict of canonical obs strings from real smoke iterations 1/51/201/end).

### Replay validation — `experiments/sc2_replay_shaper.py`

Standalone script that re-runs the existing [`game_logs/star_craft/stagnation_pr3_star_craft_smoke_20260527T094639Z/game_states.jsonl`](../tree/feat/sc2-reward-shaping/game_logs/star_craft/stagnation_pr3_star_craft_smoke_20260527T094639Z/game_states.jsonl) (2500 iterations, already on disk) through `StarCraftShaper`. Reports:

- Cumulative reward per episode
- Count of idleness-penalty fires and supply-block-penalty fires
- Reward at iter 51 (productive state, should be positive) and iter 201 (failure state, should be negative)
- Per-episode reward histogram

Decision gate: **if reward at iter 201 < reward at iter 51, ship.** Otherwise tune weights and re-run. ~5s, no SC2 needed, re-runnable across weight tweaks.

### Smoke validation (after replay passes)

Single n=1 SC2 smoke on the same config as PR3 (`gemma_26b`, Flat64, Protoss vs Zerg D4, max_steps=2500, max_episodes=10). Decision gate: `successful_executions > 0` in the final MACLA Stats block. Victory not required for v1 — just non-zero positive signal.

## File touch list

| File | Action | Approx LOC |
|---|---|---|
| [`agents/macla/online_evaluator.py`](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py) | Add `DEFAULT_SHAPING["star_craft"]`, `StarCraftShaper` class, register in `SHAPERS` | +80 |
| [`tests/test_online_evaluator_starcraft.py`](../tree/feat/sc2-reward-shaping/tests/test_online_evaluator_starcraft.py) | NEW — unit tests | +150 |
| [`experiments/sc2_replay_shaper.py`](../tree/feat/sc2-reward-shaping/experiments/sc2_replay_shaper.py) | NEW — replay validation | +60 |
| [`docs/specs/2026-05-27-sc2-reward-shaping-design.md`](../tree/feat/sc2-reward-shaping/docs/specs/2026-05-27-sc2-reward-shaping-design.md) | NEW — this doc | (this doc) |

No changes to: SC2 adapter, MACLA agent, detector stack, run.py, configs.

## Out-of-scope follow-ups

- Episode-end retrospective credit assignment — separate framework-level PR
- Tuning weights via a Hydra sweep over `reward_shaping` overrides — backlog
- Cross-game audit: do `super_mario` / `twenty_fourty_eight` / `pokemon_red` have any analogous "weak shaping" failure modes? Probably not (they already have rich shapers) but worth a quick replay-style sanity check
