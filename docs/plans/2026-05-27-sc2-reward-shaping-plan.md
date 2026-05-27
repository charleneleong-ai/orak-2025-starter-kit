# SC2 Reward Shaping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `StarCraftShaper` so MACLA gets a non-zero per-step shaped reward on SC2, unblocking procedural-memory refinement (the smoke gun: `successful_executions=0` across 2500 steps of PR3 smoke).

**Architecture:** Single new `StarCraftShaper(RewardShaper)` subclass in `agents/macla/online_evaluator.py` with regex over the obs_str `Resources/Buildings/Units/Enemy` sections. Race-agnostic patterns (works for Protoss/Terran/Zerg). Magic numbers live in `DEFAULT_SHAPING["star_craft"]` for Hydra overrides. TDD per CLAUDE.md, ~5 small commits.

**Tech Stack:** Python 3.11+, pytest, ruff (line-length=100), uv for env management. Existing project on branch `feat/sc2-reward-shaping` off `origin/master`.

**Spec:** [`docs/specs/2026-05-27-sc2-reward-shaping-design.md`](../specs/2026-05-27-sc2-reward-shaping-design.md)

**Working directory:** `/workspace/orak-futile-detector` (this worktree). All paths below are relative to this directory.

---

## Task 0: Commit the spec doc

**Files:**
- Modify: `docs/specs/2026-05-27-sc2-reward-shaping-design.md` (already written, uncommitted)

- [ ] **Step 1: Verify branch and clean state**

```bash
git rev-parse --abbrev-ref HEAD  # expect: feat/sc2-reward-shaping
git status -s                    # expect: ?? docs/specs/2026-05-27-sc2-reward-shaping-design.md  +  ?? docs/plans/2026-05-27-sc2-reward-shaping-plan.md
```

- [ ] **Step 2: Commit spec + plan**

```bash
git add docs/specs/2026-05-27-sc2-reward-shaping-design.md docs/plans/2026-05-27-sc2-reward-shaping-plan.md
git commit -m "$(cat <<'EOF'
docs(macla): SC2 intermediate reward shaping spec + plan

PR3 SC2 smoke surfaced successful_executions=0 across 2500 steps —
procedural-memory refinement is running on noise because SC2 falls
through to GenericShaper (returns 0 unless success=True, which never
fires without victories). Spec proposes a StarCraftShaper following
the existing per-game pattern (MarioShaper / TwentyFortyEightShaper /
PokemonShaper) with race-agnostic regex over obs_str and load-bearing
idleness + supply-block penalties to teach the procedural memory which
states are bad (e.g. iter-201: 3980 minerals floated + supply-blocked).
EOF
)"
```

---

## Task 1: Scaffold `StarCraftShaper` + `DEFAULT_SHAPING` entry + register

**Files:**
- Modify: `agents/macla/online_evaluator.py` — add `DEFAULT_SHAPING["star_craft"]`, `StarCraftShaper` class scaffold, register in `SHAPERS`
- Create: `tests/test_online_evaluator_starcraft.py` — test file with module-level fixtures + import smoke test

- [ ] **Step 1: Write the failing import test + a registry-smoke test**

Create `tests/test_online_evaluator_starcraft.py`:

```python
"""Tests for StarCraftShaper — per-step reward shaping for the SC2 adapter.

Fixtures lift canonical obs_str snippets from a real PR3 smoke run
(stagnation_pr3_star_craft_smoke_20260527T094639Z) at iterations 1 / 51 / 201,
covering empty-init / productive-state / floated-supply-blocked respectively.
"""

import pytest

from agents.macla.online_evaluator import (
    DEFAULT_SHAPING,
    SHAPERS,
    OnlineAgentEvaluator,
    StarCraftShaper,
)


# ── Canonical obs strings lifted from real smoke iterations ─────────────────


@pytest.fixture(scope="module")
def obs_strings() -> dict[str, str]:
    return {
        "iter_1_empty": "",
        "iter_51_productive": (
            "Summary 1: At 01:29 game time, our current StarCraft II situation is as follows:\n"
            "\n"
            "Resources:\n"
            "- Game time: 01:29\n"
            "- Worker supply: 20\n"
            "- Mineral: 515\n"
            "- Supply left: 2\n"
            "- Supply cap: 23\n"
            "- Supply used: 21\n"
            "\n"
            "Buildings:\n"
            "- Nexus count: 1\n"
            "- Pylon count: 1\n"
            "\n"
            "Units:\n"
            "- Probe count: 20\n"
            "\n"
            "In Progress:\n"
            "Building constructing:\n"
            "- Constructing gateway count: 1\n"
            "Unit producing:\n"
            "- Producing probe count: 1\n"
        ),
        "iter_201_floated": (
            "Summary 1: At 05:56 game time, our current StarCraft II situation is as follows:\n"
            "\n"
            "Resources:\n"
            "- Game time: 05:56\n"
            "- Worker supply: 23\n"
            "- Mineral: 3980\n"
            "- Supply left: -15\n"
            "- Supply cap: 8\n"
            "- Supply used: 23\n"
            "\n"
            "Buildings:\n"
            "- Pylon count: 1\n"
            "- Gateway count: 2\n"
            "\n"
            "Units:\n"
            "- Probe count: 23\n"
            "\n"
            "Enemy:\n"
            "\n"
            "Unit:\n"
            "- Enemy unittypeid.zergling: 3\n"
            "- Enemy unittypeid.ravager: 3\n"
            "- Enemy unittypeid.roach: 4\n"
        ),
    }


@pytest.fixture
def shaper() -> StarCraftShaper:
    return StarCraftShaper(DEFAULT_SHAPING["star_craft"])


class TestRegistry:
    def test_star_craft_registered_in_SHAPERS(self):
        assert SHAPERS["star_craft"] is StarCraftShaper

    def test_default_shaping_has_star_craft_entry(self):
        assert "star_craft" in DEFAULT_SHAPING
        s = DEFAULT_SHAPING["star_craft"]
        # required keys (other tasks reference these — fail loud if missing)
        for key in (
            "reward_min",
            "reward_max",
            "fatal_penalty",
            "victory_bonus",
            "supply_used_weight",
            "building_built_weight",
            "floated_minerals_penalty",
            "supply_block_penalty",
            "first_enemy_bonus",
            "survival_increment",
        ):
            assert key in s, f"missing DEFAULT_SHAPING['star_craft']['{key}']"

    def test_evaluator_routes_star_craft_to_shaper(self):
        ev = OnlineAgentEvaluator("star_craft")
        assert isinstance(ev._shaper, StarCraftShaper)
```

- [ ] **Step 2: Run test — verify it fails (RED)**

```bash
cd /workspace/orak-futile-detector
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestRegistry -v
```

Expected: FAIL with `ImportError: cannot import name 'StarCraftShaper' from 'agents.macla.online_evaluator'`.

- [ ] **Step 3: Add `DEFAULT_SHAPING["star_craft"]` entry**

In `agents/macla/online_evaluator.py`, inside the `DEFAULT_SHAPING` dict (after the `pokemon_red` entry around line 76), add:

```python
    "star_craft": {
        # Terminal rewards
        "fatal_penalty": -2.0,   # defeat / game over
        "victory_bonus": 3.0,
        # Per-step positive deltas
        "supply_used_weight": 0.2,     # per unit of supply (worker or army) built
        "building_built_weight": 0.5,  # per structure built
        "survival_increment": 0.05,    # tiny baseline reward when game_time advances
        "first_enemy_bonus": 0.5,      # one-shot when we first see an enemy unit
        # State-based penalties (the load-bearing fix)
        #
        # Floated-minerals penalty fires when mineral grows but supply_used
        # is flat — i.e. the agent is collecting resources without spending
        # them on units. Symptom from PR3 smoke iter 201: 3980 minerals + only
        # 1 Pylon + supply-blocked. Without this penalty, a naive Δ-mineral
        # term would reward the same state. Mirrors the repeat_visit_bonus
        # warning above: same class of reward hack.
        "floated_minerals_penalty": -0.3,
        # Supply-block fires when supply_left <= 0 — the agent cannot train
        # new units regardless of mineral. Critical state to penalize.
        "supply_block_penalty": -0.5,
        # Stagnation (no game_time progress for N steps in a row)
        "stagnation_threshold_steps": 3,
        "stagnation_penalty": -0.3,
        # Clamp range matches the other shapers
        "reward_min": -2.0,
        "reward_max": 3.0,
    },
```

- [ ] **Step 4: Add scaffold `StarCraftShaper` class**

In `agents/macla/online_evaluator.py`, after the `PokemonShaper` class (around line 301) and before `GenericShaper`, add:

```python
# ── StarCraft II ────────────────────────────────────────────


class StarCraftShaper(RewardShaper):
    """Per-step shaped reward for the SC2 adapter.

    Reads structured fields from the obs_str text summary emitted by
    star_craft_env.obs2text: `Game time`, `Mineral`, `Supply used/cap/left`,
    `Worker supply`, building counts, and enemy-unit counts. Race-agnostic
    by construction — the regexes don't reference Pylon/SupplyDepot/Overlord
    specifically.

    The load-bearing signal is the idleness + supply-block penalties: PR3 smoke
    showed avg_procedure_success_rate=0.51 across 2500 steps with zero
    successful_executions — procedural memory had nothing to refine against.
    Without these penalties, mere mineral accumulation would still earn
    positive reward, teaching the wrong lesson.
    """

    # Buildings list — sum of all `X count: N` matches excluding workers
    # and in-progress markers (Probe/Worker/Producing/Constructing).
    _BUILDING_EXCLUDE = ("Probe", "Worker", "Producing", "Constructing")

    def __init__(self, shaping: dict):
        super().__init__(shaping)
        self._seen_enemy_unit: bool = False

    def reset_episode(self) -> None:
        super().reset_episode()
        self._seen_enemy_unit = False

    def extract_metrics(self, state: str) -> dict:
        return {}

    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        return 0.0
```

- [ ] **Step 5: Register in `SHAPERS`**

In `agents/macla/online_evaluator.py`, in the `SHAPERS` dict (around line 322-326):

```python
SHAPERS: dict[str, type[RewardShaper]] = {
    "super_mario": MarioShaper,
    "twenty_fourty_eight": TwentyFortyEightShaper,
    "pokemon_red": PokemonShaper,
    "star_craft": StarCraftShaper,
}
```

- [ ] **Step 6: Run test — verify it passes (GREEN)**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestRegistry -v
```

Expected: 3 PASSED.

- [ ] **Step 7: Lint**

```bash
.venv/bin/ruff check tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
.venv/bin/ruff format tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
```

- [ ] **Step 8: Commit**

```bash
git add tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
git commit -m "feat(macla): scaffold StarCraftShaper + DEFAULT_SHAPING entry

Empty extract_metrics/compute_reward stubs. Subsequent commits fill in
the regex extraction and reward formula via TDD."
```

---

## Task 2: Implement `extract_metrics`

**Files:**
- Modify: `tests/test_online_evaluator_starcraft.py` — add `TestExtractMetrics` class
- Modify: `agents/macla/online_evaluator.py` — fill in `StarCraftShaper.extract_metrics`

- [ ] **Step 1: Write failing tests for extract_metrics**

Add to `tests/test_online_evaluator_starcraft.py` after `TestRegistry`:

```python
class TestExtractMetrics:
    def test_empty_state_returns_zero_defaults(self, shaper, obs_strings):
        m = shaper.extract_metrics(obs_strings["iter_1_empty"])
        assert m["game_time_sec"] == 0
        assert m["mineral"] == 0
        assert m["supply_used"] == 0
        assert m["supply_cap"] == 0
        assert m["supply_left"] == 0
        assert m["worker_supply"] == 0
        assert m["building_count"] == 0
        assert m["enemy_unit_count"] == 0

    def test_productive_state_extracts_all_fields(self, shaper, obs_strings):
        m = shaper.extract_metrics(obs_strings["iter_51_productive"])
        assert m["game_time_sec"] == 89  # 01:29 = 89s
        assert m["mineral"] == 515
        assert m["supply_used"] == 21
        assert m["supply_cap"] == 23
        assert m["supply_left"] == 2
        assert m["worker_supply"] == 20
        # building_count: Nexus(1) + Pylon(1) = 2  (Probe/Constructing/Producing excluded)
        assert m["building_count"] == 2
        assert m["enemy_unit_count"] == 0

    def test_floated_state_extracts_negative_supply_left(self, shaper, obs_strings):
        m = shaper.extract_metrics(obs_strings["iter_201_floated"])
        assert m["game_time_sec"] == 356  # 05:56 = 356s
        assert m["mineral"] == 3980
        assert m["supply_left"] == -15
        # building_count: Pylon(1) + Gateway(2) = 3
        assert m["building_count"] == 3
        # enemy_unit_count: zergling(3) + ravager(3) + roach(4) = 10
        assert m["enemy_unit_count"] == 10

    @pytest.mark.parametrize(
        "field,patched,expected_key,expected_value",
        [
            ("Game time: 01:29", "Game time: 12:34", "game_time_sec", 754),
            ("Mineral: 515", "Mineral: 9999", "mineral", 9999),
            ("Supply left: 2", "Supply left: -1", "supply_left", -1),
        ],
    )
    def test_extract_handles_value_variations(
        self, shaper, obs_strings, field, patched, expected_key, expected_value
    ):
        state = obs_strings["iter_51_productive"].replace(field, patched)
        m = shaper.extract_metrics(state)
        assert m[expected_key] == expected_value
```

- [ ] **Step 2: Run tests — verify they fail (RED)**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestExtractMetrics -v
```

Expected: 6 FAILED (extract_metrics returns `{}`, so all field accesses KeyError).

- [ ] **Step 3: Implement extract_metrics**

In `agents/macla/online_evaluator.py`, replace the empty `StarCraftShaper.extract_metrics` body with:

```python
    def extract_metrics(self, state: str) -> dict:
        # Game time `mm:ss` → seconds. Use the LAST match so multi-summary
        # states reflect the most recent frame.
        gt_matches = re.findall(r"Game time:\s*(\d+):(\d+)", state)
        if gt_matches:
            mm, ss = gt_matches[-1]
            game_time_sec = int(mm) * 60 + int(ss)
        else:
            game_time_sec = 0

        # Building count: sum all "X count: N" matches except worker/in-progress.
        building_count = 0
        for name, n in re.findall(r"([\w ]+) count:\s*(\d+)", state):
            if any(excluded in name for excluded in self._BUILDING_EXCLUDE):
                continue
            building_count += int(n)

        # Enemy unit count: sum all "Enemy unittypeid.X: N" matches.
        enemy_unit_count = sum(
            int(n) for n in re.findall(r"Enemy unittypeid\.\w+:\s*(\d+)", state)
        )

        return {
            "game_time_sec": game_time_sec,
            "mineral": _find_int(r"Mineral:\s*(\d+)", state) or 0,
            "supply_used": _find_int(r"Supply used:\s*(\d+)", state) or 0,
            "supply_cap": _find_int(r"Supply cap:\s*(\d+)", state) or 0,
            "supply_left": _find_int(r"Supply left:\s*(-?\d+)", state) or 0,
            "worker_supply": _find_int(r"Worker supply:\s*(\d+)", state) or 0,
            "building_count": building_count,
            "enemy_unit_count": enemy_unit_count,
        }
```

Note: the existing `_find_int` helper at line 88 only handles positive `\d+` — for `supply_left` we need negative-number support. Update the helper signature is risky (used by other shapers). Instead the call passes `(-?\d+)` and `_find_int` already does `int(m.group(1))` which handles negatives correctly. Verify in the test.

- [ ] **Step 4: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestExtractMetrics -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Run lint + full test suite to check no regressions**

```bash
.venv/bin/ruff check tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
.venv/bin/ruff format tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
.venv/bin/pytest tests/test_online_evaluator_starcraft.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
git commit -m "feat(macla): StarCraftShaper.extract_metrics — regex over obs_str

Race-agnostic patterns over Game time / Mineral / Supply / Worker
supply / building counts / enemy unit counts. Multi-summary states use
the LAST 'Game time' match to reflect the most recent frame."
```

---

## Task 3: Implement `compute_reward` — terminal cases

**Files:**
- Modify: `tests/test_online_evaluator_starcraft.py` — add `TestTerminal` class
- Modify: `agents/macla/online_evaluator.py` — fill in `compute_reward` for `is_fatal` / `success`

- [ ] **Step 1: Write failing tests for terminal rewards**

Add to `tests/test_online_evaluator_starcraft.py`:

```python
class TestTerminal:
    def test_is_fatal_returns_fatal_penalty(self, shaper):
        r = shaper.compute_reward(prev={}, cur={}, success=False, is_fatal=True)
        assert r == DEFAULT_SHAPING["star_craft"]["fatal_penalty"]

    def test_success_returns_victory_bonus(self, shaper):
        r = shaper.compute_reward(prev={}, cur={}, success=True, is_fatal=False)
        assert r == DEFAULT_SHAPING["star_craft"]["victory_bonus"]

    def test_is_fatal_takes_precedence_over_success(self, shaper):
        # Defensive: if both flags somehow set, defeat wins (no false positives).
        r = shaper.compute_reward(prev={}, cur={}, success=True, is_fatal=True)
        assert r == DEFAULT_SHAPING["star_craft"]["fatal_penalty"]
```

- [ ] **Step 2: Run tests — verify RED**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestTerminal -v
```

Expected: 3 FAILED (compute_reward returns 0.0).

- [ ] **Step 3: Implement terminal branches**

In `agents/macla/online_evaluator.py`, replace the empty `StarCraftShaper.compute_reward` body with:

```python
    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping

        if is_fatal:
            return s["fatal_penalty"]
        if success:
            return s["victory_bonus"]

        return self._clamp(0.0)
```

- [ ] **Step 4: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestTerminal -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
git commit -m "feat(macla): StarCraftShaper.compute_reward — terminal cases

is_fatal → fatal_penalty, success → victory_bonus, defeat takes
precedence if somehow both flags are set."
```

---

## Task 4: Implement `compute_reward` — positive deltas

**Files:**
- Modify: `tests/test_online_evaluator_starcraft.py` — add `TestPositiveDeltas` class
- Modify: `agents/macla/online_evaluator.py` — extend `compute_reward`

- [ ] **Step 1: Write failing tests for positive deltas**

Add to `tests/test_online_evaluator_starcraft.py`:

```python
class TestPositiveDeltas:
    def _metrics(self, **overrides):
        """Helper: build a fully-populated metrics dict with overrides."""
        base = {
            "game_time_sec": 100,
            "mineral": 500,
            "supply_used": 20,
            "supply_cap": 23,
            "supply_left": 3,
            "worker_supply": 18,
            "building_count": 2,
            "enemy_unit_count": 0,
        }
        return {**base, **overrides}

    def test_supply_used_delta_rewards_unit_built(self, shaper):
        prev = self._metrics(supply_used=20)
        cur = self._metrics(supply_used=22)  # 2 supply built (e.g. zealot)
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # 2 * 0.2 = 0.4 (army) + 0.05 (survival baseline) = 0.45
        # (No idleness penalty because Δ supply_used > 0.)
        assert r == pytest.approx(0.45)

    def test_building_count_delta_rewards_structure_built(self, shaper):
        prev = self._metrics(building_count=2)
        cur = self._metrics(building_count=3)  # 1 new structure
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # 1 * 0.5 = 0.5 (building) + 0.05 (survival) = 0.55
        assert r == pytest.approx(0.55)

    def test_survival_increment_when_only_time_advances(self, shaper):
        prev = self._metrics(game_time_sec=100)
        cur = self._metrics(game_time_sec=110)
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # Just survival baseline.
        assert r == pytest.approx(0.05)

    def test_no_survival_increment_if_time_did_not_advance(self, shaper):
        prev = self._metrics(game_time_sec=100)
        cur = self._metrics(game_time_sec=100)
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        assert r == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests — verify RED**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestPositiveDeltas -v
```

Expected: 4 FAILED.

- [ ] **Step 3: Extend compute_reward with positive-delta logic**

In `agents/macla/online_evaluator.py`, replace `StarCraftShaper.compute_reward` body:

```python
    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping

        if is_fatal:
            return s["fatal_penalty"]
        if success:
            return s["victory_bonus"]

        reward = 0.0

        # Survival baseline: tiny constant when game_time advances.
        if cur.get("game_time_sec", 0) > prev.get("game_time_sec", 0):
            reward += s["survival_increment"]

        # Supply_used delta — army / worker built.
        supply_delta = cur.get("supply_used", 0) - prev.get("supply_used", 0)
        if supply_delta > 0:
            reward += s["supply_used_weight"] * supply_delta

        # Building delta — structure built.
        building_delta = cur.get("building_count", 0) - prev.get("building_count", 0)
        if building_delta > 0:
            reward += s["building_built_weight"] * building_delta

        return self._clamp(reward)
```

- [ ] **Step 4: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestPositiveDeltas tests/test_online_evaluator_starcraft.py::TestTerminal -v
```

Expected: 7 PASSED (4 new + 3 previous).

- [ ] **Step 5: Commit**

```bash
git add tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
git commit -m "feat(macla): StarCraftShaper.compute_reward — positive deltas

Rewards army/worker built (supply_used delta), structure built
(building_count delta), and game_time survival baseline. Negative
deltas (units lost, minerals spent) are not penalized in this layer —
that's the role of the idleness / supply-block penalties in the next
commit."
```

---

## Task 5: Implement `compute_reward` — idleness + supply-block penalties (load-bearing)

**Files:**
- Modify: `tests/test_online_evaluator_starcraft.py` — add `TestPenalties` class
- Modify: `agents/macla/online_evaluator.py` — extend `compute_reward`

- [ ] **Step 1: Write failing tests for penalties**

Add to `tests/test_online_evaluator_starcraft.py`:

```python
class TestPenalties:
    def _metrics(self, **overrides):
        base = {
            "game_time_sec": 100,
            "mineral": 500,
            "supply_used": 20,
            "supply_cap": 23,
            "supply_left": 3,
            "worker_supply": 18,
            "building_count": 2,
            "enemy_unit_count": 0,
        }
        return {**base, **overrides}

    def test_floated_minerals_penalty_fires_when_mineral_grows_supply_flat(self, shaper):
        prev = self._metrics(mineral=500, supply_used=20)
        cur = self._metrics(mineral=800, supply_used=20)  # gained 300 minerals, no units built
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # -0.3 (floated) + 0.05 (survival baseline if time advanced — but it didn't here) = -0.3
        # Actually game_time_sec is same in both, so no survival increment.
        assert r == pytest.approx(-0.3)

    def test_no_floated_penalty_when_supply_grew(self, shaper):
        prev = self._metrics(mineral=500, supply_used=20)
        cur = self._metrics(mineral=550, supply_used=22)  # gained mineral AND built unit
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # 2 * 0.2 = 0.4 (supply) + 0 (no time advance) — NO floated penalty
        assert r == pytest.approx(0.4)

    def test_supply_block_penalty_fires_when_supply_left_zero(self, shaper):
        prev = self._metrics(supply_left=5)
        cur = self._metrics(supply_left=0)
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # -0.5 supply block. supply_used unchanged → no army reward. No time advance.
        assert r == pytest.approx(-0.5)

    def test_supply_block_penalty_fires_when_supply_left_negative(self, shaper):
        cur = self._metrics(supply_left=-15)  # iter-201 scenario
        prev = self._metrics(supply_left=-15)
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        assert r == pytest.approx(-0.5)

    def test_iter_201_failure_state_gets_strongly_negative_reward(self, shaper):
        """Smoke gun: iter-201 floated AND supply-blocked → both penalties stack."""
        prev = self._metrics(mineral=3500, supply_used=23, supply_left=-15, game_time_sec=350)
        cur = self._metrics(mineral=3980, supply_used=23, supply_left=-15, game_time_sec=356)
        r = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # -0.3 (floated) + -0.5 (supply block) + 0.05 (survival) = -0.75
        assert r == pytest.approx(-0.75)
```

- [ ] **Step 2: Run tests — verify RED**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestPenalties -v
```

Expected: 5 FAILED.

- [ ] **Step 3: Extend compute_reward with penalty logic**

In `agents/macla/online_evaluator.py`, extend `StarCraftShaper.compute_reward` (insert the two penalty blocks BEFORE the final `return self._clamp(reward)`):

```python
    def compute_reward(self, prev: dict, cur: dict, success: bool, is_fatal: bool) -> float:
        s = self._shaping

        if is_fatal:
            return s["fatal_penalty"]
        if success:
            return s["victory_bonus"]

        reward = 0.0

        # Survival baseline: tiny constant when game_time advances.
        if cur.get("game_time_sec", 0) > prev.get("game_time_sec", 0):
            reward += s["survival_increment"]

        # Supply_used delta — army / worker built.
        supply_delta = cur.get("supply_used", 0) - prev.get("supply_used", 0)
        if supply_delta > 0:
            reward += s["supply_used_weight"] * supply_delta

        # Building delta — structure built.
        building_delta = cur.get("building_count", 0) - prev.get("building_count", 0)
        if building_delta > 0:
            reward += s["building_built_weight"] * building_delta

        # Floated-minerals penalty: mineral grows but supply_used flat → idle.
        mineral_delta = cur.get("mineral", 0) - prev.get("mineral", 0)
        if mineral_delta > 0 and supply_delta == 0:
            reward += s["floated_minerals_penalty"]

        # Supply-block penalty: cannot train new units.
        if cur.get("supply_left", 1) <= 0:
            reward += s["supply_block_penalty"]

        return self._clamp(reward)
```

- [ ] **Step 4: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py -v
```

Expected: 12 PASSED (5 new + 7 previous).

- [ ] **Step 5: Commit**

```bash
git add tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
git commit -m "feat(macla): StarCraftShaper.compute_reward — idleness + supply-block penalties

Load-bearing fix for the PR3 smoke iter-201 failure mode (3980 minerals
floated + Supply left: -15). Floated penalty fires when mineral grows
but supply_used is flat (the agent collected resources without spending
them). Supply-block penalty fires when Supply left <= 0 (cannot train
new units). Both stack in the iter-201 scenario for a clearly negative
reward (-0.75)."
```

---

## Task 6: Implement first-enemy bonus + `reset_episode`

**Files:**
- Modify: `tests/test_online_evaluator_starcraft.py` — add `TestEpisodeState` class
- Modify: `agents/macla/online_evaluator.py` — add first-enemy logic + verify reset_episode

- [ ] **Step 1: Write failing tests**

Add to `tests/test_online_evaluator_starcraft.py`:

```python
class TestEpisodeState:
    def _metrics(self, **overrides):
        base = {
            "game_time_sec": 100,
            "mineral": 500,
            "supply_used": 20,
            "supply_cap": 23,
            "supply_left": 3,
            "worker_supply": 18,
            "building_count": 2,
            "enemy_unit_count": 0,
        }
        return {**base, **overrides}

    def test_first_enemy_bonus_fires_once(self, shaper):
        # First contact: enemy_unit_count goes from 0 → 1.
        prev = self._metrics(enemy_unit_count=0)
        cur = self._metrics(enemy_unit_count=1, game_time_sec=100)  # no time advance
        r1 = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        # +0.5 first-enemy bonus
        assert r1 == pytest.approx(0.5)

        # Second step: enemy still visible — no bonus.
        prev2 = self._metrics(enemy_unit_count=1)
        cur2 = self._metrics(enemy_unit_count=3, game_time_sec=100)
        r2 = shaper.compute_reward(prev=prev2, cur=cur2, success=False, is_fatal=False)
        assert r2 == pytest.approx(0.0)

    def test_reset_episode_clears_seen_enemy_flag(self, shaper):
        # Fire the one-shot bonus.
        prev = self._metrics(enemy_unit_count=0)
        cur = self._metrics(enemy_unit_count=1, game_time_sec=100)
        r1 = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        assert r1 == pytest.approx(0.5)
        assert shaper._seen_enemy_unit is True

        # Reset → flag cleared.
        shaper.reset_episode()
        assert shaper._seen_enemy_unit is False

        # Now first-enemy bonus fires again in the new episode.
        r2 = shaper.compute_reward(prev=prev, cur=cur, success=False, is_fatal=False)
        assert r2 == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests — verify RED**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py::TestEpisodeState -v
```

Expected: 2 FAILED.

- [ ] **Step 3: Add first-enemy bonus logic**

In `agents/macla/online_evaluator.py`, extend `StarCraftShaper.compute_reward` (insert the bonus block AFTER building delta but BEFORE the penalty blocks):

```python
        # First-enemy contact: one-shot bonus.
        if not self._seen_enemy_unit and cur.get("enemy_unit_count", 0) > 0:
            reward += s["first_enemy_bonus"]
            self._seen_enemy_unit = True
```

(`reset_episode` already clears `_seen_enemy_unit` — that was scaffolded in Task 1.)

- [ ] **Step 4: Run tests — verify GREEN**

```bash
.venv/bin/pytest tests/test_online_evaluator_starcraft.py -v
```

Expected: 14 PASSED (2 new + 12 previous).

- [ ] **Step 5: Commit**

```bash
git add tests/test_online_evaluator_starcraft.py agents/macla/online_evaluator.py
git commit -m "feat(macla): StarCraftShaper — first-enemy contact bonus + reset_episode

One-shot +0.5 when enemy_unit_count first becomes > 0 (mirrors
PokemonShaper's map-discovery bonus pattern). Reset via reset_episode
so the bonus fires once per episode."
```

---

## Task 7: Replay validation script + decision gate

**Files:**
- Create: `experiments/sc2_replay_shaper.py` — standalone replay validation script

- [ ] **Step 1: Write the replay script**

Create `experiments/sc2_replay_shaper.py`:

```python
"""Replay the existing PR3 smoke through the new StarCraftShaper.

Re-runs game_logs/star_craft/stagnation_pr3_star_craft_smoke_20260527T094639Z/
game_states.jsonl (2500 iterations, already on disk) through the new shaper
without needing SC2. Reports cumulative reward per episode and prints the
decision gate: reward at iter 51 (productive state) MUST be > reward at iter
201 (floated + supply-blocked failure state).

Run:
    .venv/bin/python -m experiments.sc2_replay_shaper
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import typer

from agents.macla.online_evaluator import DEFAULT_SHAPING, StarCraftShaper

app = typer.Typer(add_completion=False)

SMOKE_PATH = Path(
    "game_logs/star_craft/stagnation_pr3_star_craft_smoke_20260527T094639Z/"
    "game_states.jsonl"
)


@app.command()
def replay(
    path: Path = typer.Option(SMOKE_PATH, "--path", "-p", help="game_states.jsonl"),
    spot_iters: list[int] = typer.Option(
        [1, 51, 201, 500, 1000, 2000], "--spot", help="iterations to print"
    ),
) -> None:
    if not path.exists():
        typer.echo(f"ERROR: {path} not found", err=True)
        raise typer.Exit(1)

    shaper = StarCraftShaper(DEFAULT_SHAPING["star_craft"])
    prev_metrics: dict = {}
    spot_rewards: dict[int, float] = {}
    per_episode_reward: dict[int, float] = defaultdict(float)
    n_idleness = 0
    n_supply_block = 0
    n_building_built = 0
    cur_episode = 1

    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            it = obj.get("iteration", 0)
            obs_str = obj.get("obs", {}).get("obs_str", "")
            cur = shaper.extract_metrics(obs_str)

            # MACLA itself doesn't pass success/is_fatal here (we never won and
            # game-over is unlogged at the iter level); use False/False.
            reward = shaper.compute_reward(prev_metrics, cur, success=False, is_fatal=False)

            # Track penalty fires by reproducing the conditions.
            mineral_delta = cur.get("mineral", 0) - prev_metrics.get("mineral", 0)
            supply_delta = cur.get("supply_used", 0) - prev_metrics.get("supply_used", 0)
            building_delta = cur.get("building_count", 0) - prev_metrics.get(
                "building_count", 0
            )
            if mineral_delta > 0 and supply_delta == 0:
                n_idleness += 1
            if cur.get("supply_left", 1) <= 0:
                n_supply_block += 1
            if building_delta > 0:
                n_building_built += building_delta

            per_episode_reward[cur_episode] += reward

            if it in spot_iters:
                spot_rewards[it] = reward
                typer.echo(
                    f"  iter {it:>4}  reward={reward:+.3f}  "
                    f"mineral={cur.get('mineral'):>4}  "
                    f"supply_used={cur.get('supply_used'):>3}  "
                    f"supply_left={cur.get('supply_left'):>3}  "
                    f"buildings={cur.get('building_count')}"
                )

            prev_metrics = cur

    typer.echo("")
    typer.echo("=== Per-episode cumulative reward ===")
    for ep, r in sorted(per_episode_reward.items()):
        typer.echo(f"  episode {ep}: {r:+.3f}")

    typer.echo("")
    typer.echo("=== Aggregate penalty fires ===")
    typer.echo(f"  idleness:   {n_idleness}")
    typer.echo(f"  supply_blk: {n_supply_block}")
    typer.echo(f"  buildings:  {n_building_built}")

    typer.echo("")
    typer.echo("=== Decision gate ===")
    r_51 = spot_rewards.get(51, float("nan"))
    r_201 = spot_rewards.get(201, float("nan"))
    typer.echo(f"  reward[iter 51]  (productive) = {r_51:+.3f}")
    typer.echo(f"  reward[iter 201] (floated)    = {r_201:+.3f}")
    if r_201 < r_51:
        typer.echo("  ✓ PASS — failure state is lower-rewarded than productive state")
    else:
        typer.echo("  ✗ FAIL — magnitudes need tuning before running a fresh smoke")
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Run the script**

```bash
cd /workspace/orak-futile-detector
.venv/bin/python -m experiments.sc2_replay_shaper
```

Expected output structure:

```
  iter    1  reward=...  ...
  iter   51  reward=+...  ...   ← productive state, should be positive
  iter  201  reward=-...  ...   ← floated state, should be negative
  ...

=== Per-episode cumulative reward ===
  episode 1: ...
  ...

=== Aggregate penalty fires ===
  idleness:   <large number>
  supply_blk: <large number>
  buildings:  <small number>

=== Decision gate ===
  reward[iter 51]  (productive) = +0.x
  reward[iter 201] (floated)    = -0.x
  ✓ PASS — failure state is lower-rewarded than productive state
```

If the decision gate FAILS, do NOT proceed — instead investigate which magnitudes need adjustment in `DEFAULT_SHAPING["star_craft"]`, update Task 1, and re-run from Task 1.

- [ ] **Step 3: Lint**

```bash
.venv/bin/ruff check experiments/sc2_replay_shaper.py
.venv/bin/ruff format experiments/sc2_replay_shaper.py
```

- [ ] **Step 4: Commit**

```bash
git add experiments/sc2_replay_shaper.py
git commit -m "feat(macla): replay validation for StarCraftShaper

Re-runs the existing PR3 smoke game_states.jsonl through the new shaper
(no SC2 server needed) and prints per-episode cumulative reward, penalty
fire counts, and a PASS/FAIL decision gate (reward[iter 201] < reward[iter
51]). Cheap, deterministic, re-runnable across weight tweaks."
```

---

## Task 8: Final lint + push branch + open PR

**Files:**
- None (git operations only)

- [ ] **Step 1: Run full lint + test suite**

```bash
cd /workspace/orak-futile-detector
.venv/bin/ruff check agents/macla/online_evaluator.py tests/test_online_evaluator_starcraft.py experiments/sc2_replay_shaper.py
.venv/bin/ruff format --check agents/macla/online_evaluator.py tests/test_online_evaluator_starcraft.py experiments/sc2_replay_shaper.py
.venv/bin/pytest tests/test_online_evaluator_starcraft.py -v
```

Expected: 14 tests PASSED.

- [ ] **Step 2: Push branch**

```bash
git push -u origin feat/sc2-reward-shaping
```

- [ ] **Step 3: Open PR**

```bash
gh pr create --title "feat(macla): SC2 intermediate reward shaping (StarCraftShaper)" --body "$(cat <<'EOF'
## Summary

Adds [`StarCraftShaper`](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py#L304) to unblock MACLA procedural-memory refinement on SC2. PR3 smoke ([`stagnation_pr3_star_craft_smoke_20260527T094639Z`](../tree/feat/sc2-reward-shaping/game_logs/star_craft/stagnation_pr3_star_craft_smoke_20260527T094639Z)) finished with `successful_executions=0` across 2500 steps — the refinement loop was running on noise because SC2 was falling through to `GenericShaper` (zero reward unless `success=True`, never the case). This PR matches the per-game pattern of [`MarioShaper`](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py#L131-L163) / [`TwentyFortyEightShaper`](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py#L169-L247) / [`PokemonShaper`](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py#L253-L301) with race-agnostic regex over the obs_str's `Resources/Buildings/Units/Enemy` sections and a load-bearing **idleness + supply-block penalty** for the iter-201 failure mode (3980 minerals floated + supply-blocked).

Design spec: [`docs/specs/2026-05-27-sc2-reward-shaping-design.md`](../tree/feat/sc2-reward-shaping/docs/specs/2026-05-27-sc2-reward-shaping-design.md).

## Test plan

- [x] 14 unit tests in [`tests/test_online_evaluator_starcraft.py`](../tree/feat/sc2-reward-shaping/tests/test_online_evaluator_starcraft.py) covering registry/extract_metrics/terminal/positive deltas/penalties/episode state — all sub-second
- [x] Replay validation via [`experiments/sc2_replay_shaper.py`](../tree/feat/sc2-reward-shaping/experiments/sc2_replay_shaper.py) — re-runs existing 2500-iter smoke through the new shaper without SC2 needed, asserts `reward[iter 201] < reward[iter 51]`
- [ ] Fresh n=1 SC2 smoke confirming `successful_executions > 0` (separate follow-up after merge — needs 2h SC2 run)

## Commits

| SHA | Subject | Files |
|---|---|---|
| _(filled by `git log` after push)_ | docs(macla): SC2 intermediate reward shaping spec + plan | `docs/specs/...` · `docs/plans/...` |
| | feat(macla): scaffold StarCraftShaper + DEFAULT_SHAPING entry | [`online_evaluator.py`](../tree/feat/sc2-reward-shaping/agents/macla/online_evaluator.py) · [`test_online_evaluator_starcraft.py`](../tree/feat/sc2-reward-shaping/tests/test_online_evaluator_starcraft.py) |
| | feat(macla): StarCraftShaper.extract_metrics — regex over obs_str | same |
| | feat(macla): StarCraftShaper.compute_reward — terminal cases | same |
| | feat(macla): StarCraftShaper.compute_reward — positive deltas | same |
| | feat(macla): StarCraftShaper.compute_reward — idleness + supply-block penalties | same |
| | feat(macla): StarCraftShaper — first-enemy contact bonus + reset_episode | same |
| | feat(macla): replay validation for StarCraftShaper | [`experiments/sc2_replay_shaper.py`](../tree/feat/sc2-reward-shaping/experiments/sc2_replay_shaper.py) |

## Out-of-scope follow-ups

- Episode-end retrospective credit assignment (framework-level, separate PR)
- Tuning weights via Hydra sweep over `reward_shaping:` overrides
- Fresh n=1 SC2 smoke to confirm `successful_executions > 0`
EOF
)"
```

- [ ] **Step 4: Render-check the PR body**

```bash
PR=$(gh pr list --head feat/sc2-reward-shaping --json number --jq '.[0].number')
gh pr view $PR --json body --jq '.body' | head -60
```

Visually confirm: no escaped backticks (`\\\``), no escaped quotes (`\"`), all links rendered as markdown.

---

## Out-of-scope (post-merge follow-ups)

1. **Fresh n=1 SC2 smoke** — same config as PR3 (gemma_26b, Flat64, Protoss vs Zerg D4, max_steps=2500). Decision gate: `successful_executions > 0` in MACLA Stats. Update PR3 (#110) and this PR with the comparison numbers.
2. **Magnitude tuning** — if the replay validation in Task 7 shows weird ratios (e.g. survival_increment dominates), tune via the `reward_shaping:` override in `configs/star_craft/agent/<variant>.yaml` rather than re-pushing.
3. **Episode-end retrospective credit assignment** — lever (2) from the brainstorm, framework-level PR in `macla_lib.py`.

---

## Self-review summary

**Spec coverage**: ✓ every section in the design spec maps to a task:
- Section 1 (placement) → Task 1
- Section 2 (extract_metrics) → Task 2
- Section 3 (compute_reward formula) → Tasks 3-6 (terminal / positive deltas / penalties / first-enemy)
- Section 4 (episode state) → Task 6
- Section 5 (error handling — missing fields → 0) → covered by `or 0` in extract_metrics (Task 2) and `prev = {}` empty-dict handling
- Testing strategy → Tasks 1-6 (unit tests) + Task 7 (replay validation)
- File touch list → Tasks 1-7 exactly

**Placeholder scan**: no TBDs, no "add error handling," no "similar to Task N."

**Type/name consistency**: `_seen_enemy_unit` defined in Task 1, used in Task 6 — matches. `DEFAULT_SHAPING["star_craft"]` keys defined in Task 1, referenced in Tasks 3-6 — matches. `extract_metrics` return dict keys defined in Task 2, used in Tasks 3-6 via `cur.get(...)` — matches.
