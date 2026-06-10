# Interaction sweep — a generalised story-stall escape (2026-06-09)

## Problem

`graph_hint` + the Stage R loop-counter solved **spatial** stalls: detect "I keep
revisiting tiles" → surface the unvisited map + exact exit tile. The failure and
the fix are both spatial — the gate is a *place you haven't been*.

The pokemon M6 wall (OAK's PARCEL, blocking M5→M7) is a different species. The
agent is already in the right map (Viridian); there's no unvisited neighbour to
point at. The gate is an **interaction it hasn't performed** — enter the Mart,
talk to the clerk. Every existing signal (unvisited neighbours, position
revisits) reads "all good", so nothing fires. The n=3 truncfix reruns confirmed
it: all three seeds reach M5 in ~150 steps, then plateau for ~850–1000 steps.

## The reusable primitive: a milestone-stall detector

The generalisable unit is **not** pokemon-parcel knowledge — it's a second stall
detector that watches *milestone progress* instead of *position*:

> milestone score flat for N steps **AND** a tile over the loop threshold (the
> agent is pacing an explored map, not traversing) → the gate here is an
> interaction, not a location.

The loop clause is what stops false-trips during a legitimate long traversal
(new tiles every step, milestone flat but progressing). The detector is 100%
game-agnostic; only the *response* needs per-game data.

## Response: graduated hint → override (approach B)

The env already hands us interactables for free in the obs grid — every NPC is a
`SPRITE_*` tile, every building a `Warp→<dest>` tile, both with coordinates. No
script mining needed. The sweep's target set = visible `SPRITE_*` (talk) ∪
unentered `Warp→*` (enter).

- **Phase 1 (stall ≥ `hint_after`):** inject a hint listing untried interactables;
  the LLM picks + acts. Mirrors `graph_hint` exactly.
- **Phase 2 (stall ≥ `override_after`):** take the wheel — fire the atomic
  high-level tool for the nearest untried interactable
  (`interact_with_object` for an NPC, `warp_with_warp_point` for a warp; each
  navigates *and* interacts in one step), mark it tried, advance. Exhaust the
  in-view set → hand back to normal exploration, which moves the viewport and
  reveals new interactables. The tried set persists per-episode so nothing is
  re-swept.

Defaults `hint_after=30`, `override_after=60` — comfortably past the ~150-step
M5 arrival and well inside the ~850-step plateau.

## The generic / per-game split

| Piece | Home | Game-agnostic? |
|---|---|---|
| milestone-stall detector + tried memory | `macla_lib.py` (`EnhancedHierarchicalMemorySystem`) | yes |
| sweep controller (`decide_interaction_sweep`, `render_sweep_hint`) | `agents/macla/interaction_sweep.py` | yes |
| interactable parser + override action | `agents/pokemon_red/game_adapter.py` (`interaction_targets`, `interaction_action`) | no — per game |

Wiring lives in `unified.py::_base_fallback`, alongside the existing
graph/loop hints. Games that don't export `interaction_targets` never sweep
(`getattr` → None — same safe degradation as `graph_hint`). The stall counter +
tried set follow the loop-counter discipline: survive checkpoint round-trips,
reset once per episode at the `_subgoal_init_done` gate.

## Status

Built + unit-tested (`tests/test_macla_interaction_sweep.py`). Validation
deferred until the n=3 truncfix reruns free the GPU — the live test is a pokemon
rerun checking whether the sweep breaks the M6 parcel wall.
