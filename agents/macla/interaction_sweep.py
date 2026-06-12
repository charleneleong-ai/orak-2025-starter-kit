"""Game-agnostic interaction-sweep controller.

When the milestone-stall detector trips (see EnhancedHierarchicalMemorySystem),
the agent is "story-stalled" — in an explored map with no milestone progress, so
the gate is an *interaction* it hasn't performed, not a *place* it hasn't been.
This controller drives a graduated response over the per-game interactable list:

    stall_steps >= hint_after      → inject a hint, the LLM picks + acts
    stall_steps >= override_after  → take the wheel: walk onto/into the nearest
                                     untried interactable deterministically

Both phases are gated on ``looping`` (a tile over the loop threshold) and on there
being untried interactables left. The per-game half is just the parser + action
emitter on the adapter (interaction_targets / interaction_action); everything here
is game-independent.
"""

from dataclasses import dataclass

Interactable = tuple[str, int, int]  # (label, x, y)


@dataclass(frozen=True)
class SweepDecision:
    mode: str  # "none" | "hint" | "override"
    hint: str | None = None
    target: Interactable | None = None


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def render_sweep_hint(targets: list[Interactable], stall_steps: int) -> str:
    """Render the Phase-1 hint block. Warp→ labels render as 'enter', NPCs as 'talk to'."""
    lines = [
        f"### Interaction sweep (story-stalled {stall_steps} steps — no milestone progress)",
        "You are stuck in an explored area. Work these untried interactables:",
    ]
    for label, x, y in targets:
        verb = "enter" if label.startswith("Warp") else "talk to"
        lines.append(f"  → {verb} {label} at ({x}, {y})")
    return "\n".join(lines)


def decide_interaction_sweep(
    *,
    stall_steps: int,
    looping: bool,
    targets: list[Interactable],
    tried: set[tuple[int, int]],
    player_pos: tuple[int, int],
    hint_after: int = 30,
    override_after: int = 60,
) -> SweepDecision:
    if not looping or stall_steps < hint_after:
        return SweepDecision("none")
    untried = [(label, x, y) for (label, x, y) in targets if (x, y) not in tried]
    if not untried:
        return SweepDecision("none")
    if stall_steps >= override_after:
        nearest = min(untried, key=lambda t: _manhattan((t[1], t[2]), player_pos))
        return SweepDecision("override", target=nearest)
    return SweepDecision("hint", hint=render_sweep_hint(untried, stall_steps))
