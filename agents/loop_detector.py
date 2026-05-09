"""Game-agnostic loop detector for agentic eval harnesses.

Pokemon Stage A retry on 26B model (game_logs/pokemon_red/20260506_221856/)
showed the failure mode: the agent hit milestone 2 (entered OaksLab),
then bounced OaksLab ↔ PalletTown 14 times over 280 steps, spamming
``interact_with_object("OBJ_1_1")`` 38 times against pokeballs that
required Oak's permission gate. The reward shaper noticed (-0.400 per
stagnant step) but never told the LLM. The "Avoid Loops" rule in the
prompt is map-shape specific (only fires on staircase warps where prev
map == cur map), so an A→B oscillation walked right past it.

This module implements a generic detector that runs on any
``(map_or_phase, x, y, score)`` state primitive — the universal shape
across pokemon, mario, 2048, starcraft. It surfaces three orthogonal
loop signals:

1. **State recurrence** — current state hash visited ≥N times in the
   sliding window with no score gain.
2. **Action-class repetition** — the same tool/action family (e.g.
   ``interact_with_object``) issued ≥M times in a row with no score gain.
3. **Map oscillation** — A→B→A→B pattern over recent transitions.

The renderer emits a ``[Stuck Detector]`` block ready to inject into the
obs prompt. Each game's obs renderer wires it in separately (PR 2);
this module is pure logic + tests, no game-specific code.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass, field

# Default thresholds. Picked from a 280-step pokemon trajectory where the
# agent visited (OaksLab, 4, 1) seven times in 50 steps without progress —
# threshold of 3 catches that comfortably without false-firing on normal
# pathfinding through a corridor.
DEFAULT_WINDOW = 50
DEFAULT_STATE_REPEAT_THRESHOLD = 3
DEFAULT_ACTION_REPEAT_THRESHOLD = 5
DEFAULT_OSCILLATION_THRESHOLD = 3  # min A→B→A→B switches to flag
DEFAULT_MIN_STEPS_BEFORE_FIRING = 10  # don't nag during opening exploration
DEFAULT_SCORE_GRACE_STEPS = 0  # opt-in suppression: don't fire for N steps after a score gain


@dataclass
class LoopSignal:
    """Snapshot of the detector's findings after observing one step.

    Field semantics:
    - ``state_repeats`` is a count over the *window*, not all-time.
    - ``action_repeat_streak`` resets to 1 every time the action class
      changes — it counts the current run, not a window total.
    - ``oscillation_pair`` is set only when the *most recent* transitions
      form an alternating ABAB pattern; intermittent ABCAB doesn't count.
    - ``steps_since_score_gain`` is monotonic between score increases.
    """

    state_repeats: int
    state_window: int
    action_repeat_streak: int
    last_action_class: str | None
    oscillation_pair: tuple[Hashable, Hashable] | None
    oscillation_count: int
    steps_since_score_gain: int

    def is_loop(
        self,
        *,
        state_repeat_threshold: int = DEFAULT_STATE_REPEAT_THRESHOLD,
        action_repeat_threshold: int = DEFAULT_ACTION_REPEAT_THRESHOLD,
        oscillation_threshold: int = DEFAULT_OSCILLATION_THRESHOLD,
    ) -> bool:
        """Decide whether at least one signal trips its threshold.

        Action repetition is gated on ``state_repeats >= 2`` so that a
        clean corridor walk (same ``move_to`` class but every step
        advances to a new tile) does NOT count as a loop. The corridor
        case is exactly the one that misled an earlier draft of this
        detector: action class alone is too noisy.
        """
        return any(
            (
                self.state_repeats >= state_repeat_threshold,
                self.action_repeat_streak >= action_repeat_threshold and self.state_repeats >= 2,
                self.oscillation_count >= oscillation_threshold,
            )
        )


@dataclass
class LoopDetector:
    """Stateful per-game loop detector. One instance per running episode.

    Use:
        detector = LoopDetector()
        signal = detector.observe(state=("OaksLab", 4, 1), score=2,
                                  action_class="interact_with_object")
        block = detector.render(signal)
        if block: prompt += "\n\n" + block
    """

    window_size: int = DEFAULT_WINDOW
    state_repeat_threshold: int = DEFAULT_STATE_REPEAT_THRESHOLD
    action_repeat_threshold: int = DEFAULT_ACTION_REPEAT_THRESHOLD
    oscillation_threshold: int = DEFAULT_OSCILLATION_THRESHOLD
    min_steps_before_firing: int = DEFAULT_MIN_STEPS_BEFORE_FIRING
    score_grace_steps: int = DEFAULT_SCORE_GRACE_STEPS

    _states: deque = field(init=False)
    _maps: deque = field(init=False)  # for oscillation detection
    _action_class_streak: int = field(init=False, default=0)
    _last_action_class: str | None = field(init=False, default=None)
    _last_score: float | None = field(init=False, default=None)
    _steps_since_score_gain: int = field(init=False, default=0)
    _step: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._states = deque(maxlen=self.window_size)
        # 16 transitions is enough to detect any reasonable AB oscillation
        # without wasting memory.
        self._maps = deque(maxlen=16)

    def reset(self) -> None:
        """Drop all state — call between episodes."""
        self._states.clear()
        self._maps.clear()
        self._action_class_streak = 0
        self._last_action_class = None
        self._last_score = None
        self._steps_since_score_gain = 0
        self._step = 0

    def observe(
        self,
        *,
        state: Hashable,
        score: float,
        action_class: str | None,
    ) -> LoopSignal:
        """Record one step and return the current loop snapshot.

        ``state`` is a hashable tuple identifying the agent's *external*
        situation — typically ``(map_name, x, y)`` for grid games. It
        excludes ``score`` because we want to count how many times the
        agent has been *physically here* without progress.

        ``action_class`` is the tool/family name (``"interact_with_object"``,
        ``"warp_with_warp_point"``) — *not* the full call with args.
        Pass ``None`` if the agent didn't invoke a tool this step.
        """
        self._step += 1

        # Score-gain reset — any progress invalidates stagnation counters.
        if self._last_score is not None and score > self._last_score:
            self._steps_since_score_gain = 0
            # Don't clear the window — we still want to surface that the
            # agent IS spinning on a tile even if it just made progress
            # somewhere else. The stagnation gate handles silencing.
        else:
            self._steps_since_score_gain += 1
        self._last_score = score

        # Action-class streak.
        if action_class is not None and action_class == self._last_action_class:
            self._action_class_streak += 1
        else:
            self._action_class_streak = 1 if action_class else 0
            self._last_action_class = action_class

        # State window for recurrence count.
        self._states.append(state)
        state_repeats = sum(1 for s in self._states if s == state)

        # Map-transition oscillation. We extract the map identifier from
        # the state tuple (first element) — callers are free to pass a
        # bare string instead, in which case the whole state IS the map.
        map_id = state[0] if isinstance(state, tuple) and state else state
        if not self._maps or self._maps[-1] != map_id:
            self._maps.append(map_id)
        oscillation_pair, oscillation_count = self._detect_oscillation()

        return LoopSignal(
            state_repeats=state_repeats,
            state_window=len(self._states),
            action_repeat_streak=self._action_class_streak,
            last_action_class=self._last_action_class,
            oscillation_pair=oscillation_pair,
            oscillation_count=oscillation_count,
            steps_since_score_gain=self._steps_since_score_gain,
        )

    def _detect_oscillation(self) -> tuple[tuple[Hashable, Hashable] | None, int]:
        """Find the longest A→B→A→B run anchored at the current map.

        Walks back from the most-recent transition checking strict
        alternation between exactly two values. Returns
        ``((A, B), n_switches)`` where ``A`` is the older and ``B`` the
        more recent of the pair, and ``n_switches`` is how many times
        the value flipped over recent history (so [A,B,A,B,A,B] gives
        5 switches). Returns ``(None, 0)`` if there are fewer than 3
        distinct transitions to inspect.
        """
        if len(self._maps) < 3:
            return None, 0
        last = self._maps[-1]
        prev = self._maps[-2]
        if last == prev:
            # observe() collapses runs so this is defensive.
            return None, 0

        n = 1  # the prev→last transition itself
        # Walk further back, expecting strict alternation between
        # ``last`` and ``prev``. Each step we flip what we expect.
        expected = last
        for i in range(len(self._maps) - 3, -1, -1):
            if self._maps[i] == expected:
                n += 1
                expected = prev if expected == last else last
            else:
                break
        if n >= 2:  # at least the immediate prev→last switch
            return (prev, last), n
        return None, 0

    def render(self, signal: LoopSignal | None = None) -> str | None:
        """Return a ``[Stuck Detector]`` text block, or ``None`` if quiet.

        Stays silent during ``min_steps_before_firing`` to avoid noisy
        early-game warnings, and silent when no individual signal trips
        its threshold. ``signal`` is optional — uses the most recent if
        omitted.
        """
        if signal is None:
            # Re-derive from current state without recording a new step.
            signal = LoopSignal(
                state_repeats=0,
                state_window=len(self._states),
                action_repeat_streak=self._action_class_streak,
                last_action_class=self._last_action_class,
                oscillation_pair=None,
                oscillation_count=0,
                steps_since_score_gain=self._steps_since_score_gain,
            )

        if self._step < self.min_steps_before_firing:
            return None
        if signal.steps_since_score_gain < self.score_grace_steps:
            return None
        if not signal.is_loop(
            state_repeat_threshold=self.state_repeat_threshold,
            action_repeat_threshold=self.action_repeat_threshold,
            oscillation_threshold=self.oscillation_threshold,
        ):
            return None

        lines = ["[Stuck Detector]"]
        lines.append(f"- No score gain in last {signal.steps_since_score_gain} steps.")
        if signal.state_repeats >= self.state_repeat_threshold:
            lines.append(
                f"- Visited current position {signal.state_repeats} times in "
                f"last {signal.state_window} steps without progress."
            )
        if signal.action_repeat_streak >= self.action_repeat_threshold:
            lines.append(
                f"- Same action class (`{signal.last_action_class}`) repeated "
                f"{signal.action_repeat_streak} times in a row."
            )
        if (
            signal.oscillation_count >= self.oscillation_threshold
            and signal.oscillation_pair is not None
        ):
            a, b = signal.oscillation_pair
            lines.append(
                f"- Oscillating between `{a}` ↔ `{b}` "
                f"({signal.oscillation_count} switches in recent history)."
            )
        lines.append(
            "- Hint: try a *fundamentally different* action class — "
            "talk to a different NPC, read a sign, examine an unvisited "
            "object, or move to an unexplored region of the map."
        )
        return "\n".join(lines)
