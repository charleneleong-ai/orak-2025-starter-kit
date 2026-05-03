"""Waypoint-based goal provider — programmatic milestone selection for
long-horizon games.

Composes with :class:`LLMSubtaskPlanner`: the goal provider emits the
*current milestone* (a concrete, state-aware sub-goal) as the planner's
``goal=`` argument, and the planner's ``system_prompt`` (per-adapter via
``SUBTASK_PLANNER_SYSTEM`` if defined) expands that milestone into a
step-level subtask. The two layers are orthogonal:

* The goal provider is **deterministic code** — it inspects the
  observation and picks the first unmet milestone from an ordered list.
* The planner is **LLM-driven** — given the milestone as ``goal=``, it
  produces a near-term subtask the agent should focus on next.

This is the structural alternative to baking a static waypoint chain into
the planner's system prompt: each game contributes only data (a
``WAYPOINTS`` list literal), and the selection state machine is reused.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Milestone:
    """One step in a long-horizon game's progression curriculum.

    ``precondition(observation) -> bool`` returns True when the milestone
    is *still the current goal* (i.e. unmet). The first milestone whose
    precondition returns True is selected. Returning False means the
    milestone has been satisfied and selection moves to the next.
    """

    name: str
    goal: str
    precondition: Callable[[str], bool]


class GoalProvider(ABC):
    """Abstract interface for state-aware goal selection."""

    @abstractmethod
    def next_goal(self, observation: str) -> str:
        """Return the current goal string for the planner."""

    def stats(self) -> dict[str, Any]:
        return {}


class WaypointGoalProvider(GoalProvider):
    """Picks the first unmet milestone from an ordered list, falling
    through to ``fallback`` once every milestone is satisfied."""

    def __init__(self, milestones: list[Milestone], fallback: str) -> None:
        if not milestones:
            raise ValueError("WaypointGoalProvider requires at least one Milestone")
        self._milestones = list(milestones)
        self._fallback = fallback
        self._last_milestone: Optional[str] = None
        self._call_count: int = 0

    def next_goal(self, observation: str) -> str:
        self._call_count += 1
        for m in self._milestones:
            if m.precondition(observation or ""):
                self._last_milestone = m.name
                return m.goal
        self._last_milestone = "(all-milestones-met)"
        return self._fallback

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._call_count,
            "last_milestone": self._last_milestone or "(none-selected-yet)",
            "milestone_count": len(self._milestones),
        }
