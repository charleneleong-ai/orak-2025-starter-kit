"""Stage R v4 (6): extend pokemon subgoal stack to the full M5→M6→M7 ladder.

v3 introspect showed the ceiling at score 4.0/7 (M4 cleared, M5 blocked).
The 7-point scoring ladder in pokemon_red_env.py:276-304 is:

    M5: enter Viridian City  (Viridian in map_name)
    M6: pick up Oak's Parcel (OAK's PARCEL in inventory; Mart clerk in Viridian)
    M7: deliver Oak's Parcel (parcel leaves inventory; back to Oak in Pallet)

Stage R v4(0) made the 221-map graph + 404 exit-tile coords actually live
in the planner prompt, so the spatial knowledge to navigate
Pallet→Viridian→Mart→Pallet is now there. What's missing is *stack
entries to chase* past Viridian — without M6/M7 in the stack, the
escape-valve fires once the agent reaches Viridian and there's nothing
further to optimise toward.

The fix: extend agents/pokemon_red/game_adapter.py:initial_subgoal_stack
with score-based milestone subgoals for M5/M6/M7. The completion
predicate is the existing _completes_when_score_at_least (matches the
env's scoring trigger exactly), and each milestone gets a descriptive
name + suggested_tools so the LLM understands what to do.

Note: Brock / ViridianGym / Route22 are *post-M7* in this env — the
7-point scoring caps at parcel delivery. So the stack stops at M7.
"""

from __future__ import annotations

import pickle

import pytest

from agents.macla.macla_lib import Subgoal
from agents.pokemon_red.game_adapter import initial_subgoal_stack


def test_initial_subgoal_stack_has_full_m7_ladder():
    """Stack should be 4 entries: Route1 (top) → Viridian (M5) → Parcel pickup (M6)
    → Parcel delivery (M7, bottom). Bottom..top ordering per the docstring."""
    stack = initial_subgoal_stack()
    names = [sg.name for sg in stack]
    # Bottom = long-horizon, top = next-to-pursue. Route1 is the immediate
    # step from Pallet, then Viridian (M5), then the parcel quest.
    assert names == [
        "DeliverOaksParcel",  # bottom: M7 — final delivery
        "GetOaksParcel",  # M6 — pick up parcel at Viridian Mart
        "EnterViridian",  # M5 — enter Viridian City
        "NavigateToMap(Route1)",  # top: immediate next from Pallet
    ], f"unexpected stack shape: {names}"


def test_m5_subgoal_completes_when_score_reaches_5():
    stack = initial_subgoal_stack()
    m5 = next(sg for sg in stack if sg.name == "EnterViridian")
    assert m5.completion({"score": 5}) is True
    assert m5.completion({"score": 4}) is False
    assert m5.completion({"score": 6}) is True  # at-least semantics


def test_m6_subgoal_completes_when_score_reaches_6():
    stack = initial_subgoal_stack()
    m6 = next(sg for sg in stack if sg.name == "GetOaksParcel")
    assert m6.completion({"score": 6}) is True
    assert m6.completion({"score": 5}) is False
    assert m6.completion({"score": 7}) is True


def test_m7_subgoal_completes_when_score_reaches_7():
    stack = initial_subgoal_stack()
    m7 = next(sg for sg in stack if sg.name == "DeliverOaksParcel")
    assert m7.completion({"score": 7}) is True
    assert m7.completion({"score": 6}) is False


def test_subgoal_stack_entries_are_picklable():
    """Checkpoint roundtrip must preserve the stack (and completion
    predicates) — partial() functions are picklable, lambdas are not.
    This guards against accidental lambda regressions."""
    stack = initial_subgoal_stack()
    roundtripped = pickle.loads(pickle.dumps(stack))
    assert [sg.name for sg in roundtripped] == [sg.name for sg in stack]
    # Predicates survive the roundtrip and still fire correctly.
    m5 = next(sg for sg in roundtripped if sg.name == "EnterViridian")
    assert m5.completion({"score": 5}) is True


def test_each_milestone_has_descriptive_metadata():
    """Score-based completion is opaque to the LLM — name, description,
    and suggested_tools must be descriptive enough to plan toward."""
    stack = initial_subgoal_stack()
    by_name = {sg.name: sg for sg in stack}

    for name in ("EnterViridian", "GetOaksParcel", "DeliverOaksParcel"):
        sg = by_name[name]
        assert isinstance(sg, Subgoal)
        assert sg.description, f"{name} missing description"
        assert sg.suggested_tools, f"{name} missing suggested_tools"

    # Parcel quest specifically needs the dialog tools — pure move_to isn't
    # enough to talk to the Mart clerk or Oak.
    for name in ("GetOaksParcel", "DeliverOaksParcel"):
        tools = by_name[name].suggested_tools
        assert "interact_with_object" in tools or "continue_dialog" in tools, (
            f"{name} needs dialog tools, got {tools}"
        )


@pytest.mark.parametrize(
    "missing_score",
    [None, "not a number", {}],
)
def test_completion_is_robust_to_garbage_score(missing_score):
    """Completion predicates must not crash on observations without a
    parseable score — happens at episode init before the first eval tick."""
    stack = initial_subgoal_stack()
    m5 = next(sg for sg in stack if sg.name == "EnterViridian")
    assert m5.completion({"score": missing_score}) is False
