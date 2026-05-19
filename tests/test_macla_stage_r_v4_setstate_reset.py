"""Stage R v4 (4): EnhancedHierarchicalMemorySystem.__setstate__ must
reset per-episode state that shouldn't survive a checkpoint load.

Symptom from v3 sweep: iter 2 began with ``_subgoal_stagnation_steps=440``
inherited from iter 1's tail, so the escape valve fired from step 1
and the planner never saw the active_subgoal block. Same shape will
apply to v4's anti-perseveration position counter — both are
*per-episode* signals, not cumulative.

We only test the stagnation field here; the position-visit counter
gets its own assertion when task v4(1) lands.
"""

from __future__ import annotations

import pickle

from agents.macla.macla_lib import EnhancedHierarchicalMemorySystem, Subgoal


def _never(_obs: dict) -> bool:
    """Module-level completion predicate (picklable, unlike a lambda)."""
    return False


def _sub(name: str) -> Subgoal:
    return Subgoal(name=name, description=name, completion=_never)


def test_stagnation_counter_resets_on_unpickle():
    mem = EnhancedHierarchicalMemorySystem()
    mem.push_subgoal(_sub("NavigateToMap(Route1)"))
    # Run a handful of "steps" so the counter climbs.
    for _ in range(35):
        mem.record_subgoal_step()
    assert mem.subgoal_stagnation_steps == 35, (
        "Sanity-check the counter actually climbed before pickling."
    )

    loaded = pickle.loads(pickle.dumps(mem))

    assert loaded.subgoal_stagnation_steps == 0, (
        "After unpickle (== checkpoint load) the stagnation counter "
        "must be zero — it's a per-episode signal, not cumulative. "
        "v3 sweep bug: iter 2 began at stagnation=440 because the "
        "field was pickled with iter 1's tail value."
    )
    assert loaded._subgoal_stagnation_key is None, (
        "The stagnation key (name of the top of the stack we were "
        "watching) must also reset — paired with the counter reset, "
        "the next record_subgoal_step call seeds it fresh from the "
        "current top of stack."
    )


def test_stagnation_reset_does_not_clobber_subgoal_stack():
    """The stack itself IS cumulative state — it must survive pickle.
    Only the *stagnation tracking* resets, not the goal stack."""
    mem = EnhancedHierarchicalMemorySystem()
    mem.push_subgoal(_sub("ViridianCity"))
    mem.push_subgoal(_sub("Route1"))
    for _ in range(10):
        mem.record_subgoal_step()

    loaded = pickle.loads(pickle.dumps(mem))

    assert loaded.subgoal_depth() == 2
    assert loaded.peek_subgoal().name == "Route1"
    # And the freshly-loaded counter starts climbing again on the next step.
    loaded.record_subgoal_step()
    assert loaded.subgoal_stagnation_steps == 1


def test_stagnation_reset_safe_when_no_state_present():
    """A pre-v4 checkpoint won't have the stagnation fields in its
    pickled state dict. __setstate__ must tolerate that — restore
    whatever was saved, then force the per-episode fields to zero."""
    mem = EnhancedHierarchicalMemorySystem()
    state = mem.__dict__.copy()
    # Simulate an old checkpoint missing the v3 fields.
    state.pop("_subgoal_stagnation_key", None)
    state.pop("_subgoal_stagnation_steps", None)

    fresh = EnhancedHierarchicalMemorySystem.__new__(EnhancedHierarchicalMemorySystem)
    fresh.__setstate__(state)

    assert fresh._subgoal_stagnation_key is None
    assert fresh._subgoal_stagnation_steps == 0
