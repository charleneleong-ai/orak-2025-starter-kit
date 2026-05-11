"""Shared cognitive primitives — composable agent memory.

* ``MemoryProvider`` — abstract base for pluggable memory backends.
* ``VectorMemoryProvider`` — semantic memory via cosine similarity (lifted
  from pokemon_red and made reusable).

Independent of the harness layer. Agents opt in à la carte.
"""

from .memory_provider import MemoryProvider
from .self_reflector import LLMSelfReflector, SelfReflector
from .subtask_planner import LLMSubtaskPlanner, SubtaskPlanner
from .vector_memory import VectorMemoryProvider

__all__ = [
    "MemoryProvider",
    "VectorMemoryProvider",
    "SubtaskPlanner",
    "LLMSubtaskPlanner",
    "SelfReflector",
    "LLMSelfReflector",
]
