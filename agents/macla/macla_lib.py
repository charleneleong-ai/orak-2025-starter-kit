"""
This module implements memory components for the MACLA agent.
Modified from the original MACLA implementation to learn and refine procedures to periodically optimise performance based on execution outcomes.
Ref: https://github.com/S-Forouzandeh/MACLA-LLM-Agents-AAMAS-Conference/blob/main/MACLA.py
"""

import json
import os
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import weave
from loguru import logger

# =========================================================
# STAGE M: SALIENT STATE EXTRACTION
# =========================================================
# Generic key-value extractor for the "did this step actually move the game
# state forward" signal. Generalisable across pokemon (Score/HP/Map/Position/
# In Battle), mario (x/score/lives), 2048 (board), starcraft (minerals/gas/
# supply). When the observation is unstructured (e.g. battle dialog), the
# extractor returns () and downstream state_delta_observed records as None
# — neutral, no penalty.
_SALIENT_KEYS = (
    "score",
    "hp",
    "position",
    "map name",
    "in battle",
    "minerals",
    "gas",
    "supply",
    "lives",
    "board",
)


def _extract_salient_state(observation: str | None) -> tuple[str, ...]:
    """Return a tuple of normalised salient `Key: Value` lines from ``observation``.

    Stable signature of game state for comparing init vs term observations
    to detect whether a step actually moved the game forward. Empty tuple
    when the observation lacks structured key:value lines.
    """
    if not observation:
        return ()
    found: list[str] = []
    for line in observation.split("\n"):
        line_l = line.strip().lower()
        if not line_l or ":" not in line_l:
            continue
        prefix = line_l.split(":", 1)[0].strip()
        if prefix in _SALIENT_KEYS:
            found.append(line.strip())
    return tuple(found)


def _state_delta_observed(
    observation_init: str | None, observation_term: str | None
) -> bool | None:
    """Return True/False/None for the state-delta signal a ContrastiveContext
    should record. None when salient state can't be extracted from either
    side (unstructured observation) — recorded as neutral and excluded from
    the procedure's state_delta_rate."""
    init_s = _extract_salient_state(observation_init)
    term_s = _extract_salient_state(observation_term)
    if not init_s and not term_s:
        return None
    return init_s != term_s


# =========================================================
# OPTIONAL SEMANTIC EMBEDDINGS
# =========================================================
_EMBED_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import util as st_util

    _ST_MODEL_NAME = os.environ.get("MACLA_EMBED_MODEL", "all-MiniLM-L6-v2")
    _EMBEDDER = SentenceTransformer(_ST_MODEL_NAME)
except Exception:
    _EMBED_AVAILABLE = False
    _EMBEDDER = None
    st_util = None
    logger.info("SentenceTransformer not found; semantic similarity will use keyword heuristics.")


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class AtomicMemoryEntry:
    action: str
    observation: str
    reward: float
    context: str
    trajectory_id: str = ""
    step_index: int = 0
    goal: str = "unknown"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Procedure:
    goal: str
    preconditions: list[str]
    steps: list[str]
    postconditions: list[str] = field(default_factory=list)
    reasoning: str = ""
    concepts: set[str] = field(default_factory=set)
    alpha: int = 1
    beta: int = 1
    execution_count: int = 0
    generalisability_score: float = 0.5
    confidence: float = 0.5
    source_trajectory: str = ""
    # Stage L: map-aware procedure key. Default "unknown" matches any map
    # for backwards-compat with pre-Stage-L checkpoints.
    map_name: str = "unknown"
    # Stage M: mean per-token logprob from the LLM call that generated
    # this procedure's action sequence. ``None`` for procedures created
    # before logprobs were plumbed (backwards-compat); these score
    # neutral 0.5 in BayesianProcedureSelector._logprob_confidence.
    mean_logprob: float | None = None

    @property
    def success_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class MetaProcedure:
    goal_meta: str
    preconditions_meta: list[str]
    sub_procedures: list[str]
    composition_policy: dict[str, any]
    alpha: int = 1
    beta: int = 1
    execution_count: int = 0

    @property
    def success_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class ContrastiveContext:
    observation_init: str
    action_sequence: list[str]
    observation_term: str
    cumulative_reward: float
    trajectory_id: str
    success: bool
    context: str = ""
    preconditions_image: Any = None
    postconditions_image: Any = None
    fatal: bool = False
    observation: str = ""
    # Stage M: did this execution move the salient game state forward?
    # True/False when salient extraction succeeded on both init and term;
    # None when the observation lacked structured key:value lines (e.g.
    # battle dialog) — that case bootstraps to neutral 0.5 in the selector.
    state_delta_observed: bool | None = None


@dataclass
class ProceduralMemoryEntry:
    procedure: Procedure
    success_contexts: list[ContrastiveContext] = field(default_factory=list)
    failure_contexts: list[ContrastiveContext] = field(default_factory=list)
    discriminative_patterns: dict[str, list[str]] = field(default_factory=dict)
    contexts: set[str] = field(default_factory=set)
    goals: set[str] = field(default_factory=set)
    performance_score: float = 0.5
    last_refined: float = field(default_factory=time.time)
    # Stage L: iter at which this entry was last selected. prune_stale_procedures
    # retires entries with `last_used_iter < current_iter - max_age`.
    last_used_iter: int = 0


# =========================================================
# HIERARCHICAL MEMORY
# =========================================================
class EnhancedHierarchicalMemorySystem:
    def __init__(self, N_a: int = 1000, N_s: int = 100, N_p: int = 200, N_m: int = 50):
        self.N_a = N_a
        self.N_s = N_s
        self.N_p = N_p
        self.N_m = N_m

        self.atomic_memory: deque = deque(maxlen=N_a)
        self.procedural_memory: dict[str, ProceduralMemoryEntry] = {}
        self.meta_procedural_memory: dict[str, MetaProcedure] = {}

        self.context_index = defaultdict(set)
        self.goal_index = defaultdict(set)

        self.stats = {
            "procedures_added": 0,
            "procedures_refined": 0,
            "meta_procedures_added": 0,
            "procedures_pruned_stale": 0,
        }
        self._meta_counter = 0
        # Stage L: iter counter for the cumulative-memory chain. Bumped each
        # time the agent loads from a previous-iter checkpoint.
        self.current_iter: int = 0
        # Stage M: track which maps the cumulative-memory chain has visited.
        # select_procedure raises the effective theta_conf on unvisited maps
        # so cached procedures rarely fire and the LLM gets free rein to
        # explore. Persists across iters via the existing pickle checkpoint.
        self.visited_maps: set[str] = set()
        # Stage M (third signal): rolling distribution of recent mean
        # per-token logprobs from the agent's LLM calls. Used by the
        # selector's percentile-rank calibration. Bootstraps when len < 10.
        self._recent_logprobs: deque[float] = deque(maxlen=50)
        # Hand-off slot from the calling agent's most-recent LLM call to
        # the macla_lib procedure-creation site. Set by the agent before
        # provide_feedback; consumed (set back to None) on use.
        self._pending_logprob: float | None = None

    def record_map_visit(self, map_name: str | None) -> None:
        """Stage M: record that the agent has been on ``map_name`` in this
        cumulative-memory chain. ``"unknown"`` / empty / None are ignored —
        they represent absence-of-info, not a discovered map."""
        if not map_name or map_name == "unknown":
            return
        self.visited_maps.add(map_name)

    def is_new_map(self, map_name: str | None) -> bool:
        """Stage M: True iff ``map_name`` is a real map name (not unknown /
        empty / None) that has not yet been visited in this chain."""
        if not map_name or map_name == "unknown":
            return False
        return map_name not in self.visited_maps

    def bump_iter(self) -> int:
        """Increment the iter counter — call when a checkpoint is loaded
        (i.e. each new iter under --load-checkpoint --prev-run-id)."""
        self.current_iter += 1
        return self.current_iter

    def prune_stale_procedures(self, max_age: int = 2) -> list[str]:
        """Stage L: retire procedural entries whose ``last_used_iter`` is
        older than ``current_iter - max_age``. Returns the keys removed.

        Default ``max_age=2`` means a procedure that hasn't been selected
        for 2+ full iters is dropped from the cache and its index entries
        cleaned up.
        """
        threshold = self.current_iter - max_age
        removed: list[str] = []
        for key, entry in list(self.procedural_memory.items()):
            if entry.last_used_iter < threshold:
                removed.append(key)
                for context in entry.contexts:
                    self.context_index[context].discard(key)
                for goal in entry.goals:
                    self.goal_index[goal].discard(key)
                del self.procedural_memory[key]
        self.stats["procedures_pruned_stale"] += len(removed)
        return removed

    @weave.op()
    def add_atomic_entry(
        self,
        action: str,
        observation: str,
        reward: float,
        context: str,
        trajectory_id: str = "",
        step_index: int = 0,
        goal: str = "unknown",
    ):
        entry = AtomicMemoryEntry(
            action=action,
            observation=observation,
            reward=reward,
            context=context,
            trajectory_id=trajectory_id,
            step_index=step_index,
            goal=goal,
        )
        self.atomic_memory.append(entry)

    @weave.op()
    def add_procedural_entry(
        self, procedure: Procedure, contexts: set[str], goals: set[str], performance: float
    ) -> str:
        # Stage L: map_name is part of the key so the same step sequence
        # captured in different maps gets distinct cache entries instead
        # of colliding/merging. Procedures without an explicit map_name
        # (older checkpoints) default to "unknown" via the dataclass.
        map_name = getattr(procedure, "map_name", "unknown") or "unknown"
        proc_key = f"proc_{hash((str(procedure.steps), map_name)) % 1000000}"

        if proc_key in self.procedural_memory:
            # Merge with existing entry to avoid duplicates and preserve stats
            existing = self.procedural_memory[proc_key]
            existing.contexts.update(contexts)
            existing.goals.update(goals)
            # We don't overwrite stats as we want to accumulate history
            # potentially update concept/performance if needed
            existing.last_refined = time.time()
            self.stats["procedures_refined"] += 1  # Count as refinement/update
            return proc_key

        if len(self.procedural_memory) >= self.N_p:
            self._prune_procedural_memory()

        entry = ProceduralMemoryEntry(
            procedure=procedure,
            contexts=contexts,
            goals=goals,
            performance_score=performance,
        )
        self.procedural_memory[proc_key] = entry

        for context in contexts:
            self.context_index[context].add(proc_key)

        for goal in goals:
            self.goal_index[goal].add(proc_key)

        self.stats["procedures_added"] += 1
        return proc_key

    @weave.op()
    def add_meta_procedure(self, meta_proc: MetaProcedure) -> str:
        self._meta_counter += 1
        meta_key = f"meta_{self._meta_counter:06d}"

        if len(self.meta_procedural_memory) >= self.N_m:
            self._prune_meta_procedural_memory()

        self.meta_procedural_memory[meta_key] = meta_proc
        self.stats["meta_procedures_added"] += 1
        return meta_key

    @weave.op()
    def record_execution_outcome(
        self, proc_key: str, success: bool, context: ContrastiveContext, is_fatal: bool = False
    ):
        if proc_key not in self.procedural_memory:
            return
        entry = self.procedural_memory[proc_key]

        if success:
            entry.procedure.alpha += 1
            entry.success_contexts.append(context)
        else:
            # Heavy penalty for fatal failures (Game Over)
            penalty = 5 if is_fatal else 1
            entry.procedure.beta += penalty
            context.fatal = is_fatal
            entry.failure_contexts.append(context)

        entry.procedure.execution_count += 1
        if len(entry.success_contexts) > 15:
            entry.success_contexts.pop(0)
        if len(entry.failure_contexts) > 15:
            entry.failure_contexts.pop(0)

    def _prune_procedural_memory(self):
        if not self.procedural_memory:
            return
        utilities = []
        now = time.time()
        for key, entry in self.procedural_memory.items():
            utility = (
                0.5 * entry.procedure.success_rate
                + 0.3 * min(1.0, entry.procedure.execution_count / 10.0)
                + 0.2 * (1.0 - min(1.0, (now - entry.last_refined) / 86400))
            )
            utilities.append((key, utility))
        utilities.sort(key=lambda x: x[1])
        to_remove = utilities[0][0]

        entry = self.procedural_memory[to_remove]
        for context in entry.contexts:
            self.context_index[context].discard(to_remove)
        for goal in entry.goals:
            self.goal_index[goal].discard(to_remove)
        del self.procedural_memory[to_remove]

    def _prune_meta_procedural_memory(self):
        if not self.meta_procedural_memory:
            return
        pairs = [(k, mp.success_rate) for k, mp in self.meta_procedural_memory.items()]
        pairs.sort(key=lambda x: x[1])
        del self.meta_procedural_memory[pairs[0][0]]


# =========================================================
# BAYESIAN SELECTOR
# =========================================================
class BayesianProcedureSelector:
    def __init__(
        self,
        memory_system: EnhancedHierarchicalMemorySystem,
        context_extractor=None,
        spatial_pattern_extractor=None,
    ):
        self.memory_system = memory_system
        self.ontology: dict[str, list[str]] = {}
        self.ontology_embeddings: dict[str, Any] = {}
        self.context_extractor = context_extractor
        self.spatial_pattern_extractor = spatial_pattern_extractor
        # Master switch. When False, select_procedure short-circuits to
        # (None, 0.0) on every call so the MACLA procedure layer is bypassed
        # entirely. EnhancedMACLAAgent flips this from the agent config.
        self.use_procedure_layer = True

    # Stage L: parse the current map name out of the structured observation
    # text. Pokemon observations carry "Map Name: <Name>, (x_max,..." in the
    # [Map Info] block. Falls back to "unknown" when the pattern is absent
    # (e.g. battle screens, dialog screens, non-pokemon games).
    _MAP_NAME_RE = re.compile(r"Map Name:\s*([^\s,\n]+)", re.IGNORECASE)

    def _extract_map_name(self, observation: str) -> str:
        if not observation:
            return "unknown"
        m = self._MAP_NAME_RE.search(observation)
        return m.group(1).strip() if m else "unknown"

    def build_ontology(self, trajectories: list[dict], k_top: int = 100):
        """Build ontology from trajectories to enable semantic retrieval."""
        all_words = []
        for traj in trajectories:
            task = traj.get("task", "").lower()
            actions = " ".join(traj.get("actions", [])).lower()

            # Extract meaningful words
            words = [w for w in re.findall(r"[a-zA-Z]+", task + " " + actions) if len(w) > 3]
            all_words.extend(words)

        word_counts = Counter(all_words)
        top_words = [w for w, _ in word_counts.most_common(k_top)]

        categories = {}
        if _EMBED_AVAILABLE and _EMBEDDER:
            try:
                word_embeddings = _EMBEDDER.encode(top_words, convert_to_tensor=True)
                used = set()

                for i, w in enumerate(top_words):
                    if w in used:
                        continue
                    similar = [w]
                    for j, ow in enumerate(top_words):
                        if j != i and ow not in used:
                            sim = float(
                                st_util.cos_sim(word_embeddings[i], word_embeddings[j])[0][0]
                            )
                            if sim > 0.6:
                                similar.append(ow)
                                used.add(ow)
                    categories[w] = similar
                    used.add(w)

                self.ontology = categories

                # Pre-compute category embeddings for fast retrieval
                for cat, keys in self.ontology.items():
                    text = f"{cat} {' '.join(keys)}"
                    self.ontology_embeddings[cat] = _EMBEDDER.encode(text, convert_to_tensor=True)

                logger.info(f"Built ontology with {len(self.ontology)} categories using embeddings")
            except Exception as e:
                logger.warning(f"Failed to build ontology with embeddings: {e}")
                self._build_ontology_fallback(top_words)
        else:
            self._build_ontology_fallback(top_words)

    def _build_ontology_fallback(self, top_words):
        categories = {}
        for w in top_words:
            key = w[0]
            categories.setdefault(key, []).append(w)
        self.ontology = categories
        logger.info(f"Built fallback ontology with {len(self.ontology)} categories")

    def _extract_context(self, observation: str) -> str:
        # 1. Try injected extractor first (Game Specific)
        if self.context_extractor:
            try:
                ctx = self.context_extractor(observation)
                if ctx and ctx != "general":
                    if isinstance(ctx, dict):
                        return json.dumps(ctx, sort_keys=True)
                    return str(ctx)
            except Exception as e:
                logger.warning(f"Custom context extraction failed: {e}")

        # 2. Ontology-based extraction (Semantic)
        obs_lower = observation.lower()

        # Fast keyword match
        for category, keywords in self.ontology.items():
            if any(k in obs_lower for k in keywords):
                return category

        # Semantic match
        if _EMBED_AVAILABLE and self.ontology_embeddings:
            try:
                obs_emb = _EMBEDDER.encode(obs_lower, convert_to_tensor=True)
                best_category, best_score = None, 0.0

                for cat, emb in self.ontology_embeddings.items():
                    score = float(st_util.cos_sim(obs_emb, emb)[0][0])
                    if score > best_score:
                        best_score, best_category = score, cat

                if best_score >= 0.55 and best_category:
                    return best_category
            except Exception:
                pass

        # 3. Default fallback
        for w in obs_lower.split():
            if len(w) > 4 and w.isalpha():
                return w
        return "general"

    def _retrieve_candidates(self, observation: str, goal: str, k: int = 10) -> list[str]:
        candidates: set[str] = set()

        if goal in self.memory_system.goal_index:
            candidates.update(self.memory_system.goal_index[goal])

        if not candidates:
            gwords = set(goal.lower().split("_"))
            for g, keys in self.memory_system.goal_index.items():
                if set(g.lower().split("_")) & gwords:
                    candidates.update(keys)

        context = self._extract_context(observation)
        for ctx, keys in self.memory_system.context_index.items():
            if context in ctx.lower() or ctx.lower() in context:
                candidates.update(keys)

        if not candidates and self.memory_system.procedural_memory:
            all_procs = list(self.memory_system.procedural_memory.items())
            all_procs.sort(key=lambda x: x[1].procedure.execution_count, reverse=True)
            candidates = {k for k, _ in all_procs[:k]}

        # Stage L: drop candidates whose procedure was captured in a
        # different map. "unknown" (older procedures or non-map contexts)
        # matches any map.
        current_map = self._extract_map_name(observation)
        candidates = {
            pk
            for pk in candidates
            if pk in self.memory_system.procedural_memory
            and getattr(self.memory_system.procedural_memory[pk].procedure, "map_name", "unknown")
            in (current_map, "unknown")
        }

        clist = list(candidates)
        clist.sort(
            key=lambda pk: self.memory_system.procedural_memory[pk].procedure.execution_count
            if pk in self.memory_system.procedural_memory
            else 0,
            reverse=True,
        )
        return clist[:k]

    def _compute_relevance(
        self, entry: ProceduralMemoryEntry, observation: str, goal: str
    ) -> float:
        rel = 0.0
        if goal in entry.goals:
            rel += 0.6
        else:
            for eg in entry.goals:
                if any(w in eg for w in goal.split("_")):
                    rel += 0.3
                    break

        context = self._extract_context(observation)
        if context in entry.contexts:
            rel += 0.4
        else:
            for ec in entry.contexts:
                if context in ec or ec in context:
                    rel += 0.2
                    break

        # Use learned spatial patterns to determine if current situation matches success/failure patterns
        if hasattr(entry, "discriminative_patterns") and entry.discriminative_patterns:
            patterns = entry.discriminative_patterns
            current_spatial = self._extract_spatial_patterns_from_context(context)

            # Reward if required entities are present
            if "required_entities" in patterns and patterns["required_entities"]:
                for req_entity in patterns["required_entities"]:
                    if req_entity in current_spatial:
                        rel += 0.15  # Boost: required entity present

            # Strong reward if preferred spatial configurations match
            if "preferred_spatial" in patterns and patterns["preferred_spatial"]:
                for entity, preferred_configs in patterns["preferred_spatial"].items():
                    if entity in current_spatial:
                        current_configs = current_spatial[entity]
                        # Check if any current config matches preferred
                        if any(pc in current_configs for pc in preferred_configs):
                            rel += 0.25  # Strong signal: spatial pattern matches success

            # Strong penalty if avoided spatial configurations are present
            if "avoided_spatial" in patterns and patterns["avoided_spatial"]:
                for entity, avoided_configs in patterns["avoided_spatial"].items():
                    if entity in current_spatial:
                        current_configs = current_spatial[entity]
                        # Check if any current config matches avoided
                        if any(ac in current_configs for ac in avoided_configs):
                            rel -= 0.35  # Strong penalty: spatial pattern matches failure

        # Heavy penalty if current context matches a fatal failure context
        # Use context keys (e.g. "pit_ahead_near") which are stable across steps
        current_context = self._extract_context(observation) if observation else ""
        for fc in entry.failure_contexts:
            if getattr(fc, "fatal", False):
                fc_context = getattr(fc, "context", "")
                if fc_context and current_context:
                    fc_parts = set(fc_context.lower().split("_"))
                    cur_parts = set(current_context.lower().split("_"))
                    if fc_parts:
                        overlap = len(fc_parts & cur_parts) / len(fc_parts)
                        if overlap > 0.5:
                            rel -= 1.0
                            break

        # Verify Refined Preconditions (Crucial for avoiding over-generalisation)
        # If the procedure has learned specific token constraints (e.g. '128' must be present),
        # validation against the raw observation is required.
        if len(entry.procedure.preconditions) > 1:  # Basic context key is usually index 0
            obs_tokens = set(observation.lower().split())
            matches = 0
            required = 0
            for pre in entry.procedure.preconditions:
                # heuristic: ignore the base context key itself if it matches broadly
                if pre == context or pre in entry.contexts:
                    continue

                required += 1
                if pre in obs_tokens:
                    matches += 1
                elif "_" in pre and (pre in context or context in pre):
                    # Allow context key fuzzy match
                    matches += 1

            if required > 0:
                match_ratio = matches / required
                if match_ratio < 0.65:  # Allow slight mismatch but penalise heavy
                    rel *= 0.3
                else:
                    rel += 0.1  # Boost if specific preconditions match

        return max(0.0, min(1.0, rel))

    def _extract_spatial_patterns_from_context(self, context_str: str) -> dict[str, set[str]]:
        """Helper to extract spatial patterns from a context string for matching"""
        # Use game-specific extractor if provided (for semantic matching)
        if self.spatial_pattern_extractor:
            return self.spatial_pattern_extractor(context_str)

        # Fallback generic logic
        patterns = defaultdict(set)
        if not context_str or context_str in ["clear_path", "general"]:
            return patterns

        parts = context_str.split("_")
        i = 0
        while i < len(parts):
            if parts[i] not in ["ahead", "behind", "near", "mid", "far", "clear", "path"]:
                entity = parts[i]
                spatial_desc = []
                j = i + 1
                while j < len(parts) and parts[j] in ["ahead", "behind", "near", "mid", "far"]:
                    spatial_desc.append(parts[j])
                    j += 1
                if spatial_desc:
                    patterns[entity].add("_".join(spatial_desc))
                else:
                    patterns[entity].add("present")
                i = j
            else:
                i += 1
        return dict(patterns)

    def _compute_failure_risk(
        self, entry: ProceduralMemoryEntry, observation: str, theta_risk: float = 0.85
    ) -> float:
        if not entry.failure_contexts:
            return 0.0
        obs_lower = observation.lower()
        risk_count = 0
        for fctx in entry.failure_contexts:
            fail_obs = fctx.observation_init.lower()
            a = set(obs_lower.split())
            b = set(fail_obs.split())
            overlap = len(a & b) / max(len(a), 1)
            if overlap > theta_risk:
                risk_count += 1
        return risk_count / len(entry.failure_contexts)

    def _compute_information_gain(self, procedure: Procedure) -> float:
        alpha, beta = procedure.alpha, procedure.beta
        n = alpha + beta
        try:
            var = (alpha * beta) / (n * n * (n + 1))
            return float(var * 12.0)
        except Exception:
            return 0.0

    def _state_delta_confidence(self, entry: ProceduralMemoryEntry) -> float:
        """Stage M: fraction of this entry's success_contexts where the
        executing step actually moved the salient game state forward.

        Bootstraps to 0.5 (neutral) when there are no success_contexts or
        when all contexts have ``state_delta_observed=None`` (typical for
        non-pokemon games whose observations lack structured key:value lines).
        """
        if not entry.success_contexts:
            return 0.5
        observed = [getattr(c, "state_delta_observed", None) for c in entry.success_contexts]
        not_none = [o for o in observed if o is not None]
        if not not_none:
            return 0.5
        return sum(1 for o in not_none if o) / len(not_none)

    # Stage M (third signal): minimum sample count before the rolling
    # logprob distribution is considered calibrated. Below this, every
    # entry scores neutral 0.5 — distribution-free bootstrap.
    _LOGPROB_BOOTSTRAP_N = 10

    def _logprob_confidence(self, entry: ProceduralMemoryEntry) -> float:
        """Stage M: percentile rank of this entry's ``procedure.mean_logprob``
        against the memory system's rolling logprob deque.

        Returns 0.5 (neutral) when:
          - the entry's mean_logprob is None (pre-Stage-M procedure)
          - fewer than ``_LOGPROB_BOOTSTRAP_N`` samples in the deque
            (not enough calibration data to rank meaningfully)

        Otherwise returns rank / N ∈ [0, 1]. Cross-model safe (each model's
        procedures calibrate against that model's own distribution).
        """
        mlp = getattr(entry.procedure, "mean_logprob", None)
        if mlp is None:
            return 0.5
        recent = self.memory_system._recent_logprobs
        if len(recent) < self._LOGPROB_BOOTSTRAP_N:
            return 0.5
        rank = sum(1 for lp in recent if lp <= mlp)
        return rank / len(recent)

    def _compute_expected_utility(
        self, entry: ProceduralMemoryEntry, observation: str, goal: str
    ) -> float:
        relevance = self._compute_relevance(entry, observation, goal)
        rho_mean = entry.procedure.alpha / (entry.procedure.alpha + entry.procedure.beta)
        risk = self._compute_failure_risk(
            entry, observation, theta_risk=getattr(self, "_adaptive_theta_risk", 0.85)
        )
        info_gain = self._compute_information_gain(entry.procedure)

        eu = (relevance * rho_mean * 1.0) - (risk * (1 - rho_mean) * 0.5) + 0.1 * info_gain
        # Stage M (a): multiplicative state-delta confidence. Maps the
        # confidence ∈ [0, 1] into a multiplier ∈ [0.5, 1.0] so the signal
        # damps marginally-useful procedures without zeroing them entirely.
        sdc = self._state_delta_confidence(entry)
        eu *= 0.5 + 0.5 * sdc
        # Stage M (third signal): logprob_confidence — percentile rank of
        # this procedure's mean_logprob against the rolling distribution.
        # Same [0.5, 1.0] multiplier band as state-delta — both signals
        # are ablatable by hardcoding to 1.0 / setting mean_logprob=None
        # everywhere.
        lpc = self._logprob_confidence(entry)
        eu *= 0.5 + 0.5 * lpc
        return max(0.0, eu)

    # Stage M (b): on an unvisited map, raise theta_conf to this floor so
    # cached procedures rarely fire and the LLM is biased toward exploration.
    # 0.6 sits well above the typical EU range (~0.05-0.3) observed in
    # Stage L logs, so virtually all cached procs get rejected on a first
    # visit to a new map.
    _NEW_MAP_THETA = 0.6

    def select_procedure(
        self, observation: str, goal: str, theta_conf: float = 0.25
    ) -> tuple[str | None, float]:
        # Master switch: when the procedure layer is disabled, never select
        # a cached procedure. Every step takes the LLM-fallback path. Keeps
        # vmem + planner + reflection wired (those live on the agent, not
        # the selector).
        if not self.use_procedure_layer:
            return None, 0.0

        # Stage M (b): record the map and bump theta on first visit.
        current_map = self._extract_map_name(observation)
        is_new = self.memory_system.is_new_map(current_map)
        effective_theta = max(theta_conf, self._NEW_MAP_THETA) if is_new else theta_conf
        self.memory_system.record_map_visit(current_map)

        candidates = self._retrieve_candidates(observation, goal, k=10)
        if not candidates:
            return None, 0.0

        utilities: list[tuple[str, float]] = []
        for pk in candidates:
            if pk in self.memory_system.procedural_memory:
                entry = self.memory_system.procedural_memory[pk]
                eu = self._compute_expected_utility(entry, observation, goal)
                utilities.append((pk, eu))

        if not utilities:
            return None, 0.0

        utilities.sort(key=lambda x: x[1], reverse=True)
        best_pk, best_eu = utilities[0]
        logger.debug(
            f"[Selector] best_eu={best_eu:.3f} theta={effective_theta:.3f} "
            f"(new_map={is_new}) candidates={len(utilities)} pk={best_pk}"
        )
        # Strict threshold to prevent executing bad cached procedures
        if best_eu < effective_theta:
            return None, 0.0
        # Stage L: mark the entry as used in the current iter so it survives
        # the next prune_stale_procedures pass.
        if best_pk in self.memory_system.procedural_memory:
            self.memory_system.procedural_memory[
                best_pk
            ].last_used_iter = self.memory_system.current_iter
        return best_pk, min(1.0, best_eu)


# =========================================================
# CONTRASTIVE REFINEMENT
# =========================================================
class ContrastiveRefinementEngine:
    def __init__(
        self,
        n_min_s: int = 3,
        n_min_f: int = 2,
        postcondition_extractor=None,
        spatial_pattern_extractor=None,
    ):
        self.n_min_s = n_min_s
        self.n_min_f = n_min_f
        self.postcondition_extractor = postcondition_extractor
        self.spatial_pattern_extractor = spatial_pattern_extractor

    def should_refine(self, entry: ProceduralMemoryEntry) -> bool:
        # Asymmetric: refine early on first failure if enough successes exist
        # This catches over-generalized procedures before they cause damage
        has_enough_data = (
            len(entry.success_contexts) >= self.n_min_s
            and len(entry.failure_contexts) >= self.n_min_f
        )
        early_failure = len(entry.success_contexts) >= 2 and len(entry.failure_contexts) >= 1
        return has_enough_data or early_failure

    def _extract_spatial_patterns(self, contexts: list[ContrastiveContext]) -> dict[str, set[str]]:
        """Extract spatial patterns: entity -> {direction_distance combinations}"""
        # Use game-specific extractor if provided
        if self.spatial_pattern_extractor:
            return self.spatial_pattern_extractor(contexts)

        # Generic fallback implementation
        patterns = defaultdict(set)
        for ctx in contexts:
            # Parse context string like "goomba_ahead_near_pipe_ahead_mid"
            if not ctx.context:
                continue
            parts = ctx.context.split("_")
            i = 0
            while i < len(parts):
                # Check if this is an entity (not a direction/distance keyword)
                if parts[i] not in ["ahead", "behind", "near", "mid", "far", "clear", "path"]:
                    entity = parts[i]
                    # Collect direction and distance if available
                    spatial_desc = []
                    j = i + 1
                    while j < len(parts) and parts[j] in ["ahead", "behind", "near", "mid", "far"]:
                        spatial_desc.append(parts[j])
                        j += 1
                    if spatial_desc:
                        patterns[entity].add("_".join(spatial_desc))
                    else:
                        patterns[entity].add("present")  # Entity exists but no spatial info
                    i = j
                else:
                    i += 1
        return dict(patterns)

    def refine_procedure(self, entry: ProceduralMemoryEntry) -> ProceduralMemoryEntry:
        if not self.should_refine(entry):
            return entry

        # Extract spatial patterns from success and failure contexts
        success_patterns = self._extract_spatial_patterns(entry.success_contexts)
        failure_patterns = self._extract_spatial_patterns(entry.failure_contexts)

        # Find discriminative spatial features
        discriminative = {
            "required_entities": [],  # Entities present in success, absent in failure
            "preferred_spatial": {},  # Entity spatial configs that work
            "avoided_spatial": {},  # Entity spatial configs that fail
        }

        # Entities that appear only in successes
        success_only = set(success_patterns.keys()) - set(failure_patterns.keys())
        discriminative["required_entities"] = list(success_only)

        # For entities in both, compare spatial configurations
        common_entities = set(success_patterns.keys()) & set(failure_patterns.keys())
        for entity in common_entities:
            success_configs = success_patterns[entity]
            failure_configs = failure_patterns[entity]

            # Spatial configs that work (success only)
            preferred = success_configs - failure_configs
            if preferred:
                discriminative["preferred_spatial"][entity] = list(preferred)

            # Spatial configs that fail (failure only)
            avoided = failure_configs - success_configs
            if avoided:
                discriminative["avoided_spatial"][entity] = list(avoided)

        entry.discriminative_patterns = discriminative

        # Update procedure preconditions with discriminative insights
        # Add required entities and preferred spatial relationships
        new_preconditions = []
        for entity in discriminative["required_entities"]:
            if entity not in entry.procedure.preconditions:
                new_preconditions.append(entity)

        for entity, spatial_configs in discriminative["preferred_spatial"].items():
            # Add most common preferred spatial configuration
            for config in spatial_configs[:1]:  # Take top 1
                precond = f"{entity}_{config}"
                if precond not in entry.procedure.preconditions:
                    new_preconditions.append(precond)

        if new_preconditions:
            entry.procedure.preconditions.extend(new_preconditions)
            logger.info(
                f"Refined procedure {entry.procedure.goal}: added discriminative preconditions {new_preconditions}"
            )

        self._update_postconditions(entry)
        entry.last_refined = time.time()
        return entry

    def _update_postconditions(self, entry: ProceduralMemoryEntry):
        """Identify consistent state changes in successful executions."""
        if not entry.success_contexts:
            return

        # Use injected extractor if available (for domain-specific logic)
        if self.postcondition_extractor:
            updates = self.postcondition_extractor(entry.success_contexts)
            if updates:
                entry.discriminative_patterns.update(updates)
                # Ensure generic postconditions field is populated for compatibility
                if "postconditions_added" in updates:
                    entry.procedure.postconditions = updates["postconditions_added"]
                elif "postconditions" in updates:
                    entry.procedure.postconditions = updates["postconditions"]
            return

        # Default Generic Logic: Just emergent tokens
        success_init_vocab = set()
        success_term_vocab = set()

        for ctx in entry.success_contexts:
            success_init_vocab.update(ctx.observation_init.lower().split())
            success_term_vocab.update(ctx.observation_term.lower().split())

        emergent = list(success_term_vocab - success_init_vocab)[:3]

        entry.procedure.postconditions = emergent
        if entry.discriminative_patterns is not None:
            entry.discriminative_patterns["postconditions"] = emergent


# =========================================================
# META-PROCEDURAL LEARNING
# =========================================================
class MetaProceduralLearner:
    def __init__(
        self, memory_system: EnhancedHierarchicalMemorySystem, precondition_extractor=None
    ):
        self.memory_system = memory_system
        self.precondition_extractor = precondition_extractor

    def should_extract_meta_procedure(
        self, trajectory: dict, procedure_sequence: list[str]
    ) -> bool:
        return len(procedure_sequence) >= 3 and trajectory.get("success", False)

    def extract_meta_procedure(
        self, trajectory: dict, procedure_sequence: list[str]
    ) -> MetaProcedure | None:
        if not self.should_extract_meta_procedure(trajectory, procedure_sequence):
            return None

        goal_meta = f"meta_{trajectory.get('task', 'unknown')[:30]}"
        policy = {"type": "sequential", "ordering": procedure_sequence, "branching_rules": {}}

        # Use starting context as preconditions for this meta-procedure
        start_context = trajectory.get("start_context", "")
        start_observation = trajectory.get("start_observation", "")

        preconditions = []
        if start_context:
            if self.precondition_extractor:
                preconditions = self.precondition_extractor(start_context, start_observation)
            else:
                # Fallback: Split concatenated context into discrete items
                # Handle both comma-separated and underscore-separated formats
                if "," in start_context:
                    preconditions = [p.strip() for p in start_context.split(",") if p.strip()]
                else:
                    # Parse entities with positions (e.g., "brick_x215_y96_goomba_x227_y47_mario_x128")
                    parts = start_context.split("_")
                    i = 0
                    while i < len(parts):
                        if (
                            i + 2 < len(parts)
                            and parts[i + 1].startswith("x")
                            and parts[i + 2].startswith("y")
                        ):
                            # Entity with x,y position: brick_x215_y96
                            preconditions.append(f"{parts[i]}_{parts[i + 1]}_{parts[i + 2]}")
                            i += 3
                        elif i + 1 < len(parts) and parts[i + 1].startswith("x"):
                            # Entity with x position only: mario_x128
                            preconditions.append(f"{parts[i]}_{parts[i + 1]}")
                            i += 2
                        else:
                            # Single entity without position
                            preconditions.append(parts[i])
                            i += 1

        meta = MetaProcedure(
            goal_meta=goal_meta,
            preconditions_meta=preconditions,
            sub_procedures=procedure_sequence,
            composition_policy=policy,
            alpha=2,
            beta=1,
        )
        return meta


# =========================================================
# LLM REASONER
# =========================================================
class FrozenLLMReasoner:
    def __init__(self, generator: Any):
        """
        Args:
            generator: A callable that takes a prompt string and returns a response string.
                       OR an object with an 'invoke' method (LangChain style).
        """
        self.generator = generator

    def _generate(self, prompt: str) -> str:
        try:
            if callable(self.generator):
                return self.generator(prompt)
            elif hasattr(self.generator, "invoke"):
                from langchain_core.messages import HumanMessage

                response = self.generator.invoke([HumanMessage(content=prompt)])
                return response.content if hasattr(response, "content") else str(response)
            else:
                logger.warning(
                    "FrozenLLMReasoner: generator is neither callable nor has invoke method."
                )
                return ""
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    def discover_goal(self, task: str, actions: str, obs_init: str, obs_final: str) -> str:
        prompt = (
            "Infer the high-level intent of this episode as a short noun phrase. "
            'Return JSON: {"goal": "..."}\n'
            f"TASK: {task}\nACTIONS: {actions}\nINIT_OBS: {obs_init}\nFINAL_OBS: {obs_final}"
        )
        return self._generate(prompt)


# =========================================================
# ENHANCED MACLA AGENT
# =========================================================
class EnhancedMACLAAgent:
    def __init__(
        self,
        N_a: int = 1000,
        N_s: int = 100,
        N_p: int = 200,
        N_m: int = 50,
        context_extractor=None,
        postcondition_extractor=None,
        spatial_pattern_extractor=None,
        precondition_extractor=None,
        refinement_config: dict = None,
        use_procedure_layer: bool = True,
    ):
        self.memory_system = EnhancedHierarchicalMemorySystem(N_a, N_s, N_p, N_m)
        self.bayesian_selector = BayesianProcedureSelector(
            self.memory_system, context_extractor=context_extractor
        )
        self.bayesian_selector.use_procedure_layer = use_procedure_layer
        self.precondition_extractor = precondition_extractor

        refinement_config = refinement_config or {}
        n_min_s = refinement_config.get("n_min_s", 3)  # Reduced from 5 for faster iteration
        n_min_f = refinement_config.get("n_min_f", 3)  # Reduced from 5
        self.contrastive_refiner = ContrastiveRefinementEngine(
            n_min_s=n_min_s,
            n_min_f=n_min_f,
            postcondition_extractor=postcondition_extractor,
            spatial_pattern_extractor=spatial_pattern_extractor,
        )
        self.meta_learner = MetaProceduralLearner(
            self.memory_system, precondition_extractor=precondition_extractor
        )

        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "procedures_learned": 0,
            "meta_procedures_learned": 0,
            "procedures_refined": 0,
        }

        # Adaptive stats for self-tuning thresholds (rolling window)
        self._adaptive_stats = {
            "recent_fatals": 0,
            "recent_successes": 0,
            "recent_fallbacks": 0,
            "recent_bayesian": 0,
            "recent_total": 0,
            "best_episode_score": 0.0,
            "stagnant_episodes": 0,
        }

    def _compute_adaptive_theta(self) -> float:
        """Self-adapt confidence threshold based on recent performance."""
        s = self._adaptive_stats
        total = max(s["recent_total"], 1)

        # Base: curriculum decay over time
        base = max(0.10, 0.30 - 0.002 * self.stats["total_executions"])

        # Signal 1: High fatality → be more selective
        fatal_rate = s["recent_fatals"] / total
        if fatal_rate > 0.3:
            base += 0.10

        # Signal 2: All fallback → let procedures execute
        if total > 20:
            fallback_rate = s["recent_fallbacks"] / total
            if fallback_rate > 0.8:
                base -= 0.10

        # Signal 3: Stagnation → explore more
        if s["stagnant_episodes"] >= 3:
            base -= 0.05

        theta = max(0.05, min(0.40, base))
        logger.debug(
            f"[Adaptive] theta={theta:.3f} fatal_rate={fatal_rate:.2f} total={total} stagnant={s['stagnant_episodes']}"
        )
        return theta

    def _decay_adaptive_stats(self):
        """Decay rolling window every 50 steps."""
        if self._adaptive_stats["recent_total"] >= 50:
            for key in ["recent_fatals", "recent_successes", "recent_fallbacks", "recent_bayesian"]:
                self._adaptive_stats[key] = self._adaptive_stats[key] // 2
            self._adaptive_stats["recent_total"] = self._adaptive_stats["recent_total"] // 2

    def update_episode_score(self, score: float):
        """Called at episode end to track stagnation."""
        if score > self._adaptive_stats["best_episode_score"]:
            self._adaptive_stats["best_episode_score"] = score
            self._adaptive_stats["stagnant_episodes"] = 0
        else:
            self._adaptive_stats["stagnant_episodes"] += 1

    def execute_task(self, observation: str, goal: str, **kwargs) -> dict:
        self.stats["total_executions"] += 1
        obs_image = kwargs.get("obs_image", None)
        logger.debug(f"execute_task: obs_image type={type(obs_image)}, is_none={obs_image is None}")
        result = {
            "observation": observation,
            "goal": goal,
            "selected_procedure": None,
            "action_sequence": [],
            "confidence": 0.0,
            "method": "fallback",
            "obs_image": obs_image,
            "reasoning": "",
        }

        # Self-adaptive thresholding based on recent performance
        theta = self._compute_adaptive_theta()
        self._decay_adaptive_stats()

        # Adaptive risk aversion: high death rate → cautious, low → lenient
        s = self._adaptive_stats
        fatal_rate = s["recent_fatals"] / max(s["recent_total"], 1)
        self.bayesian_selector._adaptive_theta_risk = 0.70 + min(0.25, fatal_rate * 0.8)

        pk, conf = self.bayesian_selector.select_procedure(observation, goal, theta_conf=theta)

        if pk:
            self._adaptive_stats["recent_bayesian"] += 1
            self._adaptive_stats["recent_total"] += 1
            entry = self.memory_system.procedural_memory[pk]
            result.update(
                {
                    "selected_procedure": pk,
                    "confidence": conf,
                    "method": "bayesian_procedure",
                    "action_sequence": entry.procedure.steps,
                    "reasoning": entry.procedure.reasoning,
                }
            )
            return result

        self._adaptive_stats["recent_fallbacks"] += 1
        self._adaptive_stats["recent_total"] += 1

        fallback_result = self._generate_fallback_actions(goal, observation, **kwargs)
        if isinstance(fallback_result, tuple):
            result["action_sequence"] = fallback_result[0]
            result["reasoning"] = fallback_result[1]
        else:
            result["action_sequence"] = fallback_result
            result["reasoning"] = "Fallback execution"

        result["confidence"] = 0.5
        return result

    def provide_feedback(
        self,
        execution_result: dict,
        actual_success: bool,
        next_observation: str = "",
        next_obs_image: Any = None,
        is_fatal: bool = False,
        shaped_reward: float | None = None,
    ) -> dict:
        """
        Provides feedback on the execution result and updates memory.

        Returns:
            dict with keys:
            execution_result: dict - the original execution result
                - type: str - one of "atomic_entry", "procedure_learned", "procedure_updated"
                - procedure_key: Optional[str] - the procedure key if applicable
                - was_success: bool - whether the execution was successful
                - method_used: str - the method used (bayesian_procedure, fallback, etc.)
            actual_success: bool - whether the execution was actually successful
            next_observation: str - the observation after execution
            next_obs_image: Any - the observation image after execution
            is_fatal: bool - whether the execution resulted in a fatal failure
        """
        logger.debug(
            f"provide_feedback: next_obs_image type={type(next_obs_image)}, is_none={next_obs_image is None}"
        )
        logger.debug(
            f"provide_feedback: execution_result obs_image type={type(execution_result.get('obs_image'))}, is_none={execution_result.get('obs_image') is None}"
        )

        # Update adaptive stats
        if actual_success:
            self._adaptive_stats["recent_successes"] += 1
        if is_fatal:
            self._adaptive_stats["recent_fatals"] += 1

        update_info = {
            "type": "atomic_entry",
            "procedure_key": None,
            "was_success": actual_success,
            "method_used": execution_result.get("method", "unknown"),
        }

        ## 1. Populate Atomic Memory
        action_seq = execution_result.get("action_sequence", [])

        # If action_seq has only 1 item, store it as a raw string to avoid nested list artifacts in meta learning
        if action_seq and len(action_seq) == 1:
            action_str = str(action_seq[0])
        else:
            action_str = str(action_seq) if action_seq else "wait"

        reward = (
            shaped_reward
            if shaped_reward is not None
            else (1.0 if actual_success else (-1.0 if is_fatal else 0.0))
        )
        self.memory_system.add_atomic_entry(
            action=action_str,
            observation=execution_result.get("observation", ""),
            reward=reward,
            context=self.bayesian_selector._extract_context(
                execution_result.get("observation", next_observation)
            ),
            trajectory_id=execution_result.get("trajectory_id", "global_stream"),
            step_index=self.stats["total_executions"],
            goal=execution_result.get("goal", "unknown"),
        )
        update_info["type"] = "atomic_entry"

        pk = execution_result.get("selected_procedure")
        if pk and pk in self.memory_system.procedural_memory:
            # Existing procedure was used - record its outcome
            logger.debug(
                f"Recording outcome for existing procedure {pk}. {execution_result.get('obs_image')}, {next_obs_image}"
            )
            ctx_obs_init = execution_result.get("observation", "")
            ctx = ContrastiveContext(
                observation_init=ctx_obs_init,
                action_sequence=execution_result.get("action_sequence", []),
                observation_term=next_observation,
                cumulative_reward=shaped_reward
                if shaped_reward is not None
                else (1.0 if actual_success else 0.0),
                trajectory_id=execution_result.get("trajectory_id", "unknown"),
                success=actual_success,
                context=self.bayesian_selector._extract_context(ctx_obs_init),
                preconditions_image=execution_result.get("obs_image"),
                postconditions_image=next_obs_image,
                # Stage M (a): record whether this execution moved the
                # salient game state forward — feeds state_delta_confidence
                # in the next selection cycle.
                state_delta_observed=_state_delta_observed(ctx_obs_init, next_observation),
            )
            self.memory_system.record_execution_outcome(pk, actual_success, ctx, is_fatal=is_fatal)
            update_info["type"] = "procedure_updated"
            update_info["procedure_key"] = pk
            if actual_success:
                self.stats["successful_executions"] += 1
        elif actual_success:
            ## Fallback was used successfully; learn new procedure
            action_seq = execution_result.get("action_sequence", [])
            goal = execution_result.get("goal", "")
            obs = execution_result.get("observation", "")

            if action_seq and goal and obs:
                # Basic context extraction
                context_key = self.bayesian_selector._extract_context(obs)

                # Ensure context_key is hashable for indexing
                if isinstance(context_key, dict):
                    context_key = json.dumps(context_key, sort_keys=True)

                # Parse context into discrete preconditions using domain-specific logic
                if self.precondition_extractor:
                    # Use provided extractor
                    preconditions = self.precondition_extractor(context_key, obs)
                else:
                    # Generic fallback: split by underscore
                    preconditions = []
                    if context_key and context_key != "clear_run":
                        parts = context_key.split("_")
                        reconstructed = []
                        i = 0
                        while i < len(parts):
                            if parts[i].startswith("x") and parts[i][1:].isdigit():
                                if reconstructed:
                                    reconstructed[-1] = f"{reconstructed[-1]}_{parts[i]}"
                                i += 1
                            elif parts[i].startswith("y") and parts[i][1:].isdigit():
                                if reconstructed:
                                    reconstructed[-1] = f"{reconstructed[-1]}_{parts[i]}"
                                i += 1
                            else:
                                reconstructed.append(parts[i])
                                i += 1
                        preconditions = reconstructed
                    else:
                        preconditions = ["clear_run"]

                # Create a temporary ContrastiveContext for this first success
                logger.debug(
                    f"Creating temporary ContrastiveContext for new procedure learning {execution_result.get('obs_image')}."
                )
                temp_ctx = ContrastiveContext(
                    observation_init=obs,
                    action_sequence=action_seq,
                    observation_term=next_observation,
                    cumulative_reward=shaped_reward if shaped_reward is not None else 1.0,
                    trajectory_id=execution_result.get("trajectory_id", "unknown"),
                    success=True,
                    context=str(context_key),
                    preconditions_image=execution_result.get("obs_image"),
                    postconditions_image=next_obs_image,
                    state_delta_observed=_state_delta_observed(obs, next_observation),
                )

                # Extract initial postconditions from this successful execution
                initial_postconditions = []
                if self.contrastive_refiner.postcondition_extractor:
                    try:
                        extracted = self.contrastive_refiner.postcondition_extractor([temp_ctx])
                        if extracted and "postconditions_added" in extracted:
                            initial_postconditions = extracted["postconditions_added"]
                    except Exception as e:
                        logger.warning(f"Failed to extract initial postconditions: {e}")

                # Stage L: stamp the procedure with the map it was captured
                # in so the cache can filter retrieval by current map.
                captured_map = self.bayesian_selector._extract_map_name(obs)

                # Stage M (third signal): stamp mean_logprob from the most
                # recent LLM call. Cleared after use so the next procedure
                # doesn't inherit a stale value. ``None`` is fine — the
                # selector's _logprob_confidence falls back to neutral 0.5.
                pending_lp = self.memory_system._pending_logprob
                self.memory_system._pending_logprob = None

                new_proc = Procedure(
                    goal=goal,
                    preconditions=preconditions,
                    steps=action_seq,
                    postconditions=initial_postconditions,
                    reasoning=execution_result.get("reasoning", ""),
                    confidence=0.6,
                    execution_count=1,  # Mark as executed once
                    alpha=2,  # 1 prior + 1 success
                    map_name=captured_map,
                    mean_logprob=pending_lp,
                )

                pk = self.memory_system.add_procedural_entry(
                    new_proc, contexts={str(context_key)}, goals={goal}, performance=1.0
                )

                # Add the initial success context to the new entry
                if pk in self.memory_system.procedural_memory:
                    self.memory_system.procedural_memory[pk].success_contexts.append(temp_ctx)

                self.stats["procedures_learned"] += 1
                update_info["type"] = "procedure_learned"
                update_info["procedure_key"] = pk

        return update_info

    def learn_from_trajectories(self):
        """
        Analyzes atomic memory to find successful trajectories and extract meta-procedures.
        """
        try:
            # Group atomic entries by trajectory_id
            trajectories = defaultdict(list)
            for entry in self.memory_system.atomic_memory:
                if entry.trajectory_id:
                    trajectories[entry.trajectory_id].append(entry)

            for _traj_id, entries in trajectories.items():
                if not entries:
                    continue

                # Sort by step index to ensure correct order
                entries.sort(key=lambda x: x.step_index)

                # Extract action sequence
                action_seq = [e.action for e in entries]

                # Check if trajectory was "successful" (heuristic: positive total reward or long survival)
                # Generalised: requires some positive reward (score increase) and survival
                total_reward = sum(e.reward for e in entries)

                if len(action_seq) >= 5 and total_reward > 0:
                    # Get context from the start of the sequence
                    start_context = entries[0].context
                    start_observation = entries[0].observation

                    # Determine task name from goal history
                    trajectory_goal = entries[0].goal
                    task_name = (
                        f"{trajectory_goal}_survival"
                        if trajectory_goal != "unknown"
                        else "survival"
                    )

                    # Create a pseudo-trajectory dict for the meta-learner API
                    traj_dict = {
                        "task": task_name,
                        "success": True,
                        "actions": action_seq,
                        "cumulative_reward": total_reward,
                        "start_context": start_context,
                        "start_observation": start_observation,
                    }

                    # Suggest candidate patterns (e.g. simple 3-step patterns)
                    # In a full implementation, this would use sequence mining.
                    # Here we take the last few successful steps as a candidate strategy.
                    candidate_seq = action_seq[-5:] if len(action_seq) > 5 else action_seq

                    meta_proc = self.meta_learner.extract_meta_procedure(traj_dict, candidate_seq)

                    if meta_proc:
                        is_duplicate = False
                        for existing in self.memory_system.meta_procedural_memory.values():
                            if existing.sub_procedures == meta_proc.sub_procedures:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            self.memory_system.add_meta_procedure(meta_proc)
                            self.stats["meta_procedures_learned"] += 1

            # Periodically update ontology if we have enough data and it's not built (or sparse)
            if len(self.memory_system.atomic_memory) > 100 and (
                not self.bayesian_selector.ontology or len(trajectories) > 5
            ):
                # Convert our internal trajectory format to list of dicts for build_ontology
                traj_list = []
                for _traj_id, entries in trajectories.items():
                    if not entries:
                        continue
                    task = entries[0].goal
                    actions = [e.action for e in entries]
                    traj_list.append({"task": task, "actions": actions})

                # Only rebuild if we have a decent amount
                if len(traj_list) > 5:
                    logger.debug(f"Updating ontology with {len(traj_list)} trajectories")
                    self.bayesian_selector.build_ontology(traj_list)

        except Exception as e:
            logger.error(f"Error in learn_from_trajectories: {e}")

    def run_optimisation_cycle(self) -> dict:
        """
        Runs periodic optimisation tasks; refines procedures based on success/failure contexts.
        """
        refined_count = 0
        proc_keys = list(self.memory_system.procedural_memory.keys())

        for pk in proc_keys:
            entry = self.memory_system.procedural_memory[pk]
            if self.contrastive_refiner.should_refine(entry):
                self.contrastive_refiner.refine_procedure(entry)

                for pre in entry.procedure.preconditions:
                    self.memory_system.context_index[pre].add(pk)

                self.stats["procedures_refined"] += 1
                refined_count += 1

        self.learn_from_trajectories()

        return {"procedures_refined_this_cycle": refined_count}

    def get_detailed_memory_stats(self) -> dict:
        """Collects comprehensive stats about the agent's memory and performance."""
        # Calculate derived metrics
        total_procs = len(self.memory_system.procedural_memory)
        avg_success = 0.0
        if total_procs > 0:
            avg_success = (
                sum(
                    entry.procedure.success_rate
                    for entry in self.memory_system.procedural_memory.values()
                )
                / total_procs
            )

        return {
            "agent_stats": self.stats,
            "memory_sizes": {
                "atomic": len(self.memory_system.atomic_memory),
                "procedural": total_procs,
                "meta": len(self.memory_system.meta_procedural_memory),
                "contexts": len(self.memory_system.context_index),
                "goals": len(self.memory_system.goal_index),
            },
            "optimisation": {
                "avg_procedure_success_rate": avg_success,
            },
        }

    def log_procedures(self) -> dict:
        """Logs the currently learned procedures and returns structured data for external logging (e.g. WandB)."""
        logger.info("=== PROCEDURAL MEMORY DUMP ===")

        proc_data = []
        if not self.memory_system.procedural_memory:
            logger.info("  (Empty)")

        for key, entry in self.memory_system.procedural_memory.items():
            proc = entry.procedure
            logger.info(
                f"ID: {key} | Goal: {proc.goal} | Success Rate: {proc.success_rate:.2f} | Execs: {proc.execution_count}"
            )
            logger.info(f"  Steps: {proc.steps}")
            logger.info(f"  Preconditions: {proc.preconditions}")
            if proc.reasoning:
                logger.info(f"  Reasoning: {proc.reasoning[:100]}...")
            if entry.discriminative_patterns:
                logger.info(f"  Refinement Patterns: {entry.discriminative_patterns}")

            # Collect images from success/failure contexts
            context_images = {"success": [], "failure": []}

            sample_pre_image = None
            sample_post_image = None

            logger.info(
                f"  Success contexts: {len(entry.success_contexts)}, Failure contexts: {len(entry.failure_contexts)}"
            )

            # Use most recent contexts for image sampling

            recent_success = entry.success_contexts[-3:] if len(entry.success_contexts) > 0 else []
            for i, ctx in enumerate(recent_success):
                has_pre = ctx.preconditions_image is not None
                has_post = ctx.postconditions_image is not None
                logger.info(f"    Success ctx {i}: pre_img={has_pre}, post_img={has_post}")
                if ctx.preconditions_image or ctx.postconditions_image:
                    context_images["success"].append(
                        {
                            "index": i,
                            "preconditions_image": ctx.preconditions_image,
                            "postconditions_image": ctx.postconditions_image,
                        }
                    )
                    if sample_pre_image is None and ctx.preconditions_image:
                        sample_pre_image = ctx.preconditions_image
                    if sample_post_image is None and ctx.postconditions_image:
                        sample_post_image = ctx.postconditions_image

            recent_failure = entry.failure_contexts[-3:] if len(entry.failure_contexts) > 0 else []
            for i, ctx in enumerate(recent_failure):  # Limit to 3 most recent
                has_pre = ctx.preconditions_image is not None
                has_post = ctx.postconditions_image is not None
                logger.info(f"    Failure ctx {i}: pre_img={has_pre}, post_img={has_post}")
                if ctx.preconditions_image or ctx.postconditions_image:
                    context_images["failure"].append(
                        {
                            "index": i,
                            "preconditions_image": ctx.preconditions_image,
                            "postconditions_image": ctx.postconditions_image,
                        }
                    )
                    # Use failure context images if we still don't have samples
                    if sample_pre_image is None and ctx.preconditions_image:
                        sample_pre_image = ctx.preconditions_image
                    if sample_post_image is None and ctx.postconditions_image:
                        sample_post_image = ctx.postconditions_image

            proc_data.append(
                {
                    "id": key,
                    "goal": proc.goal,
                    "success_rate": proc.success_rate,
                    "executions": proc.execution_count,
                    "steps": str(proc.steps),
                    "preconditions": str(proc.preconditions),
                    "postconditions": str(proc.postconditions),
                    "reasoning": proc.reasoning,
                    "refinements": str(entry.discriminative_patterns),
                    "images": context_images,
                    "sample_pre_image": sample_pre_image,
                    "sample_post_image": sample_post_image,
                }
            )

        logger.info("=== META-PROCEDURAL MEMORY DUMP ===")
        meta_data = []
        if not self.memory_system.meta_procedural_memory:
            logger.info("  (Empty)")

        for key, meta in self.memory_system.meta_procedural_memory.items():
            logger.info(
                f"ID: {key} | Goal: {meta.goal_meta} | Success Rate: {meta.success_rate:.2f}"
            )
            logger.info(f"  Sub-procedures: {meta.sub_procedures}")
            logger.info(f"  Preconditions: {meta.preconditions_meta}")

            meta_data.append(
                {
                    "id": key,
                    "goal": meta.goal_meta,
                    "success_rate": meta.success_rate,
                    "sub_procedures": str(meta.sub_procedures),
                    "preconditions": str(meta.preconditions_meta),
                    "sample_pre_image": None,
                    "sample_post_image": None,
                }
            )

        return {"procedures": proc_data, "meta_procedures": meta_data}


# =========================================================
# LLM-ENHANCED WRAPPER AGENT
# =========================================================
class LLMMACLAAgent(EnhancedMACLAAgent):
    def __init__(
        self,
        generator: Any,
        fallback_generator: Any | None = None,
        N_a=1000,
        N_s=100,
        N_p=200,
        N_m=50,
        context_extractor=None,
        postcondition_extractor=None,
        spatial_pattern_extractor=None,
        precondition_extractor=None,
        refinement_config: dict = None,
        use_procedure_layer: bool = True,
    ):
        super().__init__(
            N_a,
            N_s,
            N_p,
            N_m,
            context_extractor=context_extractor,
            postcondition_extractor=postcondition_extractor,
            spatial_pattern_extractor=spatial_pattern_extractor,
            precondition_extractor=precondition_extractor,
            refinement_config=refinement_config,
            use_procedure_layer=use_procedure_layer,
        )
        self.llm = FrozenLLMReasoner(generator)
        self.fallback_generator = fallback_generator
        self.llm_calls = {"goal_discovery": 0}

    def _generate_fallback_actions(
        self, goal: str, observation: str, **kwargs
    ) -> list[str] | tuple[list[str], str]:
        """
        Generate fallback actions when no suitable procedure is found in memory.
        """
        if self.fallback_generator:
            return self.fallback_generator(goal, observation, **kwargs)

        # If no fallback generator provided, log warning and return a safe default
        logger.warning("No fallback_generator provided to LLMMACLAAgent. Returning default action.")
        return ["wait"]
