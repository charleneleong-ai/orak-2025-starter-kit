"""
UnifiedMaclaAgent: single MACLA agent that works across all games.

Game-specific behavior imported from per-game adapter modules:
  agents/{game}/game_adapter.py

Each adapter exports: action schema, valid actions, prompts, context config,
success detection params, and an extract_action() function.
"""

import base64
import importlib
import io
import re
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Any

import weave
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel

from agents._cognitive import LLMSelfReflector, LLMSubtaskPlanner, VectorMemoryProvider
from agents._harness import format_recent_history, with_retries
from agents.base import BaseOrakAgent
from agents.macla.base import BaseMaclaAgent
from agents.macla.context_extractors import build_context_extractor
from agents.macla.macla_lib import _extract_map_name
from agents.macla.reflexion import build_reflexion_summary
from agents.macla.structured_output import safe_structured_invoke

# Stage R v3 (F2): drop the planner active_subgoal block once the same
# top of the subgoal stack has held for this many steps. Lets the planner
# fall through to its standard heuristics when a subgoal is wedged (cf.
# v2 PalletTown lock — move_to(12,0) reliably stalled at (12,5)). The
# block re-engages automatically the next time the stack mutates.
SUBGOAL_STAGNATION_THRESHOLD = 30

# Universal pathology guard #1 (PR 1 of the generalized agent harness):
# Window of consecutive identical observations after which the planner is
# nudged with a "your last actions did nothing" hint. Game-agnostic —
# fires for any env where the agent walks into a wall (pokemon), spams a
# no-op move (2048), or stands still on a deadly tile (mario).
FUTILE_ACTION_WINDOW = 3
# Action-side sibling: when the last REPEATED_PLAN_WINDOW chosen actions are
# byte-identical strings, the agent is looping on the same rejected plan even
# though the obs is changing (SC2's `Game time` ticks every frame, mario's
# `Time:` counts down, etc.). Catches the cases byte-equality of obs misses.
REPEATED_PLAN_WINDOW = 4
# Semantic sibling (PR 3): hashing-based detectors above fire on byte / token
# identity. This one watches a real per-game progress signal — the adapter's
# STAGNATION_PATTERN (fallback: SCORE_PATTERN, PROGRESS_PATTERN). When the
# extracted value's variance across the last STAGNATION_WINDOW iters is zero,
# the agent is acting but not advancing the game state. Fires the strongest
# loop signal we can produce without per-game logic.
STAGNATION_WINDOW = 20


# Regex helpers for subgoal completion predicates. The adapter's
# completion functions read obs dict keys (map_name, recent_dialog,
# score); we extract those from the raw observation string here so the
# act-loop has a single source for what each subgoal sees.
_DIALOG_RE = re.compile(r"\[Filtered Screen Text\]\s*(.*?)\s*(?:\[|$)", re.DOTALL)
_SCORE_RE = re.compile(r"[Ss]core:?\s*(\d+)")
# Stage R v4 (1): pull (x, y) from "Your position (x, y): (X, Y)" so the
# anti-perseveration counter can track per-tile dwell. Pokemon-only
# format today; games without this line return None and the counter
# stays at zero (no hint, no harm).
_POSITION_RE = re.compile(r"Your position \(x, y\): \((-?\d+),\s*(-?\d+)\)")


def _extract_position(observation: str | None) -> tuple[int, int] | None:
    if not observation:
        return None
    m = _POSITION_RE.search(observation)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _extract_recent_dialog(observation: str) -> str:
    """Lift the filtered screen text block (dialog / menu choices) from the
    pokemon obs. Other games may return "" — TalkTo predicates simply
    won't fire, which is fine."""
    m = _DIALOG_RE.search(observation or "")
    return m.group(1).strip() if m else ""


def _extract_raw_score(observation: str) -> int:
    """Extract the raw 0-7 score from the obs. Falls back to 0."""
    m = _SCORE_RE.search(observation or "")
    return int(m.group(1)) if m else 0


# ── Game adapter registry ────────────────────────────────────────────

GAME_ADAPTERS: dict[str, str] = {
    "super_mario": "agents.super_mario.game_adapter",
    "pokemon_red": "agents.pokemon_red.game_adapter",
    "twenty_fourty_eight": "agents.twenty_fourty_eight.game_adapter",
    "star_craft": "agents.starcraft.game_adapter",
}


# StarCraft emits 5 actions per step as a "1: A\n2: B\n..." string;
# the env's text2action regex parses on `\d+: ` prefixes. Match the same
# shape here so the multi-action payload survives _validate_action unchanged.
_STARCRAFT_MULTI_ACTION_RE = re.compile(r"^\s*\d+:\s+\S")


def _load_adapter(game_name: str) -> ModuleType:
    module_path = GAME_ADAPTERS.get(game_name)
    if not module_path:
        raise ValueError(f"No game adapter for '{game_name}'. Available: {list(GAME_ADAPTERS)}")
    return importlib.import_module(module_path)


# ── Config-driven success detection ──────────────────────────────────


class ConfigSuccessDetector:
    def __init__(self, adapter: ModuleType):
        self.score_pattern = adapter.SCORE_PATTERN
        self.progress_pattern = adapter.PROGRESS_PATTERN
        self.progress_threshold = adapter.PROGRESS_THRESHOLD
        self.success_keywords = adapter.SUCCESS_KEYWORDS
        self.fatal_keywords = adapter.FATAL_KEYWORDS
        self.lives_pattern = adapter.LIVES_PATTERN

    def detect(self, execution_result: dict, prev_state: str, cur_state: str) -> tuple[bool, bool]:
        is_fatal = False
        if self.lives_pattern:
            prev_lives = self._extract_int(self.lives_pattern, prev_state)
            cur_lives = self._extract_int(self.lives_pattern, cur_state)
            if prev_lives is not None and cur_lives is not None and cur_lives < prev_lives:
                is_fatal = True
        for kw in self.fatal_keywords:
            if kw.lower() in cur_state.lower() and kw.lower() not in prev_state.lower():
                is_fatal = True

        strong_success = False
        if self.score_pattern:
            prev_score = self._extract_int(self.score_pattern, prev_state) or 0
            cur_score = self._extract_int(self.score_pattern, cur_state) or 0
            if cur_score > prev_score:
                strong_success = True

        if self.progress_pattern:
            prev_pos = self._extract_float(self.progress_pattern, prev_state) or 0
            cur_pos = self._extract_float(self.progress_pattern, cur_state) or 0
            if cur_pos > prev_pos + self.progress_threshold:
                strong_success = True

        for kw in self.success_keywords:
            if kw.lower() in cur_state.lower() and kw.lower() not in prev_state.lower():
                strong_success = True

        if is_fatal:
            strong_success = False
        return strong_success, is_fatal

    def _extract_int(self, pattern: str, text: str) -> int | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def _extract_float(self, pattern: str, text: str) -> float | None:
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else None


# ── The unified agent ────────────────────────────────────────────────


class UnifiedMaclaAgent(BaseMaclaAgent, BaseOrakAgent):
    """
    Single MACLA agent for all games.
    Game-specific behavior loaded from agents/{game}/game_adapter.py.
    Supports dual-model: fast model for actions, smart model for refinement.
    """

    def __init__(self, config=None, wandb_config=None, game_name: str | None = None):
        BaseOrakAgent.__init__(self, config=config, wandb_config=wandb_config)

        # Resolve game name from config or kwarg
        if not game_name and hasattr(config, "game_config_path") and config.game_config_path:
            game_name = config.game_config_path  # reuse field as game name
        if not game_name:
            raise ValueError("UnifiedMaclaAgent requires game_name (e.g. 'super_mario')")

        self._game_name = game_name
        self._adapter = _load_adapter(game_name)

        # Build strategies from adapter
        self._context_extractor = build_context_extractor(
            self._adapter.CONTEXT_EXTRACTION_MODE,
            self._adapter.CONTEXT_FIELDS,
        )
        self._success_detector = ConfigSuccessDetector(self._adapter)
        # Find the game-specific action schema (a BaseModel subclass, not BaseModel itself)
        self._action_schema = next(
            v
            for v in self._adapter.__dict__.values()
            if isinstance(v, type) and issubclass(v, BaseModel) and v is not BaseModel
        )

        self._init_macla_agent()
        self._memory_provider = self._maybe_init_memory_provider(config)
        self._subtask_planner = self._maybe_init_subtask_planner(config)
        self._self_reflector = self._maybe_init_self_reflector(config)

        # Game-specific observation preprocessor — pokemon needs the env's
        # screen-window 'Map on Screen' expanded to the full explored map,
        # otherwise off-screen tiles (e.g. RedsHouse1f exit door at (3,7))
        # are invisible to the LLM. Adapters opt in by exporting
        # ``make_observation_preprocessor``; mario / 2048 don't need this.
        factory = getattr(self._adapter, "make_observation_preprocessor", None)
        self._obs_preprocessor = factory() if factory is not None else None

    def _maybe_init_self_reflector(self, config: Any):
        """Optional self-reflection module. Precedence:
        config.use_self_reflection (explicit YAML) > adapter
        RECOMMENDED_USE_SELF_REFLECTION (per-game retro finding) > False.

        Adapters can also recommend ``RECOMMENDED_REFLECTION_EVERY`` and
        override the critique prompt via ``SELF_REFLECTOR_SYSTEM_PROMPT``.
        """
        cfg_use = getattr(config, "use_self_reflection", None)
        adapter_use = getattr(self._adapter, "RECOMMENDED_USE_SELF_REFLECTION", False)
        use = cfg_use if cfg_use is not None else adapter_use
        if not use:
            return None
        adapter_every = getattr(self._adapter, "RECOMMENDED_REFLECTION_EVERY", 10)
        cfg_every = getattr(config, "reflection_every", None)
        reflect_every = cfg_every if cfg_every is not None else adapter_every
        adapter_system = getattr(self._adapter, "SELF_REFLECTOR_SYSTEM_PROMPT", None)
        kwargs: dict[str, Any] = dict(
            reflect_every=reflect_every,
            observation_chars=getattr(config, "reflection_max_chars", 600),
        )
        if adapter_system:
            kwargs["system_prompt"] = adapter_system
        reflector = LLMSelfReflector(self._llm, **kwargs)
        logger.info(
            f"[MACLA] self-reflector enabled "
            f"(use_source={'config' if cfg_use is not None else 'adapter'}, "
            f"reflect_every={reflector._reflect_every} "
            f"({'config' if cfg_every is not None else 'adapter'}), "
            f"adapter_system_prompt={'yes' if adapter_system else 'no'})"
        )
        return reflector

    def _build_subtask_history(self) -> str:
        """Build an outcome-tagged history block for the subtask planner.

        Pulls the last K records from the live trajectory buffer and renders
        them with score deltas + state-changed tags. Falls back to the legacy
        one-line form when the trajectory writer hasn't been wired (tests,
        ad-hoc scripts).
        """
        k = max(1, int(getattr(self.config, "subtask_history_steps", 8)))
        writer = getattr(self, "_trajectory_writer", None)
        if writer is not None:
            recent = writer.recent(k)
            if recent:
                return format_recent_history(recent)

        last_action = getattr(self, "_last_action", "none")
        prev_state = getattr(self, "_prev_state_str", "") or ""
        prev_summary = prev_state[:200] if prev_state else "(no prior state)"
        return f"Last action: {last_action}\nPrior state snippet: {prev_summary}"

    def _maybe_init_subtask_planner(self, config: Any):
        """Stage D: optional subtask planner for long-horizon games (pokemon).
        Adds 1 LLM call per replan_every steps. Default off.

        Game adapters can override the planner's system prompt by exporting a
        module-level ``SUBTASK_PLANNER_SYSTEM`` constant — used to bake game-
        specific waypoint chains into the planner (see pokemon_red.game_adapter)."""
        if not getattr(config, "use_subtask_planning", False):
            return None
        kwargs: dict[str, Any] = dict(
            replan_every=getattr(config, "subtask_replan_every", 1),
            observation_chars=getattr(config, "subtask_observation_chars", 600),
        )
        adapter_system = getattr(self._adapter, "SUBTASK_PLANNER_SYSTEM", None)
        if adapter_system:
            kwargs["system_prompt"] = adapter_system
        planner = LLMSubtaskPlanner(llm=self._llm, **kwargs)
        logger.info(
            f"[MACLA] subtask planner enabled "
            f"(replan_every={planner._replan_every}, "
            f"observation_chars={planner._observation_chars}, "
            f"adapter_system_prompt={'yes' if adapter_system else 'no'})"
        )
        return planner

    def _maybe_init_memory_provider(self, config: Any):
        """Stage C: optional vector-memory provider. Activated by
        ``config.use_vector_memory``; default off so existing runs are
        unchanged. Provider has its own observability surface via stats()."""
        if not getattr(config, "use_vector_memory", False):
            return None
        provider = VectorMemoryProvider(
            max_memories=getattr(config, "vector_memory_max", 100),
            default_top_k=getattr(config, "vector_memory_top_k", 3),
            default_threshold=getattr(config, "vector_memory_threshold", 0.5),
            use_mmr=getattr(config, "vector_memory_use_mmr", False),
            mmr_lambda=getattr(config, "vector_memory_mmr_lambda", 0.5),
            repetition_decay_alpha=getattr(config, "vector_memory_decay_alpha", 0.0),
            repetition_decay_window=getattr(config, "vector_memory_decay_window", 20),
        )
        provider.initialize(
            session_id=str(getattr(self.wandb_config, "run_id", "") or ""),
            game_name=self._game_name,
        )
        logger.info(
            f"[MACLA] vector memory provider enabled "
            f"(max={provider._max_memories}, top_k={provider._default_top_k}, "
            f"threshold={provider._default_threshold})"
        )
        return provider

    def _init_episode_subgoals(self) -> None:
        """Episode-start setup — build Reflexion summary from prev
        iter's game_states.jsonl, seed an initial subgoal stack if empty.

        Fires once per episode, gated by _subgoal_init_done in _get_action.
        Safe no-op when adapter doesn't expose SUBGOAL_TEMPLATES or
        initial_subgoal_stack.
        """
        templates = getattr(self._adapter, "SUBGOAL_TEMPLATES", None)
        if templates is None:
            self._reflexion_summary = ""
            return

        mem = self._macla_agent.memory_system

        # Reflexion: scan GAME_DATA_DIR for the most recently completed
        # iter (the one with evaluation_summary.json) and build a summary.
        self._reflexion_summary = ""
        try:
            from evaluation_utils.commons import GAME_DATA_DIR  # noqa: PLC0415

            game_dir = Path(GAME_DATA_DIR) / self._game_name
            if game_dir.exists():
                completed = sorted(
                    (
                        d
                        for d in game_dir.iterdir()
                        if d.is_dir() and (d / "evaluation_summary.json").exists()
                    ),
                    key=lambda p: p.stat().st_mtime,
                )
                # Exclude the current run dir (last-modified is itself); take
                # the second-most-recent as the prev iter.
                if len(completed) >= 1:
                    prev_run = completed[-1]
                    summary = build_reflexion_summary(prev_run, self._adapter)
                    if summary:
                        self._reflexion_summary = summary
                        logger.info(f"[MACLA] built Reflexion summary from {prev_run.name}")
        except Exception as e:
            logger.warning(f"[MACLA] Reflexion build failed: {e}")

        # Seed initial subgoal stack if empty (fresh-iter or post-prune).
        if mem.subgoal_depth() == 0:
            builder = getattr(self._adapter, "initial_subgoal_stack", None)
            if builder is not None:
                try:
                    stack = builder()
                    mem.set_subgoal_stack(stack)
                    logger.info(
                        f"[MACLA] seeded initial subgoal stack "
                        f"({len(stack)} entries; top={stack[-1].name if stack else 'n/a'})"
                    )
                except Exception as e:
                    logger.warning(f"[MACLA] initial_subgoal_stack failed: {e}")

    def _determine_game_phase(self, observation: str) -> tuple[str, float]:
        """Game phase based on evaluation score (0-100 scale)."""
        score = getattr(self, "_last_score", 0) or 0
        if score < 15:
            return "early", float(score) / 100
        elif score < 50:
            return "mid", float(score) / 100
        return "late", float(score) / 100

    # ── Core MACLA _get_action (called by BaseMaclaAgent.get_action) ──

    @weave.op()
    def _get_action(self, task_description: str, cur_state_str: str, obs_image=None):
        """MACLA action loop: feedback → execute → log → validate → return 8-tuple."""
        if self._obs_preprocessor is not None:
            cur_state_str = self._obs_preprocessor.preprocess(cur_state_str)

        # One-shot episode-start hook — build Reflexion summary from
        # prev iter's game_states.jsonl and seed the subgoal stack if
        # empty. Fires on the first _get_action of each episode.
        if not getattr(self, "_subgoal_init_done", False):
            self._subgoal_init_done = True
            self._init_episode_subgoals()

        # Refresh the self-reflection critique (cached between reflect_every
        # invocations). _base_fallback reads self._last_critique to inject
        # into the action LLM's user prompt.
        if self._self_reflector is not None:
            try:
                self._last_critique = self._self_reflector.reflect(
                    observation=cur_state_str,
                    last_action=getattr(self, "_last_action", "none"),
                    history=self._build_subtask_history(),
                )
            except Exception as e:
                logger.warning(f"[MACLA] self-reflector failed; continuing without: {e}")
                self._last_critique = getattr(self, "_last_critique", "")

        # 1. Provide feedback on previous execution
        update_info = self._provide_feedback(self._prev_state_str, cur_state_str, obs_image)

        # 2. Execute task using MACLA
        action, execution_result, memory_stats = self._execute_task(
            task_description, cur_state_str, obs_image
        )

        # 3. Log MACLA stats
        self._log_stats(update_info, memory_stats)

        # 4. Validate action
        action = self._validate_action(action)

        # 5. Build output with LLM reasoning
        avg_exec_time = (
            sum(self._execution_times) / len(self._execution_times) if self._execution_times else 0
        )
        method = execution_result.get("method", "unknown")
        confidence = execution_result.get("confidence", 0.0)
        llm_reasoning = execution_result.get("reasoning", "") or getattr(self, "_llm_reasoning", "")
        selected_proc = execution_result.get("selected_procedure", "")

        reasoning_parts = [f"[{method}] conf={confidence:.3f} time={avg_exec_time:.1f}s"]
        if selected_proc:
            reasoning_parts.append(f"procedure={selected_proc}")
        if llm_reasoning:
            reasoning_parts.append(f"\n{llm_reasoning}")

        reasoning = " | ".join(reasoning_parts[:2]) + (
            f"\n{llm_reasoning}" if llm_reasoning else ""
        )
        output_text = f"Action: {action}\nMethod: {method}\nConfidence: {confidence:.3f}\nReasoning: {llm_reasoning}"
        goal = self._get_task_description({"task_description": task_description})
        game_phase, _ = self._determine_game_phase(cur_state_str)

        # Stage C: write a memory whenever score increased — that's the
        # signal worth recalling later. Skip the cold start (first few steps
        # are noisy) and the trivial no-change case.
        #
        # Event template includes the observation snippet + procedure name so
        # each memory has unique semantic content. Earlier template ("Action X
        # method Y delta N") was too uniform and got collapsed into 2 dedup
        # buckets per episode by the cosine-similarity dedup. Richer text means
        # the embedder produces distinct vectors → memory bank actually grows
        # → retrieval has more signal to work with.
        if self._memory_provider is not None and self._step_count > 3:
            score_delta = (self._last_score or 0) - getattr(self, "_prev_score_for_memory", 0) or 0
            if score_delta > 0:
                # Trim observation to ~200 chars — enough to identify game
                # state (board layout for 2048, position for mario, map for
                # pokemon) without blowing up embedding length.
                obs_snippet = (cur_state_str or "").strip().replace("\n", " ")[:200]
                self._memory_provider.add_event(
                    f"step={self._step_count} action={action} method={method} "
                    f"procedure={selected_proc or 'none'} conf={confidence:.2f} "
                    f"score_delta={score_delta} obs={obs_snippet}",
                    metadata={
                        "step": self._step_count,
                        "method": method,
                        "procedure": selected_proc or "",
                        "score_delta": score_delta,
                        "game_phase": game_phase,
                    },
                )
            self._prev_score_for_memory = self._last_score or 0

        # Use the actual injected prompt if _base_fallback was called this
        # step; otherwise fall back to the synthetic stub.
        prompt_for_log = (
            getattr(self, "_last_llm_user_text", None) or f"Goal: {goal}\nObs: {cur_state_str}"
        )
        # Reset so a step that hits procedure cache (no LLM) shows the stub
        self._last_llm_user_text = None

        # Surface LLM token usage captured by safe_structured_invoke so the
        # base get_action plumbing logs prompt/completion/total tokens. Only
        # set on steps where _base_fallback actually invoked the LLM —
        # procedure-cache hits leave usage at None.
        last_usage = getattr(self, "_last_llm_usage", None)
        if last_usage is not None:
            if isinstance(memory_stats, dict):
                memory_stats.setdefault("usage", last_usage)
            self._last_llm_usage = None

        return (
            action,
            reasoning,
            output_text,
            memory_stats,
            prompt_for_log,
            game_phase,
            self._last_update_type,
            update_info,
        )

    # ── Abstract method implementations ──────────────────────────────

    def _extract_loop_state(self, obs):
        """Dispatch to the per-game adapter's ``extract_loop_state`` if it
        exposes one. Adapters that haven't been wired return None and the
        LoopDetector stays silent for that game.

        Why: the BaseMaclaAgent default is None, but UnifiedMaclaAgent
        is the agent class actually used by the gemma_stage_a config —
        without this dispatch the per-game extractor on
        PokemonRedMaclaAgent never runs, the detector never sees a
        state primitive, and the [Stuck Detector] block stays empty even
        when the agent is clearly looping.
        """
        adapter_extract = getattr(self._adapter, "extract_loop_state", None)
        if adapter_extract is None:
            return None
        return adapter_extract(obs)

    def _extract_context(self, observation: str) -> str | dict:
        return self._context_extractor.extract(observation)

    def extract_preconditions(self, context_key: str, observation: str) -> list[str]:
        return self._context_extractor.extract_preconditions(context_key, observation)

    def extract_postconditions(self, success_contexts) -> dict[str, Any]:
        """Extract postconditions from success contexts for procedure refinement."""
        if not success_contexts:
            return {}
        # Use the terminal observation from the most recent success context
        last = success_contexts[-1]
        obs = getattr(last, "observation_term", "") or ""
        if isinstance(obs, list):
            obs = "\n".join(str(item) for item in obs)
        context = self._extract_context(obs) if obs else {}
        if isinstance(context, str):
            return {"postconditions_added": [context]} if context else {}
        return context if isinstance(context, dict) else {}

    def _detect_success(
        self, execution_result: dict, prev_state: str, cur_state: str
    ) -> tuple[bool, bool]:
        return self._success_detector.detect(execution_result, prev_state, cur_state)

    def _validate_action(self, action: str) -> str:
        if action in self._adapter.VALID_ACTIONS:
            return action
        for va in self._adapter.VALID_ACTIONS:
            if action.lower().strip() == va.lower().strip():
                return va
        # Allow game-specific action formats to pass through
        if action.startswith("use_tool(") or action.startswith("Jump Level:"):
            return action
        if _STARCRAFT_MULTI_ACTION_RE.match(action):
            return action
        return self._adapter.DEFAULT_ACTION

    def _get_task_description(self, game_info: dict) -> str:
        return game_info.get("task_description", self._adapter.DEFAULT_GOAL)

    def _get_default_goal(self) -> str:
        return self._adapter.DEFAULT_GOAL

    def _get_default_action(self) -> str:
        return self._adapter.DEFAULT_ACTION

    def _detect_futile_action(self, observation: str) -> str | None:
        """Universal no-op detector: returns a planner hint when the last
        FUTILE_ACTION_WINDOW observations are byte-identical, meaning the
        agent's recent actions produced no observable change."""
        if not hasattr(self, "_obs_hash_window"):
            self._obs_hash_window = deque(maxlen=FUTILE_ACTION_WINDOW)
            self._futile_streak_logged = False

        obs_hash = hash(observation)
        self._obs_hash_window.append(obs_hash)

        if len(self._obs_hash_window) < FUTILE_ACTION_WINDOW or not all(
            h == obs_hash for h in self._obs_hash_window
        ):
            self._futile_streak_logged = False
            return None

        if not self._futile_streak_logged:
            logger.info(
                f"[MACLA] futile_action_hint fired (last {FUTILE_ACTION_WINDOW} "
                f"obs identical — actions producing no observable change)"
            )
            self._futile_streak_logged = True

        return (
            f"[Futile-action notice] Your last {FUTILE_ACTION_WINDOW - 1} actions "
            f"produced no observable change in the game state — the action you "
            f"keep choosing is being rejected by the environment (walking into a "
            f"wall, picking an invalid move, etc.). Pick a clearly different "
            f"action this step."
        )

    def _detect_repeated_plan(self) -> str | None:
        """Action-side sibling of _detect_futile_action: fires when the
        last REPEATED_PLAN_WINDOW chosen action plans are identical strings,
        regardless of whether obs is byte-stable. Catches envs where obs
        ticks continuously (SC2, mario) but the agent loops on the same
        rejected plan."""
        if not hasattr(self, "_plan_history"):
            self._plan_history = deque(maxlen=REPEATED_PLAN_WINDOW)
            self._repeated_plan_logged = False

        if len(self._plan_history) < REPEATED_PLAN_WINDOW or len(set(self._plan_history)) > 1:
            self._repeated_plan_logged = False
            return None

        if not self._repeated_plan_logged:
            logger.info(
                f"[MACLA] repeated_plan_hint fired (last {REPEATED_PLAN_WINDOW} "
                f"plans identical: {next(iter(self._plan_history))!r})"
            )
            self._repeated_plan_logged = True

        return (
            f"[Repeated-plan notice] You have chosen the same action plan "
            f"{REPEATED_PLAN_WINDOW} steps in a row and the goal isn't "
            f"advancing — the environment is rejecting it or it's a no-op. "
            f"Pick a structurally different plan this step (different tool, "
            f"different target, different sub-goal)."
        )

    def _detect_progress_stagnation(self, observation: str) -> str | None:
        """Semantic detector: watches a per-game progress signal extracted via
        adapter.STAGNATION_PATTERN (fallback: SCORE_PATTERN, PROGRESS_PATTERN).
        Fires when the extracted numeric value has zero variance across the
        last STAGNATION_WINDOW iters — i.e. the agent is acting but not
        advancing the game-native progress metric.

        On SC2 this watches `Supply used` (army size). On pokemon/mario/2048
        it falls back to SCORE_PATTERN (milestone / x_pos / log2 max-tile).
        """
        pattern = (
            getattr(self._adapter, "STAGNATION_PATTERN", None)
            or self._adapter.SCORE_PATTERN
            or self._adapter.PROGRESS_PATTERN
        )
        if not pattern:
            return None

        m = re.search(pattern, observation or "")
        if not m:
            return None
        try:
            value = float(m.group(1))
        except (ValueError, IndexError):
            return None

        if not hasattr(self, "_stagnation_window"):
            self._stagnation_window = deque(maxlen=STAGNATION_WINDOW)
            self._stagnation_logged = False
        self._stagnation_window.append(value)

        if (
            len(self._stagnation_window) < STAGNATION_WINDOW
            or len(set(self._stagnation_window)) > 1
        ):
            self._stagnation_logged = False
            return None

        if not self._stagnation_logged:
            logger.info(
                f"[MACLA] progress_stagnation_hint fired "
                f"(value={value} flat for {STAGNATION_WINDOW} iters)"
            )
            self._stagnation_logged = True

        return (
            f"[Progress-stagnation notice] The game-native progress signal "
            f"(value={value:g}) hasn't moved in {STAGNATION_WINDOW} steps. "
            f"Your recent plans aren't advancing the goal. Try a fundamentally "
            f"different action class (build a structure, tech up, switch target)."
        )

    def _top_procedures_hint(self, k: int = 2) -> str:
        """Optional hint suffix used by all three detectors: lists the top-K
        procedures from memory by success rate. Helps the planner re-discover
        plans it's previously executed successfully, instead of inventing a
        new (likely-bad) variant.
        """
        if not self._macla_agent or not hasattr(self._macla_agent, "memory_system"):
            return ""
        procs = getattr(self._macla_agent.memory_system, "procedural_memory", None)
        if not procs:
            return ""
        # success_rate + goal live on entry.procedure (Procedure dataclass),
        # NOT on the wrapping ProceduralMemoryEntry.
        ranked = sorted(procs.items(), key=lambda kv: kv[1].procedure.success_rate, reverse=True)
        if not ranked or ranked[0][1].procedure.success_rate <= 0:
            return ""
        bullets = []
        for key, entry in ranked[:k]:
            goal = (getattr(entry.procedure, "goal", "") or "")[:60]
            bullets.append(f"{key} (success={entry.procedure.success_rate:.2f}, {goal})")
        return f" Top success-rate procedures available: {' · '.join(bullets)}."

    def _base_fallback(self, goal: str, observation: str, **kwargs) -> tuple[list[str], str]:
        """LLM fallback using game adapter prompts + structured output."""
        obs_image = kwargs.get("obs_image")

        # Universal pathology guard: if last K obs are identical, the
        # agent's actions are no-ops — nudge the planner before the LLM
        # call. Hash uses the raw observation BEFORE per-game hints
        # (map_graph, looped_positions, etc.) are prepended below, so
        # adding hints later doesn't artificially break the streak.
        futile_hint = self._detect_futile_action(observation)
        if futile_hint:
            observation = f"{futile_hint}{self._top_procedures_hint()}\n\n{observation}"

        # Action-side sibling: check before the LLM call whether the last
        # K=4 plans returned were identical. Fires the [Repeated-plan notice]
        # so the planner sees the loop hint before this step's plan is chosen.
        repeated_plan_hint = self._detect_repeated_plan()
        if repeated_plan_hint:
            observation = f"{repeated_plan_hint}{self._top_procedures_hint()}\n\n{observation}"

        # Semantic sibling: watches a real game-native progress signal. Fires
        # the loudest "stuck" hint we can produce — when neither obs nor plans
        # are byte-identical but the game's progress metric is flat.
        stagnation_hint = self._detect_progress_stagnation(observation)
        if stagnation_hint:
            observation = f"{stagnation_hint}{self._top_procedures_hint()}\n\n{observation}"

        # Use defaultdict-style formatting to handle any template vars
        class SafeDict(dict):
            def __missing__(self, key):
                return ""

        user_text = self._adapter.USER_PROMPT_TEMPLATE.format_map(
            SafeDict(
                last_action=getattr(self, "_last_action", "none"),
                cur_state_str=observation,
                task_description=goal,
                prev_state_str=getattr(self, "_prev_state_str", ""),
            )
        )

        # Prepend the latest self-reflection critique (refreshed in _get_action
        # every reflect_every steps). Cheaper than a per-step LLM call and lets
        # the action LLM see meta-feedback that the env doesn't surface.
        if self._self_reflector is not None:
            critique = getattr(self, "_last_critique", "") or ""
            if critique:
                user_text = f"[Recent critique]\n{critique}\n\n{user_text}"

        # Vector-memory recall — one query feeds both the action LLM (Stage C)
        # and the subtask planner (Stage D). Avoids duplicate embedding +
        # retrieval cost per step.
        recalled = ""
        if self._memory_provider is not None:
            recalled = self._memory_provider.prefetch(f"{goal} | {observation[:300]}")

        # Stage C: prepend recalled memories to the action LLM's user prompt.
        if recalled:
            user_text = f"[Recalled memories from prior steps]\n{recalled}\n\n{user_text}"

        # Stage D: ask the subtask planner for a near-term sub-goal and
        # prepend it. For long-horizon games (pokemon) this is the missing
        # decomposition between the overall goal and the per-step action.
        if self._subtask_planner is not None:
            try:
                history_str = self._build_subtask_history()
                # Cross-episode learning: feed the same recalled memories into
                # the planner so it sees "this kind of state previously led to
                # score=+1 via X" and biases the subtask accordingly.
                if recalled:
                    history_str = (
                        f"### Recalled prior memories\n{recalled}\n\n"
                        f"### Recent steps (this episode)\n{history_str}"
                    )
                # Stage N: novelty hint for the planner. The selector-side
                # theta-bump (Stage M b) fired 0 times in the n=5 sweep because
                # the cache was OaksLab-only and select_procedure early-returned
                # on no candidates. Moved here so the hint reaches the LLM
                # regardless of cache state; marked visited only after the
                # planner has been told, so each new map fires the hint once.
                mem = self._macla_agent.memory_system
                current_map = _extract_map_name(observation)
                novelty_hint = mem.map_visit_status(current_map)
                if novelty_hint:
                    history_str = f"### Novelty\n{novelty_hint}\n\n{history_str}"
                    mem.record_map_visit(current_map)
                    logger.info(f"[MACLA] novelty hint fired for map={current_map}")
                # Stage P + Q + R v4: every-step map-graph hint prepended
                # to the observation. The 2026-05-15 diagnosis named this
                # as the cheapest M5-gate intervention. v4 routes through
                # the per-game adapter so we pick up Stage Q's exit-tile
                # coordinates (which were defined but never reached the
                # planner because unified.py was still calling the
                # hand-authored ~30-map MAP_GRAPH in macla_lib). Mario /
                # 2048 don't export graph_hint — getattr returns None and
                # the block is skipped.
                graph_hint_fn = getattr(self._adapter, "graph_hint", None)
                graph_hint = (
                    graph_hint_fn(current_map, mem.visited_maps)
                    if graph_hint_fn is not None
                    else None
                )
                if graph_hint:
                    observation = f"{graph_hint}\n\n{observation}"
                    logger.info(f"[MACLA] map_graph_hint fired for map={current_map}")
                # Stage R v4 (1): record this step's position and surface
                # any over-threshold loops. v3 introspect found single
                # tiles revisited 44× per episode without the planner
                # noticing — injecting the count directly closes that
                # blind spot. Per-iter reset (see macla_lib __setstate__)
                # so cumulative iters don't inherit stale loop trauma.
                pos = _extract_position(observation)
                if pos is not None:
                    mem.record_position(current_map, pos[0], pos[1])
                looped_hint = mem.looped_positions_hint()
                if looped_hint:
                    observation = f"{looped_hint}\n\n{observation}"
                    logger.info(
                        f"[MACLA] looped_positions_hint fired ({sum(1 for v in mem.position_visits.values() if v >= 5)} cells over threshold)"
                    )
                # Prepend the per-iter Reflexion summary (built once
                # per episode in record_episode_end_into_reflexion).
                reflexion = getattr(self, "_reflexion_summary", "")
                if reflexion:
                    history_str = f"{reflexion}\n\n{history_str}"

                # Check the top subgoal's completion predicate against
                # the current obs and pop on fire (may cascade). The
                # active subgoal is threaded into the planner as a soft
                # preference (v3 — v2's HARD CONSTRAINT phrasing trapped
                # the planner in PalletTown). When stagnation crosses
                # SUBGOAL_STAGNATION_THRESHOLD the block is dropped
                # entirely so the planner can break out (F2).
                active_subgoal_str: str | None = None
                templates = getattr(self._adapter, "SUBGOAL_TEMPLATES", None)
                if templates is not None:
                    obs_for_completion = {
                        "map_name": current_map or "",
                        "recent_dialog": _extract_recent_dialog(observation),
                        "score": _extract_raw_score(observation),
                    }
                    popped = mem.check_active_subgoal_completion(obs_for_completion)
                    if popped is not None:
                        logger.info(
                            f"[MACLA] subgoal completed: {popped.name} "
                            f"(remaining depth={mem.subgoal_depth()})"
                        )

                    mem.record_subgoal_step()
                    active = mem.peek_subgoal()
                    if (
                        active is not None
                        and mem.subgoal_stagnation_steps < SUBGOAL_STAGNATION_THRESHOLD
                    ):
                        suggested = (
                            f" (suggested tools: {', '.join(active.suggested_tools)})"
                            if active.suggested_tools
                            else ""
                        )
                        active_subgoal_str = f"{active.name}: {active.description}{suggested}"
                    elif active is not None:
                        logger.info(
                            f"[MACLA] subgoal escape valve fired: {active.name} "
                            f"stagnation={mem.subgoal_stagnation_steps} "
                            f">= {SUBGOAL_STAGNATION_THRESHOLD} — dropping from planner prompt"
                        )

                subtask = self._subtask_planner.plan(
                    goal=goal,
                    observation=observation,
                    history=history_str,
                    active_subgoal=active_subgoal_str,
                )
                if subtask:
                    user_text = (
                        f"[Current sub-goal — focus on this for the next few steps]\n"
                        f"{subtask}\n\n"
                        f"{user_text}"
                    )
            except Exception as e:
                logger.warning(f"[MACLA] subtask planner failed; continuing without: {e}")

        # Text-only models (most local models): pass plain string content.
        # Vision models (Gemini, OpenAI gpt-4o, Llama-4-Scout): pass list with image.
        supports_vision = getattr(self, "_supports_vision", True)
        if supports_vision and obs_image:
            user_content = [{"type": "text", "text": user_text}]
            buffered = io.BytesIO()
            obs_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                }
            )
            human_content = user_content
        else:
            human_content = user_text

        messages = [
            SystemMessage(content=self._adapter.SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        # Save the actual injected prompt for telemetry — without this the
        # logger sees the stub prompt from _get_action and we can't verify
        # whether vmem/subtask injections actually fire.
        self._last_llm_user_text = user_text

        try:
            result, usage = with_retries(
                lambda: safe_structured_invoke(self._llm, messages, self._action_schema),
                label="macla_unified.llm",
            )
            reasoning = getattr(result, "reasoning", "")
            action = self._adapter.extract_action(result)
            self._llm_reasoning = reasoning
            # Stash usage so BaseMaclaAgent.get_action can pull it off and
            # surface tokens_prompt/completion/total into log_extras.
            self._last_llm_usage = usage
            # Stage M (third signal): hand off the per-call mean_logprob to
            # the macla memory system. Append to the rolling deque used by
            # the selector's percentile-rank calibration, and stash as
            # _pending_logprob so provide_feedback can stamp it on any
            # newly-learned Procedure. Cleared automatically next step.
            if usage and usage.get("mean_logprob") is not None:
                mem = self._macla_agent.memory_system
                mem._recent_logprobs.append(usage["mean_logprob"])
                mem._pending_logprob = usage["mean_logprob"]
            # Record the chosen plan for the action-side repeat detector.
            # Initialised lazily inside _detect_repeated_plan; if the
            # detector hasn't been called yet (first step), create here.
            if not hasattr(self, "_plan_history"):
                self._plan_history = deque(maxlen=REPEATED_PLAN_WINDOW)
            self._plan_history.append(action)
            return [action], reasoning
        except Exception as e:
            logger.error(f"Unified fallback failed after retries: {e}")
            self._mark_fallback(f"llm_error: {type(e).__name__}: {str(e)[:200]}")
            self._last_llm_usage = None
            return [self._adapter.DEFAULT_ACTION], f"Fallback error: {e}"

    def calculate_metrics(self, game_info: dict[str, Any]) -> dict[str, Any]:
        if hasattr(self._adapter, "calculate_metrics"):
            return self._adapter.calculate_metrics(game_info)
        metrics = {}
        for field_name in self._adapter.METRIC_FIELDS:
            if field_name in game_info:
                try:
                    metrics[field_name] = float(game_info[field_name])
                except (ValueError, TypeError):
                    pass
        return metrics

    def record_episode_end(self, episode, game_name, seed, score):
        BaseOrakAgent.record_episode_end(self, episode, game_name, seed, score)
        self._record_episode_end(episode, score)
        # Stage R v4 (5): feed the iter's raw score into the memory so
        # the next iter's prune_low_score_iter (called on checkpoint
        # load) can actually fire. The write site was missing — v2/v3
        # iter 1+2 scored 2.0/7 (below PROC_CACHE_MIN_ITER_SCORE = 4.0)
        # but procedures_pruned_low_score stayed at 0 forever because
        # mem.last_iter_score stayed None. Both threshold and score
        # are on the raw 0-7 scale — no normalisation here.
        if self._macla_agent and hasattr(self._macla_agent, "memory_system"):
            self._macla_agent.memory_system.last_iter_score = float(score)
        if self._memory_provider is not None:
            stats = self._memory_provider.stats()
            logger.info(f"[MACLA] vector memory ep={episode} stats: {stats}")
            self._memory_provider.on_session_end()
        if self._subtask_planner is not None:
            stats = self._subtask_planner.stats()
            logger.info(f"[MACLA] subtask planner ep={episode} stats: {stats}")
        # Reset episode-init flag so next episode rebuilds
        # Reflexion summary + re-seeds the subgoal stack.
        self._subgoal_init_done = False
        # Reset the futile-action detector window — short-episodic games
        # (mario, 2048) restart at a fresh state each episode, and the
        # previous episode's terminal frame shouldn't anchor the streak.
        if hasattr(self, "_obs_hash_window"):
            self._obs_hash_window.clear()
            self._futile_streak_logged = False
        if hasattr(self, "_plan_history"):
            self._plan_history.clear()
            self._repeated_plan_logged = False
        if hasattr(self, "_stagnation_window"):
            self._stagnation_window.clear()
            self._stagnation_logged = False
        if self._macla_agent and hasattr(self._macla_agent, "memory_system"):
            mem = self._macla_agent.memory_system
            if hasattr(mem, "subgoal_depth"):
                logger.info(f"[MACLA] episode end — subgoal_stack depth={mem.subgoal_depth()}")
