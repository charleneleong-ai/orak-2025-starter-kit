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
from types import ModuleType
from typing import Any

import weave
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel

from agents._cognitive import LLMSubtaskPlanner, VectorMemoryProvider
from agents._harness import with_retries
from agents.base import BaseOrakAgent
from agents.macla.base import BaseMaclaAgent
from agents.macla.context_extractors import build_context_extractor
from agents.macla.structured_output import safe_structured_invoke

# ── Game adapter registry ────────────────────────────────────────────

GAME_ADAPTERS: dict[str, str] = {
    "super_mario": "agents.super_mario.game_adapter",
    "pokemon_red": "agents.pokemon_red.game_adapter",
    "twenty_fourty_eight": "agents.twenty_fourty_eight.game_adapter",
}


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

    def _build_subtask_history(self) -> str:
        """Build a compact history string for the subtask planner."""
        last_action = getattr(self, "_last_action", "none")
        prev_state = getattr(self, "_prev_state_str", "") or ""
        prev_summary = prev_state[:200] if prev_state else "(no prior state)"
        return f"Last action: {last_action}\nPrior state snippet: {prev_summary}"

    def _maybe_init_subtask_planner(self, config: Any):
        """Stage D: optional subtask planner for long-horizon games (pokemon).
        Adds 1 LLM call per replan_every steps. Default off."""
        if not getattr(config, "use_subtask_planning", False):
            return None
        planner = LLMSubtaskPlanner(
            llm=self._llm,  # reuse the same vLLM-backed LLM
            replan_every=getattr(config, "subtask_replan_every", 1),
            observation_chars=getattr(config, "subtask_observation_chars", 600),
        )
        logger.info(
            f"[MACLA] subtask planner enabled "
            f"(replan_every={planner._replan_every}, "
            f"observation_chars={planner._observation_chars})"
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
        # 1. Provide feedback on previous execution
        update_info = self._provide_feedback(self._prev_state_str, cur_state_str)

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
        return self._adapter.DEFAULT_ACTION

    def _get_task_description(self, game_info: dict) -> str:
        return game_info.get("task_description", self._adapter.DEFAULT_GOAL)

    def _get_default_goal(self) -> str:
        return self._adapter.DEFAULT_GOAL

    def _get_default_action(self) -> str:
        return self._adapter.DEFAULT_ACTION

    def _base_fallback(self, goal: str, observation: str, **kwargs) -> tuple[list[str], str]:
        """LLM fallback using game adapter prompts + structured output."""
        obs_image = kwargs.get("obs_image")

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

        # Stage C: prepend retrieved memories to the user prompt when the
        # vector-memory provider is active. Query is the goal + a slice of
        # the current observation — enough signal for cosine retrieval.
        if self._memory_provider is not None:
            query = f"{goal} | {observation[:300]}"
            recalled = self._memory_provider.prefetch(query)
            if recalled:
                user_text = f"[Recalled memories from prior steps]\n{recalled}\n\n{user_text}"

        # Stage D: ask the subtask planner for a near-term sub-goal and
        # prepend it. For long-horizon games (pokemon) this is the missing
        # decomposition between the overall goal and the per-step action.
        if self._subtask_planner is not None:
            try:
                history_str = self._build_subtask_history()
                subtask = self._subtask_planner.plan(
                    goal=goal,
                    observation=observation,
                    history=history_str,
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
        if self._memory_provider is not None:
            stats = self._memory_provider.stats()
            logger.info(f"[MACLA] vector memory ep={episode} stats: {stats}")
            self._memory_provider.on_session_end()
        if self._subtask_planner is not None:
            stats = self._subtask_planner.stats()
            logger.info(f"[MACLA] subtask planner ep={episode} stats: {stats}")
