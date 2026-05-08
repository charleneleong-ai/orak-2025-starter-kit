import json
import os
import traceback
from typing import Any, ClassVar

import wandb
import weave
from loguru import logger
from pydantic import PrivateAttr

from agents._harness import (
    StepRecord,
    TrajectoryWriter,
    extract_cache_stats,
)
from config.agent_config import AgentConfig
from config.base import WandbConfig

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


def _resolve_weave_project(wandb_config: WandbConfig, wandb_run) -> str:
    """Build the ``entity/project`` string that runner.py feeds back into
    ``with_weave_client``.

    Why: ``runner.py`` rpartitions this string on ``'/'`` to set the per-
    act() weave client context. A bare project name (no slash) resolves
    to ``entity=None`` server-side and trace.wandb.ai answers ``403
    Forbidden / Project not found`` — once per agent step. A 300-step
    pokemon run produced 4,507 of these in the log. The wandb run wandb
    just created already has the resolved entity (from ``~/.netrc``);
    fall back to that when ``WANDB_ENTITY`` was not set explicitly.
    """
    entity = wandb_config.entity
    if not entity and wandb_run is not None:
        entity = getattr(wandb_run, "entity", None)
    if entity:
        return f"{entity}/{wandb_config.project}"
    return wandb_config.project


class BaseOrakAgent(weave.Model):
    TRACK: ClassVar[str] = "TRACK1"

    config: AgentConfig
    wandb_config: WandbConfig

    _prev_state_str: str = PrivateAttr(default="N/A")
    _last_action: str = PrivateAttr(default="No action yet")
    _step_count: int = PrivateAttr(default=0)
    _last_score: int = PrivateAttr(default=0)

    # Stats tracking
    _stats: dict[str, int] = PrivateAttr(
        default_factory=lambda: {
            "total_inference_calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
        }
    )
    _requests_log_path: str | None = PrivateAttr(default=None)
    _trajectory_writer: TrajectoryWriter | None = PrivateAttr(default=None)
    _cached_tokens_total: int = PrivateAttr(default=0)
    _pending_fallback: dict[str, Any] | None = PrivateAttr(default=None)

    # Per-episode stats
    _episode_stats: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _current_episode_stats: dict[str, int] = PrivateAttr(
        default_factory=lambda: {
            "inference_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tokens": 0,
        }
    )
    # Action distribution tracking
    _action_counts: dict[str, int] = PrivateAttr(default_factory=lambda: {})

    def __init__(self, config: AgentConfig = None, wandb_config: WandbConfig = None):
        super().__init__(config=config, wandb_config=wandb_config)
        self._wandb_run = None

        if self.wandb_config and self.wandb_config.enabled:
            # Ensure tags is a list
            tags = list(self.wandb_config.tags) if self.wandb_config.tags else []

            # Add agent specific tags if available
            if hasattr(self, "AGENT_TAGS"):
                tags.extend(self.AGENT_TAGS)

            # Use project-specific run_id to avoid cross-game collisions
            run_id = self.wandb_config.run_id
            if run_id and self.wandb_config.project:
                run_id = f"{run_id}_{self.wandb_config.project}"
            self._wandb_run = wandb.init(
                project=self.wandb_config.project,
                entity=self.wandb_config.entity,
                id=run_id,
                resume="allow",
                reinit="create_new",
                config=self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                tags=tags,
                notes=self.wandb_config.notes,
                name=self.wandb_config.run_id,
            )
            # Initialize weave for this game's project and store client for act() switching.
            # See _resolve_weave_project for why the entity prefix is mandatory.
            self._weave_project = _resolve_weave_project(self.wandb_config, self._wandb_run)
            try:
                self._weave_client = weave.init(self._weave_project)
                if self._weave_client and self._wandb_run:
                    self._weave_client.set_wandb_run_context(run_id=self._wandb_run.id)
            except Exception:
                self._weave_client = None

    def set_log_dir(self, log_dir: str):
        """Set directory for logging raw requests + structured trajectory."""
        os.makedirs(log_dir, exist_ok=True)
        self._requests_log_path = os.path.join(log_dir, "raw_requests.jsonl")
        model_name = getattr(self.config, "model", "unknown") if self.config else "unknown"
        self._trajectory_writer = TrajectoryWriter(log_dir, model=model_name)

    def get_model_declaration(self) -> dict[str, Any]:
        """Return model declaration."""
        return {
            "name": self.config.model,
            "version": "unknown",
            "provider": self.AGENT_TAGS[0]
            if hasattr(self, "AGENT_TAGS") and self.AGENT_TAGS
            else "unknown",
            "parameter_count": "unknown",
        }

    def get_evaluation_summary(self, episodes: int) -> dict[str, Any]:
        """Return evaluation summary."""
        return {
            "total_inference_calls": self._stats["total_inference_calls"],
            "total_tokens": self._stats["total_tokens"],
            "evaluation_episodes": episodes,
            "mean_calls_per_episode": self._stats["total_inference_calls"] / episodes
            if episodes > 0
            else 0,
            "mean_tokens_per_episode": self._stats["total_tokens"] / episodes
            if episodes > 0
            else 0,
            "episodes": self._episode_stats,
        }

    def record_episode_end(self, episode_id: int, game_name: str, seed: Any, final_score: float):
        """Record stats for a completed episode."""
        if self._trajectory_writer is not None:
            try:
                self._trajectory_writer.flush_episode(
                    episode_id=episode_id,
                    completed=True,
                    final_score=float(final_score),
                    game_name=game_name,
                )
            except Exception as e:
                logger.warning(f"Failed to flush trajectory at episode end: {e}")

        self._episode_stats.append(
            {
                "episode_id": episode_id,
                "game_name": game_name,
                "seed": seed,
                "inference_calls": self._current_episode_stats["inference_calls"],
                "tokens": self._current_episode_stats["tokens"],
                "final_score": final_score,
            }
        )

        # Log episode summary to wandb
        if self.wandb_config and self.wandb_config.enabled:
            num_episodes = len(self._episode_stats)
            total_score = sum(e["final_score"] for e in self._episode_stats)
            mean_score = total_score / num_episodes if num_episodes > 0 else 0
            max_score = (
                max([e["final_score"] for e in self._episode_stats]) if num_episodes > 0 else 0
            )
            self._wandb_run.log(
                {
                    # Episode stats (renamed from episode/ to episode_end/)
                    "episode_end/id": episode_id,
                    "episode_end/final_score": final_score,
                    "episode_end/inference_calls": self._current_episode_stats["inference_calls"],
                    "episode_end/tokens": self._current_episode_stats["tokens"],
                    "episode_end/input_tokens": self._current_episode_stats["input_tokens"],
                    "episode_end/output_tokens": self._current_episode_stats["output_tokens"],
                    "agg/total_inference_calls": self._stats["total_inference_calls"],
                    "agg/total_tokens": self._stats["total_tokens"],
                    "agg/total_input_tokens": self._stats["total_input_tokens"],
                    "agg/total_output_tokens": self._stats["total_output_tokens"],
                    "agg/mean_calls_per_episode": self._stats["total_inference_calls"]
                    / num_episodes
                    if num_episodes > 0
                    else 0,
                    "agg/mean_tokens_per_episode": self._stats["total_tokens"] / num_episodes
                    if num_episodes > 0
                    else 0,
                    "agg/mean_score": mean_score,
                    "agg/max_score": max_score,
                    "agg/total_episodes": num_episodes,
                },
                step=self._step_count,
            )

        # Reset current episode stats
        self._current_episode_stats = {
            "inference_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tokens": 0,
        }

        # Reset step count for next episode
        self._step_count = 0

    @weave.op()
    def act(self, obs: dict[str, Any], step: int = None) -> dict[str, Any]:
        """Main action method tracked by Weave."""
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")
        obs_image = obs.get("obs_image", None)

        current_score = int(float(game_info.get("score", 0)))
        if step is not None:
            self._step_count = step
        else:
            self._step_count += 1

        # Get action from subclass
        action, log_extras = self.get_action(obs)

        # Update stats
        if not log_extras or log_extras.get("inference_called", True):
            self._stats["total_inference_calls"] += 1
            self._current_episode_stats["inference_calls"] += 1

        if log_extras:
            tokens_prompt = log_extras.get("tokens_prompt", 0)
            tokens_completion = log_extras.get("tokens_completion", 0)
            tokens_total = log_extras.get("tokens_total", 0)

            # If total is not provided but parts are
            if tokens_total == 0 and (tokens_prompt > 0 or tokens_completion > 0):
                tokens_total = tokens_prompt + tokens_completion

            self._stats["total_input_tokens"] += tokens_prompt
            self._stats["total_output_tokens"] += tokens_completion
            self._stats["total_tokens"] += tokens_total

            self._current_episode_stats["input_tokens"] += tokens_prompt
            self._current_episode_stats["output_tokens"] += tokens_completion
            self._current_episode_stats["tokens"] += tokens_total

        if self._requests_log_path and log_extras and "user_prompt" in log_extras:
            try:
                with open(self._requests_log_path, "a", encoding="utf-8") as f:
                    record = {
                        "step": self._step_count,
                        "prompt": log_extras["user_prompt"],
                        "response": log_extras.get("output_text", ""),
                        "action": action,
                        "tokens": {
                            "prompt": log_extras.get("tokens_prompt", 0),
                            "completion": log_extras.get("tokens_completion", 0),
                            "total": log_extras.get("tokens_total", 0),
                            "cached": log_extras.get("tokens_cached", 0),
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Failed to log raw request: {e}")
                raise ValueError(f"Failed to log raw request: {traceback.format_exc()}")

        if self._trajectory_writer is not None and log_extras and "user_prompt" in log_extras:
            try:
                self._trajectory_writer.add_step(
                    StepRecord(
                        step=self._step_count,
                        system_prompt=log_extras.get("system_prompt"),
                        user_prompt=log_extras.get("user_prompt", ""),
                        assistant_output=log_extras.get("output_text", ""),
                        action=action,
                        reasoning=log_extras.get("reasoning", "") or "",
                        tokens_prompt=log_extras.get("tokens_prompt", 0),
                        tokens_completion=log_extras.get("tokens_completion", 0),
                        tokens_total=log_extras.get("tokens_total", 0),
                        cached_tokens=log_extras.get("tokens_cached", 0),
                        is_fallback=bool(log_extras.get("is_fallback", False)),
                        fallback_reason=log_extras.get("fallback_reason"),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to record trajectory step: {e}")

        if self.wandb_config and self.wandb_config.enabled:
            log_data = {
                "step": self._step_count,
                "score": current_score,
                "score_delta": current_score - self._last_score,
                "action": action,
            }

            # Add game_phase if available
            if hasattr(self, "_game_phase"):
                log_data["game_phase"] = self._game_phase

            # Add update_type if available
            if hasattr(self, "_last_update_type"):
                log_data["update_type"] = self._last_update_type

            # Add _method_used if available
            if hasattr(self, "_method_used"):
                log_data["method_used"] = self._method_used

            # Add game specific metrics from game_info
            # We can log everything in game_info that is a number
            for k, v in game_info.items():
                if isinstance(v, (int, float)):
                    log_data[f"game_info/{k}"] = v

            # Add custom metrics from subclass
            custom_metrics = self.calculate_metrics(game_info)
            log_data.update(custom_metrics)

            # Add extras from get_action
            if log_extras:
                # Filter out output_text from wandb log to avoid clutter if they are huge
                # But keep tokens and reasoning length
                for k, v in log_extras.items():
                    if k == "reasoning":
                        # Wrap reasoning in pre-wrap for better readability in wandb
                        log_data[k] = wandb.Html(f"<div style='white-space: pre-wrap;'>{v}</div>")
                    elif k == "user_prompt":
                        # Log prompt as HTML for readability
                        log_data[k] = wandb.Html(f"<pre style='white-space: pre-wrap;'>{v}</pre>")
                    elif k not in ["output_text"]:
                        if PILImage and isinstance(v, PILImage.Image):
                            log_data[k] = wandb.Image(v)
                        else:
                            log_data[k] = v

            # Log action distribution with simplified action name
            # Subclasses can provide "simplified_action" in log_extras to override default behavior
            simplified_action = log_extras.get("simplified_action", action)

            # Simple fallback for generic function calls if not provided
            if simplified_action == action and "(" in action:
                try:
                    simplified_action = action.split("(")[0]
                except (IndexError, AttributeError):
                    pass

            # Log action distributio
            self._action_counts[simplified_action] = (
                self._action_counts.get(simplified_action, 0) + 1
            )
            total_actions = sum(self._action_counts.values())

            for act, count in self._action_counts.items():
                log_data[f"action_history/{act}"] = count / total_actions

            episode_num = len(self._episode_stats) + 1

            # Log obs_str as text
            if cur_state_str:
                log_data["obs_str"] = wandb.Html(f"<pre>Ep: {episode_num}\n{cur_state_str}</pre>")

            # Log obs_image if available
            if obs_image is not None:
                try:
                    log_data["obs_image"] = wandb.Image(
                        obs_image, caption=f"Ep: {episode_num} | Step {self._step_count}"
                    )
                except Exception:
                    # If image logging fails, just continue
                    logger.warning(f"Warning: Could not log image: {traceback.format_exc()}")

            self._wandb_run.log(log_data, step=self._step_count)

        self._prev_state_str = cur_state_str
        self._last_action = action
        self._last_score = current_score

        # Return dict for Weave to capture metadata
        result = {"action": action}

        # Game-specific metrics (evaluation_score, score, etc.)
        metrics = self.calculate_metrics(game_info)
        result.update(metrics)

        if hasattr(self, "_game_phase"):
            result["game_phase"] = self._game_phase
        if hasattr(self, "_last_update_type"):
            result["update_type"] = self._last_update_type
        if hasattr(self, "_method_used"):
            result["method_used"] = self._method_used

        if log_extras:
            if "precondition_image" in log_extras:
                result["precondition_image"] = log_extras["precondition_image"]
            if "postcondition_image" in log_extras:
                result["postcondition_image"] = log_extras["postcondition_image"]
            if "reasoning" in log_extras:
                result["reasoning"] = log_extras["reasoning"]

        return result

    # Schema map for variable-length _get_action() tuples across game agents.
    # The harness parser uses this so subclasses can keep their existing
    # tuple shape without breaking cache-stats / fallback / trajectory plumbing.
    _GET_ACTION_TUPLE_SCHEMAS: ClassVar[dict[int, list[str]]] = {
        # SuperMarioAgent — historical 5-tuple (no current_goal)
        5: ["action", "reasoning", "output_text", "usage", "prompt"],
        # PokemonRedAgent / StarCraftAgent / BaseOrakAgent — canonical 6-tuple
        6: ["action", "reasoning", "current_goal", "output_text", "usage", "prompt"],
        # TwentyFourtyEightAgent — 7-tuple with phase + update_type
        7: ["action", "reasoning", "output_text", "usage", "prompt", "game_phase", "update_type"],
        # UnifiedMaclaAgent — 8-tuple with memory_stats (not usage) + update_info
        8: [
            "action",
            "reasoning",
            "output_text",
            "memory_stats",
            "prompt",
            "game_phase",
            "update_type",
            "update_info",
        ],
    }

    def _parse_get_action_result(self, result: Any) -> dict[str, Any]:
        """Map any known _get_action tuple/dict shape to a uniform field dict.

        Lets `BaseOrakAgent.get_action` stay tolerant of the historical
        per-game tuple variants without each subclass re-implementing
        cache-stats and fallback plumbing.
        """
        if isinstance(result, dict):
            return result
        if not isinstance(result, tuple):
            raise TypeError(f"_get_action must return tuple or dict, got {type(result).__name__}")
        schema = self._GET_ACTION_TUPLE_SCHEMAS.get(len(result))
        if schema is None:
            raise ValueError(
                f"Unknown _get_action tuple length {len(result)}. "
                f"Register the shape in BaseOrakAgent._GET_ACTION_TUPLE_SCHEMAS."
            )
        parsed = dict(zip(schema, result))
        # MACLA's 8-tuple replaces `usage` with `memory_stats` — fall back to None
        # so the cache-stats path no-ops gracefully.
        parsed.setdefault("usage", None)
        return parsed

    def get_action(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Get action from LLM.
        Common implementation that delegates to _get_action.
        """
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")
        obs_image = obs.get("obs_image", None)

        task_description = game_info.get("task_description", "")

        result = self._get_action(
            task_description=task_description,
            cur_state_str=cur_state_str,
            obs_image=obs_image,
        )
        parsed = self._parse_get_action_result(result)

        action = parsed["action"]
        usage = parsed.get("usage")

        log_extras: dict[str, Any] = {}
        if parsed.get("prompt"):
            log_extras["user_prompt"] = parsed["prompt"]
        if parsed.get("output_text"):
            log_extras["output_text"] = parsed["output_text"]
        if parsed.get("reasoning"):
            current_goal = parsed.get("current_goal") or ""
            log_extras["reasoning"] = (
                f"Action: {action}\n\nGoal: {current_goal}\n\n{parsed['reasoning']}"
            )
            log_extras["reasoning_length"] = len(parsed["reasoning"])
        if parsed.get("game_phase"):
            log_extras["game_phase"] = parsed["game_phase"]
            self._game_phase = parsed["game_phase"]
        if parsed.get("update_type"):
            log_extras["update_type"] = parsed["update_type"]
            self._last_update_type = parsed["update_type"]

        if usage is not None:
            if hasattr(usage, "prompt_tokens"):
                log_extras["tokens_prompt"] = usage.prompt_tokens
                log_extras["tokens_completion"] = usage.completion_tokens
                log_extras["tokens_total"] = usage.total_tokens
            elif isinstance(usage, dict):
                log_extras.update(usage)

        self._postprocess_log_extras(log_extras, usage)
        return action, log_extras

    def _postprocess_log_extras(self, log_extras: dict[str, Any], usage: Any) -> None:
        """Add cache stats + fallback flag + prompt/completion tokens to log_extras.

        Reusable from `BaseMaclaAgent.get_action` and any other override that
        builds log_extras manually — keeps cache/fallback/token plumbing in
        one place. Mutates ``log_extras`` in place.

        ``usage`` may be a LangChain UsageMetadata dict, an OpenAI usage
        object with ``.prompt_tokens`` attrs, or the normalised dict
        produced by ``safe_structured_invoke._extract_usage`` (which
        already uses ``tokens_prompt`` / ``tokens_completion`` /
        ``tokens_total`` keys). Without this, MACLA's get_action path
        only logged cached tokens and the per-step record showed
        ``"tokens": {"prompt": 0, "completion": 0, "total": 0}`` for
        every LLM call.
        """
        cache_stats = extract_cache_stats(usage)
        if cache_stats["cached_tokens"]:
            log_extras["tokens_cached"] = cache_stats["cached_tokens"]
            self._cached_tokens_total += cache_stats["cached_tokens"]
            log_extras["cached_tokens_total"] = self._cached_tokens_total

        if usage is not None and not log_extras.get("tokens_total"):
            if isinstance(usage, dict):
                # Dict-shaped usage from safe_structured_invoke's normaliser
                # already uses the canonical keys.
                for key in ("tokens_prompt", "tokens_completion", "tokens_total"):
                    val = usage.get(key)
                    if val:
                        log_extras[key] = val
            elif hasattr(usage, "prompt_tokens"):
                # OpenAI-shaped usage object (Pydantic). Mirror the
                # surfacing block in BaseOrakAgent.get_action so MACLA
                # and non-MACLA paths produce identical telemetry.
                log_extras["tokens_prompt"] = usage.prompt_tokens
                log_extras["tokens_completion"] = usage.completion_tokens
                log_extras["tokens_total"] = usage.total_tokens

        if self._pending_fallback is not None:
            log_extras["is_fallback"] = True
            log_extras["fallback_reason"] = self._pending_fallback.get("reason", "")
            self._pending_fallback = None

    def _mark_fallback(self, reason: str) -> None:
        """Subclasses call this when their _get_action falls back to a default
        action because of an LLM error. Surfaces in trajectory + log_extras."""
        self._pending_fallback = {"reason": reason}

    def _get_action(
        self, task_description: str, cur_state_str: str, obs_image: Any = None
    ) -> tuple[str, str, str, str, Any, str]:
        """
        Get action from LLM.
        This method should be overridden by subclasses and often decorated with @weave.op().
        Returns: (action, reasoning, current_goal, output_text, usage, prompt)
        """
        raise NotImplementedError

    def calculate_metrics(self, game_info: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate custom metrics based on game info.
        Override this in subclasses.
        """
        return {}

    def __del__(self):
        """Cleanup wandb on agent destruction."""
        if hasattr(self, "wandb_config") and self.wandb_config and self.wandb_config.enabled:
            try:
                wandb.finish()
            except:
                pass

    # ===== Checkpoint Methods =====

    def get_state(self) -> dict[str, Any]:
        """
        Get agent state for checkpointing.

        Subclasses should override this to add their specific state,
        calling super().get_state() first to include base state.

        Returns:
            Dictionary containing agent state.
        """
        return {
            "stats": self._stats,
            "episode_stats": self._episode_stats,
            "current_episode_stats": self._current_episode_stats,
            "step_count": self._step_count,
            "last_score": self._last_score,
            "prev_state_str": self._prev_state_str,
            "last_action": self._last_action,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load agent state from checkpoint.

        Subclasses should override this to restore their specific state,
        calling super().load_state(state) first to restore base state.

        Args:
            state: State dictionary from get_state()
        """
        self._stats = state.get("stats", self._stats)
        self._episode_stats = state.get("episode_stats", [])
        self._current_episode_stats = state.get(
            "current_episode_stats",
            {"inference_calls": 0, "input_tokens": 0, "output_tokens": 0, "tokens": 0},
        )
        self._step_count = state.get("step_count", 0)
        self._last_score = state.get("last_score", 0)
        self._prev_state_str = state.get("prev_state_str", "N/A")
        self._last_action = state.get("last_action", "No action yet")

        logger.info(
            f"Loaded agent state: {self._step_count} steps, {len(self._episode_stats)} episodes"
        )

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """
        Get metadata for checkpoint summary.

        Subclasses can override to add their own metadata,
        calling super().get_checkpoint_metadata() to include base metadata.

        Returns:
            Metadata dictionary with summary information.
        """
        return {
            "agent_class": self.__class__.__name__,
            "total_steps": self._step_count,
            "total_episodes": len(self._episode_stats),
            "total_inference_calls": self._stats.get("total_inference_calls", 0),
            "total_tokens": self._stats.get("total_tokens", 0),
        }
