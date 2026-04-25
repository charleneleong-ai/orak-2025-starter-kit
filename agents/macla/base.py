"""
Base MACLA Agent - Game-agnostic implementation of the MACLA framework.

This module provides a base class that implements the core MACLA (Memory-Augmented 
Contrastive Learning Agent) logic while allowing game-specific implementations to 
customise behavior through abstract methods.
"""
import time
from abc import abstractmethod
from collections import deque
from typing import Any, Optional
from pydantic import PrivateAttr
import wandb
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from loguru import logger
from config.agent_config import GeminiConfig, LocalConfig, OpenAIConfig
from pydantic import BaseModel
from agents.macla.macla_lib import LLMMACLAAgent


class BaseMaclaAgent(BaseModel):
    """
    Base MACLA agent providing core memory-augmented learning functionality.
    
    Game-specific agents should inherit from both this class and their game's 
    base agent class, implementing the abstract methods to customize behavior.
    """
    
    # MACLA-specific private attributes
    _macla_agent: LLMMACLAAgent = PrivateAttr()
    _last_execution_result: Optional[dict] = PrivateAttr(default=None)
    _macla_step_count: int = PrivateAttr(default=0)
    _learning_step_interval: int = PrivateAttr(default=10)
    _execution_times: deque = PrivateAttr(default_factory=lambda: deque(maxlen=10))
    _episode_lengths: deque = PrivateAttr(default_factory=lambda: deque(maxlen=20))
    _steps_in_current_episode: int = PrivateAttr(default=0)
    _update_counters: dict = PrivateAttr(default_factory=lambda: {
        "atomic_entry": 0, 
        "procedure_learned": 0, 
        "procedure_updated": 0
    })
    _method_counters: dict = PrivateAttr(default_factory=lambda: {})
    _last_update_type: str = PrivateAttr(default="atomic_entry")
    _llm_reasoning: str = PrivateAttr(default="")  # Store LLM reasoning from fallback
    
    def _init_macla_agent(self):
        """Initialize the MACLA agent based on config type."""
        if isinstance(self.config, LocalConfig):
            self._init_local_macla()
        elif isinstance(self.config, GeminiConfig):
            self._init_gemini_macla()
        elif isinstance(self.config, OpenAIConfig):
            self._init_openai_macla()
        else:
            logger.warning(f"Unsupported config type: {type(self.config)}. Defaulting to Gemini initialisation.")
            self._init_gemini_macla()
    
    def _init_gemini_macla(self):
        """Initialize MACLA with Gemini/Vertex AI."""
        self._llm = ChatVertexAI(
            model_name=self.config.model,
            temperature=self.config.temperature,
            project=self.config.gcp_project,
            location=self.config.gcp_location,
        )
        self._macla_agent = LLMMACLAAgent(
            generator=self._llm, 
            fallback_generator=self._base_fallback,
            context_extractor=self._extract_context,
            postcondition_extractor=self.extract_postconditions,
            spatial_pattern_extractor=getattr(self, 'extract_spatial_patterns', None),
            precondition_extractor=self.extract_preconditions
        )
    
    def _init_openai_macla(self):
        """Initialize MACLA with OpenAI."""
        model_lower = self.config.model.lower()
        is_reasoning_model = any(keyword in model_lower for keyword in ['o1', 'o3', 'gpt-5'])
        
        temperature = self.config.temperature
        if is_reasoning_model:
            temperature = 1.0  # Default for reasoning models
             
        self._llm = ChatOpenAI(
            model=self.config.model,
            api_key=self.config.api_key,
            temperature=temperature,
        )
        self._macla_agent = LLMMACLAAgent(
            generator=self._llm, 
            fallback_generator=self._base_fallback,
            context_extractor=self._extract_context,
            postcondition_extractor=self.extract_postconditions,
            spatial_pattern_extractor=getattr(self, 'extract_spatial_patterns', None),
            precondition_extractor=self.extract_preconditions
        )

    def _init_local_macla(self):
        """Initialize MACLA with a local model via OpenAI-compatible API (vLLM, Ollama, MLX)."""
        extra_body = self.config.extra_body or {}
        logger.info(
            f"Initializing local model: {self.config.model} "
            f"via {self.config.server_type} at {self.config.base_url} "
            f"(vision={self.config.supports_vision}, extra_body={extra_body})"
        )
        self._supports_vision = self.config.supports_vision
        llm_kwargs = dict(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if extra_body:
            llm_kwargs["extra_body"] = extra_body
        self._llm = ChatOpenAI(**llm_kwargs)
        self._macla_agent = LLMMACLAAgent(
            generator=self._llm,
            fallback_generator=self._base_fallback,
            context_extractor=self._extract_context,
            postcondition_extractor=self.extract_postconditions,
            spatial_pattern_extractor=getattr(self, 'extract_spatial_patterns', None),
            precondition_extractor=self.extract_preconditions,
        )

    @abstractmethod
    def _extract_context(self, observation: str) -> str:
        """
        Extract game-specific context from observation.
        
        Args:
            observation: String representation of game state
            
        Returns:
            Context key string used for procedure matching
        """
        pass
    
    @abstractmethod
    def extract_preconditions(self, context_key: str, observation: str) -> list[str]:
        """
        Extract preconditions from context and observation.
        
        Args:
            context_key: The context key string
            observation: The raw observation string
            
        Returns:
            List of precondition strings
        """
        pass

    @abstractmethod
    def extract_postconditions(self, success_contexts):
        """
        Extract game-specific postconditions from successful executions.
        
        Args:
            success_contexts: List of ContrastiveContext objects
            
        Returns:
            dict with postcondition updates or None for default behavior
        """
        pass
    
    @abstractmethod
    def _detect_success(self, execution_result: dict, prev_state_str: str, cur_state_str: str) -> tuple[bool, bool]:
        """
        Determine if an action was successful in game-specific terms.
        
        Args:
            execution_result: The MACLA execution result
            prev_state_str: Previous game state string
            cur_state_str: Current game state string
            
        Returns:
            Tuple of (success, is_fatal):
                - success: Whether the action was beneficial
                - is_fatal: Whether the action ended the game
        """
        pass
    
    @abstractmethod
    def _base_fallback(self, goal: str, observation: str, **kwargs) -> list[str] | tuple[list[str], str]:
        """
        Generate fallback actions when no procedure matches.
        
        Args:
            goal: The current goal/task
            observation: Current observation string
            **kwargs: Additional context (e.g., obs_image)
            
        Returns:
            List of action strings or tuple (actions, reasoning)
        """
        pass
    
    @abstractmethod
    def _validate_action(self, action: str) -> str:
        """
        Validate and potentially correct an action.
        
        Args:
            action: The action to validate
            
        Returns:
            Valid action string
        """
        pass
    
    @abstractmethod
    def _get_task_description(self, game_info: dict) -> str:
        """
        Get task description for the current game.
        
        Args:
            game_info: Dictionary containing game information
            
        Returns:
            Task description string to use as goal
        """
        pass
    
    def _provide_feedback(self, prev_state_str: str, cur_state_str: str, cur_obs_image: Any = None) -> Optional[dict]:
        """
        Provide feedback to MACLA agent on previous execution.
        
        Args:
            prev_state_str: Previous observation text
            cur_state_str: Current observation text
            cur_obs_image: Current observation image (PIL.Image)
        
        Returns:
            Update info dict or None if no previous execution
        """
        if not self._last_execution_result or self._last_action == "No action yet":
            return None
        
        # Detect success using game-specific logic
        strong_success, is_fatal_game_over = self._detect_success(
            self._last_execution_result, 
            prev_state_str, 
            cur_state_str
        )
        logger.debug(f"Providing feedback to MACLA: strong_success={strong_success}, is_fatal_game_over={is_fatal_game_over} Image: {cur_obs_image}")
        update_info = self._macla_agent.provide_feedback(
            self._last_execution_result, 
            strong_success, 
            next_observation=cur_state_str,
            next_obs_image=cur_obs_image,
            is_fatal=is_fatal_game_over
        )
        
        # Track update type
        self._update_counters[update_info["type"]] += 1
        self._last_update_type = update_info["type"]
        self._method_used = update_info["method_used"]
        
        # Log significant updates
        if update_info["type"] != "atomic_entry":
            logger.info(
                f"Step {self._macla_step_count}: {update_info['type']} "
                f"(success={update_info['was_success']}, "
                f"method={update_info['method_used']}, "
                f"pk={update_info['procedure_key']})"
            )
        
        return update_info
    
    def _execute_task(self, task_description: str, cur_state_str: str, obs_image: Any = None) -> tuple[str, dict, dict]:
        """
        Execute task using MACLA and return action with metadata.
        
        Returns:
            Tuple of (action, execution_result, memory_stats)
        """
        # Augment observation with action history for context
        augmented_obs = f"LastAction: {self._last_action}\n{cur_state_str}"
        
        goal = task_description if task_description else self._get_default_goal()
        
        # Execute task
        start_time = time.time()
        execution_result = self._macla_agent.execute_task(augmented_obs, goal, obs_image=obs_image)
        duration = time.time() - start_time
        
        self._execution_times.append(duration)
        avg_exec_time = sum(self._execution_times) / len(self._execution_times)
        
        self._last_execution_result = execution_result
        
        # Extract action
        action_sequence = execution_result.get("action_sequence", [])
        action = action_sequence[0] if action_sequence else self._get_default_action()
        
        self._macla_step_count += 1
        self._steps_in_current_episode += 1
        
        # Collect memory stats periodically
        memory_stats = {}
        if self._macla_step_count > 0 and self._macla_step_count % self._learning_step_interval == 0:
            optimisation_stats = self._macla_agent.run_optimisation_cycle()
            memory_stats = self._macla_agent.get_detailed_memory_stats()
            memory_stats.setdefault("optimisation", {}).update(optimisation_stats)
            memory_stats["avg_exec_time_last_10"] = avg_exec_time
            logger.info(f"MACLA Stats & Optimisation (Step {self._macla_step_count}): {memory_stats}")
        
        # Track inference usage
        method = execution_result.get('method', 'unknown')
        self._method_counters[method] = self._method_counters.get(method, 0) + 1
        
        memory_stats["inference_called"] = (method != "bayesian_procedure")
        memory_stats["method_counts"] = self._method_counters.copy()
        
        return action, execution_result, memory_stats
    
    def _build_reasoning_with_macla_metadata(self, execution_result: dict, action: str = None, goal: str = None) -> str:
        """
        Build reasoning string combining LLM reasoning (if available) with MACLA metadata.
        
        Args:
            execution_result: The execution result dictionary from MACLA
            action: The selected action (optional)
            goal: The goal/task description (optional)
            
        Returns:
            Combined reasoning string
        """
        avg_exec_time = sum(self._execution_times) / len(self._execution_times) if self._execution_times else 0
        
        # Build MACLA metadata with all available information
        metadata_parts = [
            f"Strategy: {execution_result.get('method', 'unknown')}",
            f"Confidence: {execution_result.get('confidence', 0.0):.3f}",
            f"Avg Time: {avg_exec_time:.4f}s"
        ]
        
        if action:
            metadata_parts.append(f"\n\nAction: {action}")
        
        if goal:
            metadata_parts.append(f"\n\nGoal: {goal}")
        
        macla_metadata = " | ".join(metadata_parts)
        
        # Combine LLM reasoning with MACLA metadata
        if self._llm_reasoning:
            reasoning = f"{self._llm_reasoning}\n\n[MACLA] {macla_metadata}"
            self._llm_reasoning = ""
        elif execution_result.get("reasoning"):
            # Include stored procedure reasoning when using bayesian_procedure
            reasoning = f"{execution_result['reasoning']}\n\n[MACLA] {macla_metadata}"
        else:
            reasoning = f"[MACLA] {macla_metadata}"
        
        return reasoning
    
    def _log_stats(self, update_info: Optional[dict], memory_stats: dict):
        """Log MACLA statistics to WandB."""
        if update_info:
            memory_stats["update_type"] = update_info["type"]
            memory_stats["procedure_key"] = update_info["procedure_key"]
            memory_stats["cumulative_updates"] = self._update_counters.copy()
            
            # Log to wandb periodically
            if wandb.run and self._macla_step_count % 10 == 0:
                log_data = {
                    "step": self._macla_step_count,
                    "update_type": update_info["type"],
                    "procedures_learned_total": self._update_counters["procedure_learned"],
                    "procedures_updated_total": self._update_counters["procedure_updated"],
                    "learning_rate": self._update_counters["procedure_learned"] / max(self._macla_step_count, 1),
                }
                
                # Add method counts to wandb log
                for method, count in self._method_counters.items():
                    log_data[f"method_counts/{method}"] = count
                
                # Add optimization metrics if available
                if "optimisation" in memory_stats:
                    for key, value in memory_stats["optimisation"].items():
                        if isinstance(value, (int, float)):
                            log_data[f"optimisation/{key}"] = value
                
                # Add memory sizes if available
                if "memory_sizes" in memory_stats:
                    for key, value in memory_stats["memory_sizes"].items():
                        if isinstance(value, (int, float)):
                            log_data[f"memory/{key}"] = value
                    
                if hasattr(self, '_wandb_run') and self._wandb_run:
                    self._wandb_run.log(log_data, step=self._step_count)
                else:
                    wandb.log(log_data, step=self._step_count)
    
    def _build_procedure_table(self, procedure_list: list[dict], table_type: str = "procedure") -> tuple[list, list]:
        """
        Build WandB table data for procedures or meta-procedures.
        
        Args:
            procedure_list: List of procedure dictionaries
            table_type: Type of table for logging ("procedure" or "meta_procedure")
            
        Returns:
            Tuple of (columns, data) for wandb.Table
        """
        columns = ["id", "goal", "steps", "preconditions", "postconditions", "reasoning", "success_rate", "executions", "refinements", "sample_pre_img", "sample_post_img"]
        data = []
        
        for p in procedure_list:
            # Get sample images from pre-calculated fields
            pre_img_raw = p.get("sample_pre_image")
            post_img_raw = p.get("sample_post_image")
            
            pre_img = None
            post_img = None
            
            if pre_img_raw:
                try:
                    pre_img = wandb.Image(pre_img_raw)
                except Exception as e:
                    logger.warning(f"Failed to convert {table_type} pre_img to wandb.Image: {e}")
                    
            if post_img_raw:
                try:
                    post_img = wandb.Image(post_img_raw)
                except Exception as e:
                    logger.warning(f"Failed to convert {table_type} post_img to wandb.Image: {e}")
            
            row = [
                p.get("id"),
                p.get("goal"),
                p.get("steps"),
                p.get("preconditions"),
                p.get("postconditions"),
                p.get("reasoning", ""),
                p.get("success_rate"),
                p.get("executions"),
                p.get("refinements"),
                pre_img,
                post_img
            ]
            data.append(row)
        
        return columns, data
    
    def _record_episode_end(self, episode: int, score: float):
        """Record MACLA-specific episode end statistics."""
        # Update adaptive stagnation tracking
        if self._macla_agent and hasattr(self._macla_agent, 'update_episode_score'):
            self._macla_agent.update_episode_score(score)

        # Track episode length
        self._episode_lengths.append(self._steps_in_current_episode)
        avg_steps = sum(self._episode_lengths) / len(self._episode_lengths) if self._episode_lengths else 0
        
        # Reset per-episode counters
        self._steps_in_current_episode = 0
        
        try:
            memory_stats = self._macla_agent.get_detailed_memory_stats()
            logger.info(f"MACLA End-of-Episode Stats (Episode {episode}): {memory_stats}")
            
            # Log procedures
            procedures_data = self._macla_agent.log_procedures()
            
            logger.info(f"DEBUG: procedures_data keys: {list(procedures_data.keys())}")
            if "procedures" in procedures_data:
                 logger.info(f"DEBUG: num procedures: {len(procedures_data['procedures'])}")
            
            logger.info(f"DEBUG: wandb.run is {wandb.run}")
            
            if wandb.run:
                episode_stats = {f"episode_end/{k}": v for k, v in memory_stats.items()}
                episode_stats["episode"] = episode
                episode_stats["final_score"] = score
                episode_stats["episode_length"] = self._episode_lengths[-1]
                episode_stats["avg_steps_per_episode"] = avg_steps

                # Log Procedures Table
                if procedures_data.get("procedures"):
                    proc_list = procedures_data["procedures"]
                    columns, data = self._build_procedure_table(proc_list, "procedure")
                    
                    logger.info(f"DEBUG: creating learned_procedures table with {len(data)} rows. Columns: {columns}")
                    if data:
                        logger.info(f"DEBUG: First row image check - Pre: {data[0][9] is not None}, Post: {data[0][10] is not None}")

                    episode_stats["learned_procedures"] = wandb.Table(data=data, columns=columns)
                    
                # Log Meta-Procedures Table
                if procedures_data.get("meta_procedures"):
                    meta_list = procedures_data["meta_procedures"]
                    columns, data = self._build_procedure_table(meta_list, "meta_procedure")
                    episode_stats["learned_meta_procedures"] = wandb.Table(data=data, columns=columns)

                if hasattr(self, '_wandb_run') and self._wandb_run:
                    self._wandb_run.log(episode_stats)
                else:
                    wandb.log(episode_stats)
                
        except Exception as e:
            logger.error(f"Failed to log MACLA end-of-episode stats: {e}")
    
    def get_action(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Common get_action implementation for all MACLA agents.
        This method extracts MACLA metrics and adds them to log_extras.
        
        Game-specific agents should implement _get_action() instead of overriding this.
        """
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")
        obs_image = obs.get("obs_image", None)
        
        # Call game-specific _get_action (should be implemented by subclass)
        result_tuple = self._get_action(
            task_description=game_info.get("task_description", ""),
            cur_state_str=cur_state_str,
            obs_image=obs_image
        )
        
        update_info = None
        if len(result_tuple) == 8:
            action, reasoning, output_text, memory_stats, prompt, game_phase, update_type, update_info = result_tuple
        else:
            action, reasoning, output_text, memory_stats, prompt, game_phase, update_type = result_tuple
        
        self._last_update_type = update_type
        
        log_extras = {
            "update_type": update_type,
            "output_text": output_text,
            "reasoning": reasoning,
            "user_prompt": prompt,
        }
        
        # Add images from update_info if available
        if update_info:
             if "precondition_image" in update_info:
                 log_extras["precondition_image"] = update_info["precondition_image"]
             if "postcondition_image" in update_info:
                 log_extras["postcondition_image"] = update_info["postcondition_image"]
        
        # Add MACLA-specific metrics from memory_stats
        if memory_stats:
            # Add method counts
            if "method_counts" in memory_stats:
                for method, count in memory_stats["method_counts"].items():
                    log_extras[f"macla/method_counts/{method}"] = count
            
            # Add other MACLA metrics
            if "inference_called" in memory_stats:
                log_extras["macla/inference_called"] = memory_stats["inference_called"]
            
            # Add detailed memory stats if available
            for key in ["num_procedures", "num_meta_procedures", "avg_exec_time_last_10"]:
                if key in memory_stats:
                    log_extras[f"macla/{key}"] = memory_stats[key]
            
            # Add optimization metrics if available
            if "optimisation" in memory_stats:
                opt_stats = memory_stats["optimisation"]
                for key in ["procedures_refined_this_cycle", "avg_procedure_success_rate"]:
                    if key in opt_stats:
                        log_extras[f"optimisation/{key}"] = opt_stats[key]
        
        return action, log_extras
    
    def get_state(self) -> dict[str, Any]:
        """
        Get agent state for checkpointing.
        Saves MACLA memory and stats.
        """
        # Call the next class in MRO (likely BaseOrakAgent via game agent)
        state = super().get_state()
        
        # Add MACLA memory state
        if self._macla_agent and hasattr(self._macla_agent, "memory_system"):
            # We rely on pickle to handle the deep object structure of memory_system
            state["macla_memory"] = self._macla_agent.memory_system
            state["macla_stats"] = self._macla_agent.stats
            
        return state
    
    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load agent state from checkpoint.
        Restores MACLA memory and stats.
        """
        # Call the next class in MRO
        super().load_state(state)
        
        # Restore MACLA memory state
        if "macla_memory" in state and self._macla_agent:
            self._macla_agent.memory_system = state["macla_memory"]
            # Re-link selector and meta-learner to the restored memory system
            self._macla_agent.bayesian_selector.memory_system = state["macla_memory"]
            self._macla_agent.meta_learner.memory_system = state["macla_memory"]
            logger.info("Restored MACLA memory system from checkpoint")
            
        if "macla_stats" in state and self._macla_agent:
            self._macla_agent.stats = state["macla_stats"]

    @abstractmethod
    def _get_default_goal(self) -> str:
        """Get default goal for the game."""
        pass
    
    @abstractmethod
    def _get_default_action(self) -> str:
        """Get default fallback action."""
        pass
