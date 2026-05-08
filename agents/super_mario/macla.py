import re
from collections import defaultdict
from typing import Any

import weave
from loguru import logger

from agents.macla.base import BaseMaclaAgent
from agents.super_mario.base import SuperMarioAgent
from config.agent_config import GeminiConfig, OpenAIConfig
from config.base import WandbConfig


class MarioMaclaAgent(BaseMaclaAgent, SuperMarioAgent):
    config: GeminiConfig | OpenAIConfig

    @property
    def AGENT_TAGS(self):
        tags = ["super_mario", "macla"]
        if isinstance(self.config, GeminiConfig):
            tags.extend(["gemini", self.config.model])
        elif isinstance(self.config, OpenAIConfig):
            tags.extend(["openai", self.config.model])
        return tags

    def _get_task_description(self, task_description: str | dict[str, str] = None) -> str:
        if isinstance(task_description, dict):
            return task_description.get(
                "task_description", "Complete stage 1-1 by moving right and avoiding enemies."
            )
        return task_description or "Complete stage 1-1 by moving right and avoiding enemies."

    def _get_default_action(self) -> str:
        return "NOOP"

    def _get_default_goal(self) -> str:
        return "Move right to reach the goal, gain points and power-ups and avoid obstacles and enemies to complete the level."

    def _validate_action(self, action: str) -> str:
        """Validate Mario action."""
        # Check if action matches "Jump Level: X" or just "X"
        match = re.search(r"Jump Level:\s*(\d)", action, re.IGNORECASE)
        if match:
            level = int(match.group(1))
            if 0 <= level <= 6:
                return f"Jump Level: {level}"

        # Try to parse just a number
        try:
            level = int(action.strip())
            if 0 <= level <= 6:
                return f"Jump Level: {level}"
        except:
            pass

        logger.warning(f"Invalid action '{action}', defaulting to 'Jump Level: 0'")
        return "Jump Level: 0"

    def __init__(
        self,
        config: GeminiConfig | OpenAIConfig = None,
        wandb_config: WandbConfig = None,
    ):
        config = config or GeminiConfig()
        wandb_config = wandb_config or WandbConfig()

        # Initialize parent classes
        SuperMarioAgent.__init__(self, config=config, wandb_config=wandb_config)

        # Initialize MACLA agent
        self._init_macla_agent()

    def _base_fallback(self, goal: str, observation: str, **kwargs) -> tuple[list[str], str]:
        """
        Mario-specific fallback that calls the base SuperMarioAgent LLM logic.
        """
        obs_image = kwargs.get("obs_image")
        try:
            # Call the parent SuperMarioAgent implementation
            # Note: SuperMarioAgent._get_action returns (action, reasoning, output_text, usage, prompt)
            action, reasoning, _, _, _ = SuperMarioAgent._get_action(
                self, task_description=goal, cur_state_str=observation, obs_image=obs_image
            )
            return [action], reasoning
        except Exception as e:
            logger.error(f"Base fallback failed: {e}")
            return ["Jump Level: 0"], "Fallback Error"

    def _extract_context(self, observation: str) -> str:
        """
        Extract detailed task-relevant context from Mario observation.
        Includes relative position information for obstacles (distance from Mario).
        """
        obs_lower = observation.lower()

        # Debug: Log raw observation snippet to verify format
        if "brick" in obs_lower or "pipe" in obs_lower:
            logger.debug(f"[Position Debug] Observation snippet: {observation[:400]}")

        # 1. Get Mario's Position first
        mario_x, mario_y = 0, 0
        mario_match = re.search(r"Position of Mario:\s*\((\d+),\s*(\d+)\)", observation)
        if mario_match:
            mario_x = int(mario_match.group(1))
            mario_y = int(mario_match.group(2))
        else:
            # Fallback if standard format missing
            x_match = re.search(r"(?:x_pos|x)[:\s=]+(\d+)", obs_lower)
            y_match = re.search(r"(?:y_pos|y)[:\s=]+(\d+)", obs_lower)
            if x_match:
                mario_x = int(x_match.group(1))
            if y_match:
                mario_y = int(y_match.group(1))

        context_parts = []

        def extract_entity_with_relative_position(keywords, label):
            """Extract entities and convert to relative distance tokens: entity_dist_xNN_yNN"""
            if isinstance(keywords, str):
                keywords = [keywords]

            found_tokens = []

            for k in keywords:
                # Match line format: "- Bricks: (x, y), (x2, y2), ..." or "- Bricks: None"
                line_pattern = rf"-\s*{k}[^\n]*?:\s*(.+?)(?:\n|$)"
                line_match = re.search(line_pattern, observation, re.IGNORECASE)

                if line_match:
                    positions_str = line_match.group(1)

                    if "None" in positions_str or "none" in positions_str:
                        continue

                    # Extract all tuples
                    tuple_pattern = r"\((\d+),\s*(\d+)(?:,\s*\d+)?\)"
                    all_matches = re.findall(tuple_pattern, positions_str)

                    for x_str, y_str in all_matches:
                        x, y = int(x_str), int(y_str)

                        # Calculate relative distance
                        dx = x - mario_x
                        _dy = y - mario_y

                        # Filter irrelevant entities (too far behind or too far ahead)
                        # Mario mostly cares about what's 0-150 units ahead
                        if dx < -20 or dx > 180:
                            continue

                        direction = "ahead" if dx >= 0 else "behind"
                        abs_dx = abs(dx)

                        if abs_dx <= 60:
                            dist_label = "near"
                        elif abs_dx <= 140:
                            dist_label = "mid"
                        else:
                            dist_label = "far"

                        token = f"{label}_{direction}_{dist_label}"
                        found_tokens.append(token)

            return found_tokens

        # Extract ALL positions for each entity converted to relative tokens
        # 1. Obstacles (with positions)
        entity = extract_entity_with_relative_position("pit", "pit")
        if entity:
            logger.debug(f"Extracted pit: {entity}")
            context_parts.extend(entity)

        entity = extract_entity_with_relative_position(
            ["monster goomba", "goomba", "enemy"], "goomba"
        )
        if entity:
            logger.debug(f"Extracted goomba: {entity}")
            context_parts.extend(entity)

        entity = extract_entity_with_relative_position(
            ["monster koopas", "koopa", "turtle"], "koopa"
        )
        if entity:
            logger.debug(f"Extracted koopa: {entity}")
            context_parts.extend(entity)

        entity = extract_entity_with_relative_position(["warp pipe", "pipe"], "pipe")
        if entity:
            logger.debug(f"Extracted pipe: {entity}")
            context_parts.extend(entity)

        # 2. Opportunities (Resources with positions)
        entity = extract_entity_with_relative_position(["item mushrooms"], "mushroom")
        if entity:
            logger.debug(f"Extracted mushroom: {entity}")
            context_parts.extend(entity)

        # 3. Environment Features
        entity = extract_entity_with_relative_position(["bricks"], "brick")
        if entity:
            logger.debug(f"Extracted brick: {entity}")
            context_parts.extend(entity)

        entity = extract_entity_with_relative_position(["question blocks"], "question")
        if entity:
            logger.debug(f"Extracted question: {entity}")
            context_parts.extend(entity)

        entity = extract_entity_with_relative_position(["inactivated blocks"], "block")
        if entity:
            logger.debug(f"Extracted block: {entity}")
            context_parts.extend(entity)

        if not context_parts:
            return "clear_run"

        # Return underscore-separated sorted unique tokens
        context_str = "_".join(sorted(list(set(context_parts))))
        logger.debug(f"[Context Debug] Extracted relative context: {context_str}")
        return context_str

    def extract_preconditions(self, context_key: str, observation: str) -> list[str]:
        """
        Parse Mario-specific preconditions from context string.
        Extracts entity positions and Mario's current position.
        Converts raw relative context (dist_x...) to high-level spatial patterns (ahead_near...)
        """
        preconditions = []

        if context_key and context_key != "clear_run":
            # Context is underscore-separated entities that already have positions
            # Each entity is formatted as: entity_direction_dist_label or entity
            parts = context_key.split("_")

            i = 0
            while i < len(parts):
                # Identify entity type
                if parts[i] in ["goomba", "koopa", "pipe", "brick", "question", "pit", "mushroom"]:
                    entity = parts[i]
                    i += 1

                    # Look for spatial tokens (ahead/behind, near/mid/far)
                    direction = None
                    distance = None

                    if i < len(parts) and parts[i] in ["ahead", "behind"]:
                        direction = parts[i]
                        i += 1

                    if i < len(parts) and parts[i] in ["near", "mid", "far"]:
                        distance = parts[i]
                        i += 1

                    # Convert to spatial pattern if we found valid spatial info
                    if direction and distance:
                        pattern = f"{entity}_{direction}_{distance}"
                        preconditions.append(pattern)

                        # Add threat level for enemies
                        if entity in ["goomba", "koopa", "pit"] and direction == "ahead":
                            if distance == "near":
                                preconditions.append(f"{entity}_threat_high")
                            elif distance == "mid":
                                preconditions.append(f"{entity}_threat_medium")
                    else:
                        # Entity without position info found
                        preconditions.append(entity)

                # Check for Mario position (less critical for generalisation context, but good for specific replays)
                elif parts[i].startswith("mario"):
                    # Skip mario parts as we grab them from observation usually,
                    # but keep them if they are in context key
                    preconditions.append(parts[i])
                    i += 1
                elif parts[i] in ["clear", "path"]:
                    preconditions.append("clear_path")
                    i += 1
                else:
                    # Generic handling - just skip non-entity tokens if orphaned,
                    # or append if it looks like a meaningful token
                    if len(parts[i]) > 3:
                        preconditions.append(parts[i])
                    i += 1

        else:
            preconditions = ["clear_run"]

        # Extract Mario's position from observation if not present
        if not any(p.startswith("mario") for p in preconditions):
            try:
                # Look for Mario's position: "Position of Mario: (128, 107)"
                mario_match = re.search(r"Position of Mario:\s*\((\d+),\s*(\d+)\)", observation)
                if mario_match:
                    x_pos = mario_match.group(1)
                    y_pos = mario_match.group(2)
                    preconditions.append(f"mario_x{x_pos}_y{y_pos}")
                else:
                    # Fallback: Look for "x_pos: 100" or "x: 100"
                    x_match = re.search(r"(?:x_pos|x)[:\s=]+(\d+)", observation.lower())
                    y_match = re.search(r"(?:y_pos|y)[:\s=]+(\d+)", observation.lower())

                    if x_match:
                        preconditions.append(f"mario_x{x_match.group(1)}")
                    if y_match:
                        preconditions.append(f"mario_y{y_match.group(1)}")
            except Exception as e:
                logger.warning(f"Failed to extract Mario position for preconditions: {e}")

        return sorted(list(set(preconditions)))

    def _detect_success(
        self, execution_result: dict, prev_state_str: str, cur_state_str: str
    ) -> tuple[bool, bool]:
        """
        Mario-specific success detection based on Progress, Resource Gain, and Survival.
        """
        prev_x, cur_x = 0, 0
        is_fatal = False

        # 1. Parse State
        try:
            # Parse X Position
            p_match = re.search(r"x_pos:?\s*(\d+)", prev_state_str, re.IGNORECASE)
            c_match = re.search(r"x_pos:?\s*(\d+)", cur_state_str, re.IGNORECASE)
            if p_match:
                prev_x = int(p_match.group(1))
            if c_match:
                cur_x = int(c_match.group(1))

            # Parse Lives for Fatality Check
            p_lives_match = re.search(r"lives:?\s*(\d+)", prev_state_str, re.IGNORECASE)
            c_lives_match = re.search(r"lives:?\s*(\d+)", cur_state_str, re.IGNORECASE)
            if p_lives_match and c_lives_match:
                if int(c_lives_match.group(1)) < int(p_lives_match.group(1)):
                    is_fatal = True

            if "game over" in cur_state_str.lower():
                is_fatal = True
        except:
            pass

        if is_fatal:
            return False, True

        # 2. Check for Positive Outcomes
        # A. Positional Progress (moved right)
        progressed = cur_x > prev_x + 2

        # B. Resource Gain (Score increase implies coin collected or enemy skipped/killed)
        score_improved = False
        try:
            p_score = int(re.search(r"score:?\s*(\d+)", prev_state_str, re.IGNORECASE).group(1))
            c_score = int(re.search(r"score:?\s*(\d+)", cur_state_str, re.IGNORECASE).group(1))
            if c_score > p_score:
                score_improved = True
        except:
            pass

        # 3. Definition of Success
        # We consider it a success if we survived AND (moved forward OR gained points)
        return (progressed or score_improved), False

    def extract_postconditions(self, success_contexts):
        """
        Extract postconditions (outcomes) from Mario observation.
        Learn what specifically changes when a procedure succeeds.
        """
        if not success_contexts:
            return None

        changes = {
            "postconditions_added": set(),
            "postconditions_removed": set(),
            "action_differences": [],
        }

        # Track average progression
        start_xs = []
        end_xs = []

        for ctx in success_contexts:
            # Simple keyword extraction for now
            init_words = set(ctx.observation_init.lower().split())
            term_words = set(ctx.observation_term.lower().split())

            # Did we pick up a coin?
            if "coin" in init_words and "coin" not in term_words:
                changes["postconditions_added"].add("collected_coin")

            # Did we defeat an enemy?
            if ("goomba" in init_words or "koopa" in init_words) and (
                "goomba" not in term_words and "koopa" not in term_words
            ):
                changes["postconditions_added"].add("defeated_enemy")

            # Did we clear an obstacle?
            if "pipe" in init_words and "pipe" not in term_words:
                changes["postconditions_added"].add("cleared_pipe")

            # Extract coordinates to track movement
            try:
                # Extract Mario's x_pos
                x_pattern = r"(?:mario[^\n]{0,20})?(?:x_pos|x)[:\s=]+(\d+)"

                x_init_match = re.search(x_pattern, ctx.observation_init, re.IGNORECASE)
                x_term_match = re.search(x_pattern, ctx.observation_term, re.IGNORECASE)

                if x_init_match and x_term_match:
                    start_x = int(x_init_match.group(1))
                    end_x = int(x_term_match.group(1))
                    start_xs.append(start_x)
                    end_xs.append(end_x)

                    # Also extract Y position if available for jump height tracking
                    y_pattern = r"(?:mario[^\n]{0,20})?(?:y_pos|y)[:\s=]+(\d+)"
                    y_init_match = re.search(y_pattern, ctx.observation_init, re.IGNORECASE)
                    y_term_match = re.search(y_pattern, ctx.observation_term, re.IGNORECASE)

                    if y_init_match and y_term_match:
                        y_diff = int(y_term_match.group(1)) - int(y_init_match.group(1))
                        if abs(y_diff) > 10:  # Significant vertical movement
                            if y_diff < 0:
                                changes["postconditions_added"].add(f"jumped_height_{abs(y_diff)}")
                            elif y_diff > 0:
                                changes["postconditions_added"].add("fell")
            except Exception:
                pass

        # 2. Add Average Movement metrics
        if start_xs and end_xs:
            avg_dist = sum([e - s for s, e in zip(start_xs, end_xs)]) / len(start_xs)
            # Bin the distance to avoid too many unique postconditions
            if avg_dist > 5:
                # Add abstract distance concept (e.g. moved_forward_approx_40)
                changes["postconditions_added"].add(
                    f"moved_forward_approx_{int(avg_dist / 10) * 10}"
                )

        # Convert sets to lists
        return {k: list(v) if isinstance(v, set) else v for k, v in changes.items()}

    def extract_spatial_patterns_from_string(self, context_str: str) -> dict[str, set[str]]:
        """Helper to extract spatial patterns from a context string for matching"""
        patterns = defaultdict(set)
        if not context_str or context_str in ["clear_path", "clear_run", "general"]:
            return patterns

        parts = context_str.split("_")
        i = 0
        while i < len(parts):
            # Check if this is an entity (not a direction/distance keyword)
            if parts[i] not in ["ahead", "behind", "near", "mid", "far", "clear", "path", "dist"]:
                entity = parts[i]
                spatial_desc = []
                j = i + 1
                while j < len(parts) and parts[j] in ["ahead", "behind", "near", "mid", "far"]:
                    spatial_desc.append(parts[j])
                    j += 1
                if spatial_desc:
                    config = "_".join(spatial_desc)
                    patterns[entity].add(config)

                    # Add Mario-specific threat assessment (logic moved here)
                    if entity in ["goomba", "koopa", "pit"]:
                        if "ahead_near" in config:
                            patterns[f"{entity}_threat"].add("high")
                        elif "ahead_mid" in config:
                            patterns[f"{entity}_threat"].add("medium")
                        elif "ahead_far" in config:
                            patterns[f"{entity}_threat"].add("low")
                else:
                    patterns[entity].add("present")
                i = j
            else:
                i += 1
        return dict(patterns)

    def extract_spatial_patterns(self, contexts):
        """
        Mario-specific spatial pattern extraction for contrastive learning.
        extracts entity types, directions, distances, and Mario-specific threat levels.
        """
        patterns = defaultdict(set)

        for ctx in contexts:
            ctx_patterns = self.extract_spatial_patterns_from_string(ctx.context)
            for entity, configs in ctx_patterns.items():
                patterns[entity].update(configs)

        return dict(patterns)

    @weave.op()
    def _get_action(
        self, task_description: str, cur_state_str: str, obs_image: Any = None
    ) -> tuple[str, str, str, Any, str, str, str, dict]:
        """Get action from LLM using MACLA. Tracked by Weave for observability."""
        # 1. Provide feedback on previous execution
        update_info = self._provide_feedback(self._prev_state_str, cur_state_str, obs_image)

        # 2. Execute task using MACLA
        action, execution_result, memory_stats = self._execute_task(
            task_description, cur_state_str, obs_image
        )

        # 3. Log MACLA stats
        self._log_stats(update_info, memory_stats)

        # 4. Log procedure-related images if a procedure was executed
        # Note: Weave automatically captures inputs and outputs of @weave.op() functions.
        # If manual logging is needed, use wandb.log if wandb is initialized.

        # 5. Validate action
        action = self._validate_action(action)

        # 6. Build output
        avg_exec_time = (
            sum(self._execution_times) / len(self._execution_times) if self._execution_times else 0
        )
        reasoning = f"MACLA Strategy: {execution_result.get('method', 'unknown')}. Confidence: {execution_result.get('confidence', 0.0)}. Avg Time (last 10): {avg_exec_time:.4f}s"
        output_text = f"Result: {execution_result}\nStats: {memory_stats}"
        goal = self._get_task_description({"task_description": task_description})

        # Mario doesn't have sophisticated phase detection yet
        game_phase = "PLAYING"

        return (
            action,
            reasoning,
            output_text,
            memory_stats,
            f"Goal: {goal}\nObs: {cur_state_str}",
            game_phase,
            self._last_update_type,
            update_info,
        )

    def record_episode_end(self, episode, game_name, seed, score):
        """Called by runner at the end of an episode."""
        SuperMarioAgent.record_episode_end(self, episode, game_name, seed, score)
        self._record_episode_end(episode, score)
