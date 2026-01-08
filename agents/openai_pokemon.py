"""
OpenAI Pokemon Agent

"""

import openai
import re
import json
import numpy as np
from typing import List, Dict, Optional
import time

from agents.openai_pokemon_memory_utils import (
    parse_game_state,
    get_map_memory_dict,
    replace_map_on_screen_with_full_map,
    replace_filtered_screen_text
)
from .pokemon_prompts import (
    SYSTEM_PROMPT, 
    USER_PROMPT, 
    HISTORY_SUMMARY_SYSTEM_PROMPT, 
    HISTORY_SUMMARY_USER_PROMPT, 
    SELF_REFLECTION_SYSTEM_PROMPT, 
    SELF_REFLECTION_USER_PROMPT, 
    SUBTASK_PLANNING_SYSTEM_PROMPT, 
    SUBTASK_PLANNING_USER_PROMPT
)

MODEL = "gpt-5-nano" 


class VectorMemory:
    """
    A simple vector database for storing and retrieving memories based on semantic similarity.
    Uses OpenAI's text-embedding-3-small model for fast, cost-effective embeddings.
    """
    
    def __init__(self, max_memories: int = 100):
        self.client = openai.OpenAI()
        self.memories = []
        self.max_memories = max_memories
        self.embedding_model = "text-embedding-3-small"
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            return np.zeros(1536)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def add_memory(self, content: str, metadata: Optional[Dict] = None):
        """Add a memory to the vector store"""
        if not content or content.strip() == "":
            return
        
        embedding = self._get_embedding(content)
        memory = {
            'content': content,
            'embedding': embedding,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        self.memories.append(memory)
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]
        
    
    def retrieve_similar(self, query: str, top_k: int = 3, threshold: float = 0.5) -> List[Dict]:
        """Retrieve most similar memories to a query"""
        if not self.memories or not query or query.strip() == "":
            return []
        
        query_embedding = self._get_embedding(query)
        similarities = []
        
        for memory in self.memories:
            similarity = self._cosine_similarity(query_embedding, memory['embedding'])
            if similarity >= threshold:
                similarities.append({
                    'content': memory['content'],
                    'metadata': memory['metadata'],
                    'similarity': similarity
                })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        results = similarities[:top_k]
        
        
        return results
    
    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """Format retrieved memories for LLM prompts"""
        if not memories:
            return "N/A"
        
        formatted = []
        for i, memory in enumerate(memories):
            meta_str = ""
            if memory['metadata']:
                parts = []
                if 'step' in memory['metadata']:
                    parts.append(f"Step {memory['metadata']['step']}")
                if 'map_name' in memory['metadata']:
                    parts.append(f"Map: {memory['metadata']['map_name']}")
                if parts:
                    meta_str = f" ({', '.join(parts)})"
            
            formatted.append(f"[Memory {i+1}{meta_str}] {memory['content']}")
        
        return "\n".join(formatted)


def extract_memory_entries(reflection: str) -> list:
    """Extract NewFacts list from self_reflection JSON output."""
    try:
        json_str = re.sub(r"^```json\s*|\s*```$", "", reflection.strip(), flags=re.DOTALL)
        json_str = re.sub(r"^'''json\s*|\s*'''$", "", json_str.strip(), flags=re.DOTALL)
        reflection_json = json.loads(json_str)
        return reflection_json.get("NewFacts", [])
    except:
        return []

def build_memory_query(goal: str, environment_perception: str) -> str:
    """Build a memory retrieval query from goal and environment context."""
    return f"Information related to 'Goal: {goal}' based on 'Context: {environment_perception}'"


class OpenAIPokemonAgent:
    TRACK = "TRACK1"

    def __init__(self, env=None):
        self.client = openai.OpenAI()
        self.env = env

        # Memory tracking
        self.step_count = 0
        self.histories = []
        self.max_history = 20
        self.action_buffer = []  # Track recent actions for history summary
        self.num_action_buffer = 10

        self.map_memory_dict = {}
        self.state_dict = {}
        self.prev_state_dict = {}
        self.dialog_buffer = []

        # Module outputs for tracking
        self.last_subtask = None
        self.last_subtask_reasoning = None
        self.last_module = None  # Track last executed module
        self.last_self_reflection = None  # Store last reflection for NewFacts extraction

        # Vector memory for long-term semantic retrieval
        self.vector_memory = VectorMemory(max_memories=100)
    
    def _process_observation(self, raw_obs):
        """Process raw observation to add map memory and dialog buffer"""
        try:
            # Handle both dict and string observations
            if isinstance(raw_obs, dict):
                obs_text = raw_obs.get("obs_str", str(raw_obs))
            else:
                obs_text = str(raw_obs)
            
            # Parse the game state
            self.prev_state_dict = self.state_dict.copy() if self.state_dict else {}
            self.state_dict = parse_game_state(obs_text)
            
            # Update map memory - tracks explored areas
            if self.state_dict['map_info']['map_name']:
                self.map_memory_dict = get_map_memory_dict(self.state_dict, self.map_memory_dict)
            
            # Track dialog buffer
            if self.state_dict['state'] == 'Dialog':
                dialog_text = self.state_dict.get('filtered_screen_text', '')
                if dialog_text and dialog_text != 'N/A':
                    self.dialog_buffer.append(dialog_text)
                    self.dialog_buffer = self.dialog_buffer[-5:]
            
            # Start with text observation
            processed_obs = obs_text
            
            # Replace map on screen with full explored map
            current_map = self.state_dict['map_info'].get('map_name')
            if current_map and current_map in self.map_memory_dict:
                explored_map = self.map_memory_dict[current_map]['explored_map']
                processed_obs = replace_map_on_screen_with_full_map(processed_obs, explored_map)
            
            # Add dialog buffer when returning to field state
            if self.dialog_buffer and self.state_dict['state'] == 'Field':
                processed_obs = replace_filtered_screen_text(processed_obs, self.dialog_buffer)
                self.dialog_buffer = []

            return processed_obs
                
        except Exception as e:
            if isinstance(raw_obs, dict):
                return raw_obs.get("obs_str", str(raw_obs))
            return str(raw_obs)
    
    def _build_environment_perception(self) -> str:
        """Build structured environment perception from current state."""
        cur_state = self.state_dict.get('state', 'Unknown')
        if cur_state == "Title":
            return f"State:{cur_state}"
        elif cur_state == "Field":
            map_info = self.state_dict.get('map_info', {})
            return f"State:{cur_state}, MapName:{map_info.get('map_name')}, PlayerPos:({map_info.get('player_pos_x')},{map_info.get('player_pos_y')})"
        elif cur_state == 'Dialog':
            return f"State:{cur_state}, ScreenText:{self.state_dict.get('filtered_screen_text', 'N/A')}"
        else:
            return f"State:{cur_state}, Enemy:{self.state_dict.get('enemy_pokemon', {})}, Party:{self.state_dict.get('your_party', '')}"

    def _memory_management(self, goal: str) -> str:
        """
        Memory management module: Extract NewFacts from reflection, build query, retrieve relevant memory.
        Returns formatted relevant memory string.
        """
        # 1. Extract NewFacts from last self_reflection and add to LTM
        if self.last_module == 'self_reflection' and self.last_self_reflection:
            memory_entries = extract_memory_entries(self.last_self_reflection)
            if memory_entries:
                for entry in memory_entries:
                    # Add to vector memory with deduplication via similarity check
                    existing = self.vector_memory.retrieve_similar(entry, top_k=1, threshold=0.8)
                    if not existing:  # Only add if not too similar to existing
                        self.vector_memory.add_memory(entry, metadata={
                            'step': self.step_count,
                            'map_name': self.state_dict.get('map_info', {}).get('map_name', 'Unknown'),
                            'type': 'fact'
                        })

        # 2. Build environment perception and query
        environment_perception = self._build_environment_perception()
        query = build_memory_query(goal or "Continue playing", environment_perception)

        # 3. Retrieve relevant memories
        retrieved = self.vector_memory.retrieve_similar(query, top_k=3, threshold=0.4)
        return self.vector_memory.format_memories_for_prompt(retrieved)

    def _build_raw_history(self):
        """Build raw history for history summarization module"""
        if not self.histories:
            return "No history yet (first step)"

        recent = self.histories[-10:]
        lines = []
        for h in recent:
            step_num = h.get('step', '?')
            state = h.get('state', 'Unknown')
            map_name = h.get('map_name', 'Unknown')
            action = h.get('action', 'none')

            state_msg = f"{step_num}th_state: {state}"
            if state == 'Field':
                state_msg += f" in {map_name}"
            action_msg = f"{step_num}th_action: {action}"

            lines.append(f"{{{state_msg}}}")
            lines.append(f"{{{action_msg}}}")

        return "\n".join(lines)
    
    def _parse_section(self, text, section_name):
        """Extract content after a markdown section header"""
        pattern = rf"### {section_name}\s*(.+?)(?=###|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _module_history_summarization(self):
        """Module 1: History Summarization"""

        raw_history = self._build_raw_history()
        user_prompt = HISTORY_SUMMARY_USER_PROMPT.format(short_term_history=raw_history)

        response = self.client.responses.create(
            model=MODEL,
            input=user_prompt,
            instructions=HISTORY_SUMMARY_SYSTEM_PROMPT,
            reasoning={"effort": "low"}
        )

        output = response.output_text.strip()

        summary = self._parse_section(output, "Short_term_summary")
        if not summary:
            summary = "No significant history to summarize."

        # Append executed action sequence (like Orak MCP agent)
        if self.action_buffer:
            action_seq = '->'.join(self.action_buffer[-self.num_action_buffer:])
            summary += f"\nExecuted Action Sequence: (oldest)[{action_seq}](latest)"

        self.last_module = 'history_summarization'
        return summary
    
    def _store_experience_in_memory(self, subtask, action, outcome_description):
        """Store important experiences in vector memory"""
        current_map = self.state_dict['map_info'].get('map_name', 'Unknown')
        current_state = self.state_dict.get('state', 'Unknown')
        
        memory_content = f"Subtask: {subtask}. Action: {action}. Outcome: {outcome_description}. Location: {current_map}."
        
        metadata = {
            'step': self.step_count,
            'map_name': current_map,
            'state': current_state,
            'subtask': subtask,
            'action': action
        }
        
        self.vector_memory.add_memory(memory_content, metadata)
    
    def _detect_significant_events(self):
        """Detect significant events worth storing in memory"""
        events = []
        
        # Badge acquisition
        if self.prev_state_dict and self.state_dict:
            prev_badges = self.prev_state_dict.get('badge_list', [])
            curr_badges = self.state_dict.get('badge_list', [])
            new_badges = [b for b in curr_badges if b not in prev_badges]
            if new_badges:
                events.append(f"Acquired badge(s): {', '.join(new_badges)}")
        
        # Map transitions
        if self.prev_state_dict:
            prev_map = self.prev_state_dict.get('map_info', {}).get('map_name', '')
            curr_map = self.state_dict.get('map_info', {}).get('map_name', '')
            if prev_map and curr_map and prev_map != curr_map:
                events.append(f"Entered new area: {curr_map} (from {prev_map})")
        
        return events
    
    def _module_self_reflection(self, short_term_summary, prev_state_str, cur_state_str):
        """Module 2: Self Reflection"""

        subtask = self.last_subtask or "N/A"
        subtask_reasoning = self.last_subtask_reasoning or "N/A"

        query = f"Subtask: {subtask}. Current state: {self.state_dict.get('state')} in {self.state_dict['map_info'].get('map_name')}"
        retrieved_memories = self.vector_memory.retrieve_similar(query, top_k=3, threshold=0.6)
        relevant_memory = self.vector_memory.format_memories_for_prompt(retrieved_memories)

        user_prompt = SELF_REFLECTION_USER_PROMPT.format(
            short_term_summary=short_term_summary,
            subtask=subtask,
            subtask_reasoning=subtask_reasoning,
            prev_state_str=prev_state_str,
            cur_state_str=cur_state_str,
            relevant_memory=relevant_memory
        )

        response = self.client.responses.create(
            model=MODEL,
            input=user_prompt,
            instructions=SELF_REFLECTION_SYSTEM_PROMPT,
            reasoning={"effort": "low"}
        )

        output = response.output_text.strip()
        reflection_json = self._parse_section(output, "Self_reflection")

        if reflection_json:
            json_match = re.search(r'```json\s*(\{.+?\})\s*```', reflection_json, re.DOTALL)
            if json_match:
                reflection_json = json_match.group(1)
            # Store for NewFacts extraction in memory_management
            self.last_self_reflection = reflection_json
            self.last_module = 'self_reflection'
            return reflection_json

        self.last_module = 'self_reflection'
        return "{}"
    
    def _module_subtask_planning(self, short_term_summary, self_reflection, cur_state_str):
        """Module 3: Subtask Planning"""
        
        query = f"Planning next task in {self.state_dict.get('state')} at {self.state_dict['map_info'].get('map_name')}"
        retrieved_memories = self.vector_memory.retrieve_similar(query, top_k=3, threshold=0.6)
        relevant_memory = self.vector_memory.format_memories_for_prompt(retrieved_memories)
        
        user_prompt = SUBTASK_PLANNING_USER_PROMPT.format(
            short_term_summary=short_term_summary,
            self_reflection=self_reflection,
            cur_state_str=cur_state_str,
            relevant_memory=relevant_memory
        )
        
        response = self.client.responses.create(
            model=MODEL,
            input=user_prompt,
            instructions=SUBTASK_PLANNING_SYSTEM_PROMPT,
            reasoning={"effort": "low"}
        )
        
        output = response.output_text.strip()
        
        subtask_reasoning = self._parse_section(output, "Subtask_reasoning")
        subtask = self._parse_section(output, "Subtask")
        
        if not subtask:
            subtask = "Continue playing the game"
        if not subtask_reasoning:
            subtask_reasoning = "No reasoning provided"
        
        self.last_subtask = subtask
        self.last_subtask_reasoning = subtask_reasoning
        
        return subtask, subtask_reasoning

    def _module_action_inference(self, short_term_summary, cur_state_str, self_reflection, subtask_description):
        """Module 4: Action Inference - outputs use_tool() or low-level actions"""

        query = f"Action for: {subtask_description}. Location: {self.state_dict['map_info'].get('map_name')}"
        retrieved_memories = self.vector_memory.retrieve_similar(query, top_k=3, threshold=0.6)
        relevant_memory = self.vector_memory.format_memories_for_prompt(retrieved_memories)

        user_prompt = USER_PROMPT.format(
            short_term_summary=short_term_summary,
            cur_state_str=cur_state_str,
            self_reflection=self_reflection,
            subtask_description=subtask_description,
            relevant_memory=relevant_memory
        )

        # Use Responses API with reasoning
        response = self.client.responses.create(
            model=MODEL,
            input=user_prompt,
            instructions=SYSTEM_PROMPT,
            reasoning={"effort": "low"}
        )

        output = response.output_text.strip()

        # Check for use_tool() format
        tool_pattern = r'use_tool\s*\([^)]+\)'
        tool_match = re.search(tool_pattern, output, re.IGNORECASE)
        if tool_match:
            action = tool_match.group(0)
            return action

        # Parse low-level commands from ### Actions section
        actions_match = re.search(r"### Actions\s*(.+)", output, re.IGNORECASE | re.DOTALL)
        if actions_match:
            actions_block = actions_match.group(1).strip().splitlines()[0].strip()

            if actions_block.lower() == "quit":
                return "quit"

            # Parse low-level commands
            commands = [cmd.strip().lower() for cmd in actions_block.split("|")]
            valid_actions = ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select', 'none']
            if all(cmd in valid_actions for cmd in commands):
                action = " | ".join(commands)
                return action

        # Fallback
        return "none"
    
    def act(self, obs):
        """
        Act loop:
        1. History Summarization
        2. Self Reflection
        3. Memory Management (extract NewFacts, retrieve relevant memory)
        4. Subtask Planning
        5. Action Inference - returns use_tool() or low-level actions
        """

        # Process observation
        processed_obs = self._process_observation(obs)

        prev_state_str = str(self.prev_state_dict) if self.prev_state_dict else "N/A"
        cur_state_str = processed_obs

        # MODULE 1: History Summarization
        short_term_summary = self._module_history_summarization()

        # MODULE 2: Self Reflection (skip first step)
        if self.step_count > 0:
            self_reflection = self._module_self_reflection(
                short_term_summary, prev_state_str, cur_state_str
            )
        else:
            self_reflection = "N/A (first step)"

        # MODULE 3: Memory Management - extract NewFacts from reflection, store to LTM
        self._memory_management(self.last_subtask)

        # MODULE 4: Subtask Planning
        subtask_description, _ = self._module_subtask_planning(
            short_term_summary, self_reflection, cur_state_str
        )

        # MODULE 5: Action Inference - returns action string directly
        action = self._module_action_inference(
            short_term_summary, cur_state_str, self_reflection, subtask_description
        )

        # Update history and action buffer
        self.step_count += 1
        self.action_buffer.append(action)
        self.action_buffer = self.action_buffer[-self.num_action_buffer:]

        self.histories.append({
            'step': self.step_count,
            'state': self.state_dict.get('state', 'Unknown'),
            'map_name': self.state_dict['map_info'].get('map_name', 'Unknown'),
            'action': action
        })
        self.histories = self.histories[-self.max_history:]

        # Store significant events
        significant_events = self._detect_significant_events()
        for event in significant_events:
            event_memory = f"Significant event: {event}. Context: {subtask_description}"
            metadata = {
                'step': self.step_count,
                'map_name': self.state_dict['map_info'].get('map_name', 'Unknown'),
                'state': self.state_dict.get('state', 'Unknown'),
                'event_type': 'significant_event'
            }
            self.vector_memory.add_memory(event_memory, metadata)

        return action

