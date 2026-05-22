import time
import re
import heapq
import sys
import logging
import codecs
from mcp_game_servers.pokemon_red.game.utils.map_utils import *
from mcp_agent_servers.memory_utils import *
from mcp_game_servers.pokemon_red.game.utils.pokemon_tools import (
    classify_post_move_outcome,
)

class PokemonToolset:
    def __init__(self, client, logger, game_server_id, agent_server_id):
        self.client = client
        self.logger = logger
        self.game_server_id = game_server_id
        self.agent_server_id = agent_server_id
        
        # Configure logger to use UTF-8 encoding
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = codecs.getwriter('utf-8')(handler.stream.buffer if hasattr(handler.stream, 'buffer') else handler.stream)
                
        self.map_memories = {
            'state_dict': {},
            'map_memory_dict': {},
            'step_count': 0,
            'dialog_buffer': []
        }
        self.state_dict = {}
        self.map_memory_dict = {}
        self.step_count = 0
        self.dialog_buffer = []

    async def execute_action_response(self, action_response: str):
        try:
            # self.logger.info(f"[DEBUG] Starting execute_action_response with action: {action_response}")
            inside = action_response[len("use_tool("):-1]
            tool_name, arg_str = inside.split(",", 1)
            tool_name = tool_name.strip()
            arg_str = arg_str.strip().lstrip("(").rstrip(")")
            kwargs = dict(eval(f"dict({arg_str})"))
            
            # self.logger.info(f"[DEBUG] Calling method '{tool_name}' with args: {kwargs}")
            
            # Execute functions in toolset
            method = getattr(self, tool_name)
            try:
                result = await method(**kwargs)
                # self.logger.info(f"[DEBUG] Method '{tool_name}' completed with result: {result}")
            except Exception as method_error:
                self.logger.error(f"[DEBUG] Error in method '{tool_name}': {str(method_error)}", exc_info=True)
                return f"[execute_tool error] {str(method_error)}"
            
            # self.logger.info("[DEBUG] Updating map memories")
            self.map_memories = {
                'state_dict': self.state_dict,
                'map_memory_dict': self.map_memory_dict,
                'step_count': self.step_count,
                'dialog_buffer': self.dialog_buffer
            }

            await self.client.call_set_map_memories(self.agent_server_id, self.map_memories)
            self.dialog_buffer = []
                
            return result

        except Exception as e:
            self.logger.error(f"[DEBUG] Error in execute_action_response: {str(e)}", exc_info=True)
            return f"[execute_tool error] {str(e)}"


    async def _send_action_set(self, action_set):
        try:
            # self.logger.info(f"[DEBUG] Sending action set: {action_set}")
            # Ensure game_server_id is a string
            if isinstance(self.game_server_id, (list, tuple)):
                server_id = str(self.game_server_id[0]) if self.game_server_id else ""
            else:
                server_id = str(self.game_server_id)
            
            # Ensure action_set is a list
            if isinstance(action_set, str):
                action_set = [action_set]
            elif not isinstance(action_set, list):
                action_set = list(action_set)
                
            await self.client.call_send_action_set(action_set, server_id)
            # self.logger.info(f"[DEBUG] Successfully sent action set: {action_set}")
        except Exception as e:
            self.logger.error(f"[DEBUG] Error in _send_action_set: {str(e)}", exc_info=True)
            raise

    async def _get_current_state(self):
        try:
            # First, await the async calls and store their results
            current_state = await self.client.call_get_current_state(self.game_server_id)
            map_memories_result = await self.client.call_load_map_memories(self.agent_server_id)
            
            # Replace ¥ with unicode escape sequence for logging
            log_safe_state = current_state.replace('¥', '\\u00A5')
            # self.logger.info(f"[DEBUG] Raw current_state value: {repr(log_safe_state)}")
            
            # Parse game state with encoding handling
            try:
                # self.logger.info("[DEBUG] Attempting to parse game state")
                self.state_dict = parse_game_state(current_state)  # Use original state for parsing
                # self.logger.info("[DEBUG] Successfully parsed game state")
            except Exception as parse_error:
                self.logger.error(f"[DEBUG] Error parsing game state: {str(parse_error)}")
                self.logger.error(f"[DEBUG] Problematic state string: {log_safe_state}")
                raise
            
            # self.logger.info(f"[DEBUG] Parsed state dict: {self.state_dict}")
            
            # Unpack map memories after awaiting
            _, map_memory_dict, step_count, _ = map_memories_result
            
            # Process map memory data
            self.map_memory_dict = get_map_memory_dict(self.state_dict, map_memory_dict)
            self.step_count = step_count
            
            # self.dialog_buffer = dialog_buffer
            
            # Update map memories
            self.map_memories = {
                'state_dict': self.state_dict,
                'map_memory_dict': self.map_memory_dict,
                'step_count': self.step_count,
                'dialog_buffer': self.dialog_buffer
            }
            
            # Update server state
            await self.client.call_set_map_memories(self.agent_server_id, self.map_memories)
            
        except Exception as e:
            self.logger.error(f"[DEBUG] Error in _get_current_state: {str(e)}", exc_info=True)
            raise

    async def _get_player_pos(self):
        await self._get_current_state()
        return self.state_dict['map_info']['player_pos_x'], self.state_dict['map_info']['player_pos_y'], self.state_dict['map_info']['map_name']

    def _reconstruct_directions(self, parent, x_start, y_start, x_dest, y_dest):
        """
        Trace back the path from (x_start, y_start) to (x_dest, y_dest) and return as a string like "up | right | ..."
        """
        path = []
        cur = (x_dest, y_dest)

        while cur != (x_start, y_start):
            prev = parent[cur]
            dx = cur[0] - prev[0]
            dy = cur[1] - prev[1]

            # Jumping over a ledge is also treated as a one-tile movement
            if dx > 0:
                path.append("right")
            elif dx < 0:
                path.append("left")
            elif dy > 0:
                path.append("down")
            elif dy < 0:
                path.append("up")

            cur = prev

        path.reverse()
        return " | ".join(path)

    def _find_path_inner(self, x_dest, y_dest, isSurf=False):
        """
        Find a path to the target coordinate and return direction sequence.
        """
        # agent state, map access
        current_map_id = self.state_dict['map_info']['map_name']
        explored_map = self.map_memory_dict[current_map_id]["explored_map"]
        x_player = self.state_dict['map_info']["player_pos_x"]
        y_player = self.state_dict['map_info']["player_pos_y"]
        
        max_y = len(explored_map)
        max_x = len(explored_map[0])

        def in_bounds(x, y):
            return (0 <= x < max_x) and (0 <= y < max_y)

        def can_land(x, y, is_destination=False):
            """
            - 'WarpPoint' is only allowed when it is the destination
            - '~' is only accessible if isSurf=True
            - 'C' is not walkable
            """
            if not in_bounds(x, y):
                return False
            tile = explored_map[y][x]

            if tile in {'O', 'G'}:
                return True
            if tile == '~' and isSurf:
                return True
            if tile == 'WarpPoint' and is_destination:
                return True
            
            return False
        
        if not can_land(x_dest, y_dest, is_destination=True):
            return (False, f"Destination coordinate is not walkable ('{explored_map[y_dest][x_dest]}'). Please reset the destination.")

        def can_jump_ledge(x_ledge, y_ledge, dx, dy, is_destination=False):
            """
            If it's possible to jump over ledge (D/L/R), return the landing coordinate (x2, y2)
            """
            tile = explored_map[y_ledge][x_ledge]
            if tile == 'D':
                # 'D' is only passable when jumping from top to bottom (dy=1, dx=0)
                if dx == 0 and dy == 1:
                    x2, y2 = x_ledge, y_ledge + 1
                    if can_land(x2, y2, is_destination):
                        return (x2, y2)
            elif tile == 'L':
                # 'L' is only passable when jumping from right to left (dx=-1, dy=0)
                if dx == -1 and dy == 0:
                    x2, y2 = x_ledge - 1, y_ledge
                    if can_land(x2, y2, is_destination):
                        return (x2, y2)
            elif tile == 'R':
                # 'R' is only passable when jumping from left to right (dx=1, dy=0)
                if dx == 1 and dy == 0:
                    x2, y2 = x_ledge + 1, y_ledge
                    if can_land(x2, y2, is_destination):
                        return (x2, y2)
            return None

        def heuristic(cx, cy):
            # Manhattan distance
            return abs(cx - x_dest) + abs(cy - y_dest)

        # Use tuples for coordinates to ensure hashability
        start_pos = (x_player, y_player)
        end_pos = (x_dest, y_dest)
        
        heap = []
        heapq.heappush(heap, (heuristic(x_player, y_player), 0, start_pos))

        came_from = {}  # Dictionary using tuples as keys
        g_score = {start_pos: 0}

        while heap:
            _, cost_g, current_pos = heapq.heappop(heap)
            cx, cy = current_pos

            # Destination reached
            if current_pos == end_pos:
                return (True, self._reconstruct_directions(came_from, x_player, y_player, x_dest, y_dest))

            # 4 directions
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = cx + dx, cy + dy
                next_pos = (nx, ny)  # Create tuple for next position
                is_dest = (next_pos == end_pos)

                if in_bounds(nx, ny):
                    tile_next = explored_map[ny][nx]
                    # Check if the next tile is walkable
                    if tile_next in {'O', 'G', '~', 'WarpPoint', 'D', 'L', 'R'}:
                        if tile_next in {'D','L','R'}:
                            # Jumping over ledge
                            jump = can_jump_ledge(nx, ny, dx, dy, is_dest)
                            if jump is not None:
                                nx2, ny2 = jump
                                jump_pos = (nx2, ny2)  # Create tuple for jump position
                                new_g = cost_g + 1
                                if jump_pos not in g_score or new_g < g_score[jump_pos]:
                                    g_score[jump_pos] = new_g
                                    f_score = new_g + heuristic(nx2, ny2)
                                    heapq.heappush(heap, (f_score, new_g, jump_pos))
                                    came_from[jump_pos] = current_pos
                        else:
                            # Normal tile
                            if can_land(nx, ny, is_dest):
                                new_g = cost_g + 1
                                if next_pos not in g_score or new_g < g_score[next_pos]:
                                    g_score[next_pos] = new_g
                                    f_score = new_g + heuristic(nx, ny)
                                    heapq.heappush(heap, (f_score, new_g, next_pos))
                                    came_from[next_pos] = current_pos

        return (False, "No valid path to the destination currently. Reveal other '?' tiles first, then find a possible route.")
    
    async def _nudge_around_and_return(self, x, y, delay=0.3):
        """
        If the player is already at (x, y), move one step to an adjacent walkable tile and return.
        Useful for triggering warp points or resetting state.
        """
        explored_map = self.map_memory_dict[self.state_dict['map_info']['map_name']]["explored_map"]
        max_y = len(explored_map)
        max_x = len(explored_map[0])

        def in_bounds(x, y):
            return 0 <= x < max_x and 0 <= y < max_y

        adjacent_moves = {
            (0, -1): 'up',
            (0, 1): 'down',
            (-1, 0): 'left',
            (1, 0): 'right'
        }

        for (dx, dy), move_cmd in adjacent_moves.items():
            nx, ny = x + dx, y + dy
            if in_bounds(nx, ny) and explored_map[ny][nx] in {'O', 'G', '~'}:
                await self._send_action_set([move_cmd])
                time.sleep(delay)
                reverse_cmd = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}[move_cmd]
                await self._send_action_set([reverse_cmd])
                time.sleep(delay)
                return True
        return False

    async def _start_interact_inner(self, object_name, isSurf=False):
        """
        Return a direction sequence to move near and interact with the object.
        """
        try:
            # self.logger.info(f"[DEBUG] Starting _start_interact_inner for {object_name}")
            current_map_id = self.state_dict['map_info']['map_name']
            explored_map = self.map_memory_dict[current_map_id]["explored_map"]
            # self.logger.info(f"[DEBUG] Current map: {current_map_id}")
            # self.logger.info(f"[DEBUG] Explored map type: {type(explored_map)}, content type: {type(explored_map[0]) if explored_map else 'empty'}")

            max_y = len(explored_map)
            max_x = len(explored_map[0])
            
            def find_object_coordinates(explored_map, object_name):
                for y, row in enumerate(explored_map):
                    for x, cell in enumerate(row):
                        if cell == object_name:
                            return (x, y)
                return None

            def in_bounds(x, y):
                return (0 <= x < max_x) and (0 <= y < max_y)

            def can_land_for_interaction(x, y):
                """
                Check if the player can stand on the tile during interaction.
                (Same logic as for intermediate path or destination tiles)
                """

                if not in_bounds(x, y):
                    return False
                tile = explored_map[y][x]
                if tile in {'O', 'G'}:
                    return True
                if tile == '~' and isSurf:
                    return True
                if tile == 'WarpPoint':
                    return True
                return False

            coord_obj = find_object_coordinates(explored_map, object_name)
            if coord_obj is None:
                return (False, f"{object_name} is not found. Try exploring the map first, or refer to the full map data to identify the correct object name.")
            x_obj, y_obj = coord_obj

            directions = [(0,1),(0,-1),(1,0),(-1,0)]
            stand_candidates = []
            for dx, dy in directions:
                adjx = x_obj + dx
                adjy = y_obj + dy
                if in_bounds(adjx, adjy):
                    tile = explored_map[adjy][adjx]
                    if tile == 'C':
                        # If it's 'C', step back one more tile
                        bx = adjx + dx
                        by = adjy + dy
                        if can_land_for_interaction(bx, by):
                            stand_candidates.append((bx, by))
                    else:
                        # Normally adjacent tiles
                        if can_land_for_interaction(adjx, adjy):
                            stand_candidates.append((adjx, adjy))

            candidates_info = []

            def get_direction_priority(direction):
                if direction == "up":
                    return 1
                elif direction == "left":
                    return 2
                else:
                    return 3

            for (tx, ty) in stand_candidates:
                success, path_str = self._find_path_inner(tx, ty, isSurf)
                if success:
                    # Path length
                    steps_count = 1 + path_str.count("|")

                    dx = x_obj - tx
                    dy = y_obj - ty

                    # Diagonal positions are not valid for interaction
                    if dx != 0 and dy != 0:
                        continue

                    if dx > 0 and dy == 0:
                        facing = "right"
                    elif dx < 0 and dy == 0:
                        facing = "left"
                    elif dx == 0 and dy > 0:
                        facing = "down"
                    elif dx == 0 and dy < 0:
                        facing = "up"
                    else:
                        # Just in case (dx,dy) = (0,0), meaning the same location?
                        continue

                    priority = get_direction_priority(facing)
                    candidates_info.append((priority, steps_count, path_str, facing, (tx, ty)))

            if not candidates_info:
                return (False, "No valid path to the destination currently. Reveal other '?' tiles first, then find a possible route.")

            candidates_info.sort(key=lambda x: (x[0], x[1]))

            # Optimal candidate (shortest path and best direction priority)
            best_path_length, best_priority, best_path_str, best_facing, best_coord = candidates_info[0]

            # Return in the form 'path | direction | a'
            if best_path_str == '':
                return (True ,(best_coord, f"{best_facing} | a"))
            return (True, (best_coord, best_path_str + f" | {best_facing} | a"))

        except Exception as e:
            self.logger.error(f"[DEBUG] Error in _start_interact_inner: {str(e)}", exc_info=True)
            raise

    async def interact_with_object(self, object_name, isSurf=False, max_attempts=10):
        try:
            # self.logger.info(f"[DEBUG] Starting interact_with_object with object: {object_name}")
            await self._get_current_state()
            if self.state_dict['state'] != 'Field':
                return (False, f"Current state: '{self.state_dict['state']}' state, not 'Field' state")
            if object_name == 'WarpPoint':
                return (False, f"'WarpPoint' cannot be the target of this tool. Use `warp_with_warp_point` tool.")

            for attempt in range(max_attempts):
                # self.logger.info(f"[DEBUG] Attempt {attempt + 1} to interact with {object_name}")
                success, results = await self._start_interact_inner(object_name, isSurf)
                if success:
                    target_coord, actions = results
                    # self.logger.info(f"[DEBUG] Found path to {object_name}. Target coord: {target_coord}, Actions: {actions}")
                else:
                    # self.logger.error(f"[DEBUG] Failed to find path: {results}")
                    return (False, f"{results}")

                commands = re.split(r'[|/;, \t\n]+', actions)
                if not commands or commands == ['']:
                    return (False, f"Invalid command sequence")

                # Execute movement commands first
                movement_commands = commands[:-2]  # All except the last two commands (facing direction and 'a')
                if movement_commands:
                    # self.logger.info(f"[DEBUG] Executing movement commands: {movement_commands}")
                    for action in movement_commands:
                        await self._send_action_set([action])
                        time.sleep(0.3)
                        await self._get_current_state()
                        if self.state_dict['state'] != 'Field':
                            return (False, f"Interrupted during movement! Current state: '{self.state_dict['state']}'")

                # Execute interaction commands
                interaction_commands = commands[-2:]  # Last two commands (facing direction and 'a')
                # self.logger.info(f"[DEBUG] Executing interaction commands: {interaction_commands}")
                await self._send_action_set(interaction_commands)
                time.sleep(0.3)

                # Check for successful interaction
                x, y, map = await self._get_player_pos()
                # self.logger.info(f"[DEBUG] Current state_dict (line 464: pokemon_tools_mcp.py): {self.state_dict}")
                if self.state_dict['state'] == 'Dialog' and (x, y) == target_coord:
                    self.dialog_buffer.append(self.state_dict['filtered_screen_text'])
                    # self.logger.info("[DEBUG] Successfully entered Dialog state")
                    time.sleep(1.0)
                    success, _ = await self.continue_dialog()
                    if success:
                        return (True, f"Successfully interacted with {object_name}")
                    else:
                        return (False, f"Dialog continuation failed with {object_name}")
                elif self.state_dict['state'] == 'Dialog' and (x, y) != target_coord:
                    return (False, "Interrupt by Someone's Dialog.")
                elif self.state_dict['state'] == 'Field':
                    continue  # Try again if still in Field state
                elif 'Battle' in self.state_dict['state']:
                    return (False, "Interrupted by Battle")

            return (False, f"Unable to interact with {object_name} after {max_attempts} attempts")

        except Exception as e:
            self.logger.error(f"[DEBUG] Error in interact_with_object: {str(e)}", exc_info=True)
            raise

    async def move_to(self, x_dest, y_dest, isSurf=False, max_attempts=3):
        try:
            # self.logger.info(f"[DEBUG] Starting move_to to coordinates ({x_dest}, {y_dest})")
            await self._get_current_state()
            
            if self.state_dict['state'] != 'Field':
                return (False, f"'{self.state_dict['state']}' state, not 'Field' state")
                
            prev_map = self.state_dict['map_info']['map_name']
            explored_map = self.map_memory_dict[prev_map]["explored_map"]
            target_coord = (x_dest, y_dest)
            x_player = self.state_dict['map_info']["player_pos_x"]
            y_player = self.state_dict['map_info']["player_pos_y"]
            
            # self.logger.info(f"[DEBUG] Current position: ({x_player}, {y_player}), Target: {target_coord}")
            
            if explored_map[y_dest][x_dest] == 'WarpPoint':
                return (False, f"The destination is 'WarpPoint'. Use 'warp_with_warp_point' tool.")
            elif (x_dest, y_dest) == (x_player, y_player):
                return (False, f"The destination is the current position. Set another destination.")

            for attempt in range(max_attempts):
                # self.logger.info(f"[DEBUG] Attempt {attempt + 1} to move to destination")
                success, actions = self._find_path_inner(x_dest, y_dest, isSurf)
                if not success:
                    # self.logger.error(f"[DEBUG] Failed to find path: {actions}")
                    return (False, f"{actions}")
                    
                commands = re.split(r'[|/;, \t\n]+', actions)
                if not commands or commands == ['']:
                    return (False, f"The destination is already your position")
                    
                # self.logger.info(f"[DEBUG] Executing movement commands: {commands}")
                for action in commands:
                    await self._send_action_set([action])
                    time.sleep(0.3)
                    await self._get_current_state()
                    if self.state_dict['state'] != 'Field':
                        return (False, f"Interrupted! Current state: '{self.state_dict['state']}' state, not 'Field' state")
                time.sleep(0.1)

                state_dict = self.state_dict
                if state_dict['state'] != 'Field':
                    return (False, f"Cannot move the position. Currently in {state_dict['state']} state.")

                x_player = state_dict['map_info']["player_pos_x"]
                y_player = state_dict['map_info']["player_pos_y"]
                verdict = classify_post_move_outcome(
                    prev_map=prev_map,
                    current_map=state_dict['map_info']['map_name'],
                    current_pos=(x_player, y_player),
                    target=target_coord,
                    state=state_dict['state'],
                    x_dest=x_dest,
                    y_dest=y_dest,
                )
                if verdict is not None:
                    return verdict

            return (False, f"Unable to move to ({x_dest}, {y_dest}) after {max_attempts} attempts.")
            
        except Exception as e:
            self.logger.error(f"[DEBUG] Error in move_to: {str(e)}", exc_info=True)
            raise

    async def warp_with_warp_point(self, x_dest, y_dest, max_attempts=3):
        await self._get_current_state()
        if self.state_dict['state'] != 'Field':
            return (False, f"Current state: '{self.state_dict['state']}' state, not 'Field' state")
        prev_map = self.state_dict['map_info']['map_name']
        explored_map = self.map_memory_dict[prev_map]["explored_map"]
        
        if not explored_map[y_dest][x_dest] == 'WarpPoint':
            return (False, f"({x_dest}, {y_dest}) is not 'WarpPoint'")
        
        for attempt in range(max_attempts):
            if self.state_dict['state'] != 'Field':
                return (False, f"Cannot move the position. Currently in {self.state_dict['state']} state.")
            
            player_x = self.state_dict['map_info']["player_pos_x"]
            player_y = self.state_dict['map_info']["player_pos_y"]
            if (player_x, player_y) == (x_dest, y_dest):
                x1, y1, map1 = await self._get_player_pos()
                await self._nudge_around_and_return(x_dest, y_dest)
                x2, y2, map2 = await self._get_player_pos()
            else:
                success, actions = self._find_path_inner(x_dest, y_dest)
                if not success:
                    return (success, f"{actions}")

                commands = re.split(r'[|/;, \t\n]+', actions)

                await self._send_action_set(commands[:-1])
                x1, y1, map1 = await self._get_player_pos()
                time.sleep(0.1)
                
                await self._send_action_set(commands[-1:])
                x2, y2, map2 = await self._get_player_pos()

                time.sleep(0.1)

            warp_cond1 = abs(x1 - x2) > 1 or abs(y1 - y2) > 1
            warp_cond2 = (map1 != map2)
            warp_cond = warp_cond1 or warp_cond2

            await self._get_current_state()
            state_dict = self.state_dict
            current_map = self.state_dict['map_info']['map_name']

            if warp_cond:
                return (True, f"Success to warp to {current_map} ({x2}, {y2}) using a warp point ({x_dest}, {y_dest}) in {prev_map}")
            elif state_dict["state"] == 'Dialog':
                return (False, f"Interrupt by Someone's Dialog before you move to a warp point ({x_dest}, {y_dest}).")
            elif 'Battle' in state_dict["state"]:
                return (False, "Interrupt by Battle.")
            else:
                # Check if it's a boundary
                player_x = state_dict['map_info']["player_pos_x"]
                player_y = state_dict['map_info']["player_pos_y"]
                x_max = state_dict['map_info']['x_max']
                y_max = state_dict['map_info']['y_max']

                direction = None
                if player_y == 0:
                    direction = 'up'
                elif player_y == y_max:
                    direction = 'down'
                elif player_x == 0:
                    direction = 'left'
                elif player_x == x_max:
                    direction = 'right'

                if direction:
                    # Try moving one more step
                    await self._send_action_set([direction])
                    time.sleep(0.5)

                    # Check the state again
                    await self._get_current_state()
                    state_dict = self.state_dict
                    current_map = self.state_dict['map_info']['map_name']
                    x2, y2, map2 = await self._get_player_pos()

                    if prev_map != current_map:
                        return (True, f"Success to warp to {current_map} ({x2}, {y2}) using a warp point ({x_dest}, {y_dest}) in {prev_map} (via boundary move)")
        return (False, f"Unable to warp after {max_attempts} attempts.")
        

    async def overworld_map_transition(self, direction='north', max_attempts=3):
        await self._get_current_state()
        if self.state_dict['state'] != 'Field':
            return (False, f"Current state: '{self.state_dict['state']}' state, not 'Field' state")
        map_info = self.state_dict['map_info']
        if map_info['map_type'] != 'overworld':
            return (False, "Map type is not 'overworld'")

        # Check if the map can be expanded in the given direction
        expansion_direction = map_info['expansion_direction']
        expansion_direction = expansion_direction.strip('\'"')
        tokens = re.split(r'[|,/;\s]+', expansion_direction)
        expansion_direction_list = [token.lower() for token in tokens if token.strip()]

        if direction not in expansion_direction_list:
            return (False, f"{direction} is not a valid direction")

        x_max = map_info['x_max']
        y_max = map_info['y_max']

        # TODO: '~' will be added after the 'SURF'
        walkable_tiles = {'O', 'G', 'WarpPoint'}  # Walkable tiles

        # for attempt in range(max_attempts):
        prev_map = self.state_dict['map_info']['map_name']
        explored_map = self.map_memory_dict[prev_map]['explored_map']

        for attempt in range(max_attempts):
            if self.state_dict['state'] != 'Field':
                return (False, f"Cannot move the position. Currently in {self.state_dict['state']} state.")
            
            if direction == 'north':
                candidates = [(x, 0) for x in range(x_max) if explored_map[0][x] in walkable_tiles]
                post_action = 'up'
            elif direction == 'south':
                candidates = [(x, y_max) for x in range(x_max) if explored_map[y_max][x] in walkable_tiles]
                post_action = 'down'
            elif direction == 'west':
                candidates = [(0, y) for y in range(y_max) if explored_map[y][0] in walkable_tiles]
                post_action = 'left'
            elif direction == 'east':
                candidates = [(x_max, y) for y in range(y_max) if explored_map[y][x_max] in walkable_tiles]
                post_action = 'right'
            else:
                return (False, f"{direction} is not valid direction")

            if len(candidates)==0:
                return (False, f"The {direction} side boundary is fully unknown ('?'). Prioritize uncovering other '?' {direction} boundary tiles first, then retry map transition.")

            for i, (x, y) in enumerate(candidates):
                success, actions = self._find_path_inner(x, y)
                if success:
                    break
                elif i==len(candidates)-1:
                    return (False, f"No valid path to the {direction} boundary. Prioritize uncovering other '?' {direction} boundary tiles first, then retry map transition.")
            if actions == '':
                commands = [post_action]
            else:
                actions += f' | {post_action}'
                commands = re.split(r'[|/;, \t\n]+', actions)

            await self._send_action_set(commands)
            time.sleep(0.1)

            await self._get_current_state()
            state_dict = self.state_dict
            current_map = self.state_dict['map_info']['map_name']

            if prev_map != current_map:
                return (True, f"Success to warp from {prev_map} to {current_map}")
            elif state_dict["state"] == 'Dialog':
                return (False, f"Interrupt by Someone's Dialog before your transition.")
            elif 'Battle' in state_dict["state"]:
                return (False, "Interrupt by Battle.")
        return (False, f"Unable to warp after {max_attempts} attempts.")
    
    async def continue_dialog(self):
        "Continuing dialog until selectable options appear or the dialog is over (go to Field or Battle state)"        
        for _ in range(30):
            await self._send_action_set(['a'])
            time.sleep(0.5)

            # text_obs = self.agent.env._receive_state()
            current_state = await self.client.call_get_current_state(self.game_server_id)
            
            # Replace ¥ with unicode escape sequence for logging
            current_state = current_state.replace('¥', '\\u00A5')
            
            # self.logger.info(f"[DEBUG] Toolset: get current state: {current_state}")
            self.state_dict = parse_game_state(current_state)
            self.dialog_buffer.append(self.state_dict['filtered_screen_text'])

            if self.state_dict['state'] == 'Field':
                return (True, f"Success to finish the dialog and enter {self.state_dict['state']} state")
            elif self.state_dict['selection_box_text'] != "N/A":
                return (True, "Selection box appears")

        return (True, "Still in Dialog State")
            
    async def select_move_in_battle(self, move_name, max_attempts=3):
        await self._get_current_state()
        if not 'Battle' in self.state_dict['state']:
            return (False, f"Current state is {self.state_dict['state']}, not Battle! Don't use this battle tool and directly choose {move_name} through POKéMON menu")
        # select 'FIGHT' option
        for i in range(max_attempts):
            if 'FIGHT' not in self.state_dict['selection_box_text']:
                action_sequence = ['b'] * 4
                await self._send_action_set(action_sequence)
                time.sleep(0.1)
                await self._get_current_state()
                if i==2:
                    return (False, "Something went wrong")
            else:
                break
        self.dialog_buffer.append(self.state_dict['filtered_screen_text'])

        action_sequence = ['up', 'left', 'a']
        await self._send_action_set(action_sequence)
        time.sleep(0.1)

        # select [move_name] option
        for _ in range(30):
            await self._get_current_state()
            state_dict = self.state_dict
            
            options = state_dict['selection_box_text'].split('\n')
            cursor_row = -1
            move_row = -1
            for row_idx, row in enumerate(options):
                if "▶" in row:
                    cursor_row = row_idx
                if move_name in row:
                    move_row = row_idx
                if cursor_row != -1 and move_row != -1:
                    break
            if move_row == -1:
                return (False, f"No {move_name} in this options")
            if cursor_row == move_row:
                action = 'a'
                break
            elif cursor_row < move_row:
                action = 'down'
            else:
                action = 'up'
            self.dialog_buffer.append(self.state_dict['filtered_screen_text'])
            await self._send_action_set([action])
            time.sleep(0.1)
        
        self.dialog_buffer.append(self.state_dict['filtered_screen_text'])
        await self._send_action_set([action])
        time.sleep(1.0)

        # continue dialog
        await self._get_current_state()
        success, _ = await self.continue_dialog()
        if success:
            return (True, f"Successfully used {move_name}")

    async def switch_pkmn_in_battle(self, pokemon_name, max_attempts=3):
        await self._get_current_state()
        if not 'Battle' in self.state_dict['state']:
            return (False, f"Current state is {self.state_dict['state']}, not Battle! Don't use this battle tool and directly access to POKéMON menu")
        
        # select '<Pk><Mn>' option
        for i in range(max_attempts):
            if 'FIGHT' not in self.state_dict['selection_box_text']:
                action_sequence = ['b'] * 4
                await self._send_action_set(action_sequence)
                time.sleep(0.1)
                await self._get_current_state()
                if i==2:
                    return (False, "Something went wrong")
            else:
                break
        self.dialog_buffer.append(self.state_dict['filtered_screen_text'])

        action_sequence = ['up', 'right', 'a']
        await self._send_action_set(action_sequence)
        time.sleep(0.1)
        
        # select [pokemon_name] option
        for _ in range(30):
            await self._get_current_state()
            state_dict = self.state_dict
            
            options = state_dict['filtered_screen_text'].split('\n')
            cursor_row = -1
            name_row = -1
            for row_idx, row in enumerate(options):
                if "▶" in row:
                    cursor_row = row_idx - 1
                if pokemon_name in row:
                    name_row = row_idx
                if cursor_row != -1 and name_row != -1:
                    break
            if name_row == -1:
                return (False, f"No {pokemon_name} in this options")
            if cursor_row == name_row:
                action = 'a'
                break
            elif cursor_row < name_row:
                action = 'down'
            else:
                action = 'up'
            self.dialog_buffer.append(self.state_dict['filtered_screen_text'])
            await self._send_action_set([action])
            time.sleep(0.1)
        
        self.dialog_buffer.append(self.state_dict['filtered_screen_text'])
        await self._send_action_set([action])
        time.sleep(0.1)

        # select [SWITCH] option
        await self._get_current_state()
        self.dialog_buffer.append(self.state_dict['filtered_screen_text'])
        await self._send_action_set(['a'])
        time.sleep(1.0)

        # continue dialog
        await self._get_current_state()
        success, _ = await self.continue_dialog()
        if success:
            return (True, f"Successfully switched to {pokemon_name}")

    async def run_away(self, max_attempts=3):
        await self._get_current_state()
        # select 'RUN' option
        for i in range(max_attempts):
            if 'FIGHT' not in self.state_dict['selection_box_text']:
                action_sequence = ['b'] * 4
                await self._send_action_set(action_sequence)
                time.sleep(0.1)

                await self._get_current_state()
                if i==2:
                    return (False, "Something went wrong")
            else:
                break
        self.dialog_buffer.append(self.state_dict['filtered_screen_text'])

        action_sequence = ['down', 'right', 'a']
        await self._send_action_set(action_sequence)
        time.sleep(1.0)
        
        # continue dialog
        await self._get_current_state()
        success, _ = await self.continue_dialog()
        if success:
            return (True, f"Successfully run!")

    async def use_item_in_battle(self, item_name, pokemon_name=None, max_attempts=3):
        await self._get_current_state()
        # Check if item_name is in the bag
        if not item_name in self.state_dict['inventory']:
            return (False, f"No {item_name} in your bag.")
        
        if not 'Battle' in self.state_dict['state']:
            return (False, f"Current state is {self.state_dict['state']}, not Battle! Don't use this battle tool and directly access to ITEM menu")
        
        # select 'ITEM' option
        for i in range(max_attempts):
            if 'FIGHT' not in self.state_dict['selection_box_text']:
                action_sequence = ['b'] * 4
                await self._send_action_set(action_sequence)
                time.sleep(0.1)
                
                await self._get_current_state()
                if i==2:
                    return (False, "Something went wrong")
            else:
                break
        self.dialog_buffer.append(self.state_dict['filtered_screen_text'])

        action_sequence = ['down', 'left', 'a']
        await self._send_action_set(action_sequence)
        time.sleep(0.1)
        
        bag_state = self.state_dict['inventory'].split('\n')
        for i, item_info in enumerate(bag_state):
            if item_name in item_info:
                break
        query_item_idx = i

        # Move cursor to [item_name] and select it
        for _ in range(30):
            # Current cursor item position
            await self._get_current_state()
            state_dict = self.state_dict
            options = state_dict['selection_box_text'].split('\n')
            for row in options:
                if "▶" in row:
                    break
            current_option = row[1:]

            for i, item_info in enumerate(bag_state):
                if current_option in item_info:
                    break
            current_item_idx = i
            if query_item_idx == current_item_idx:
                action = 'a'
                break
            elif query_item_idx < current_item_idx:
                action = 'up'
            else:
                action = 'down'
                
            self.dialog_buffer.append(self.state_dict['filtered_screen_text'])
            
            await self._send_action_set([action])
            time.sleep(0.1)
        
        self.dialog_buffer.append(self.state_dict['filtered_screen_text'])

        await self._send_action_set([action])
        time.sleep(0.1)
        
        # If 'Use item on which' detected, select [pokemon_name]
        if 'Use item on which' in self.state_dict['filtered_screen_text'] and pokemon_name is None:
            return (False, f"You have to select a specific Pokemon.")
        elif 'Use item on which' in self.state_dict['filtered_screen_text'] and pokemon_name is not None:
            for _ in range(30):
                await self._get_current_state()
                state_dict = self.state_dict
                
                options = state_dict['filtered_screen_text'].split('\n')
                cursor_row = -1
                name_row = -1
                for row_idx, row in enumerate(options):
                    if "▶" in row:
                        cursor_row = row_idx - 1
                    if pokemon_name in row:
                        name_row = row_idx
                    if cursor_row != -1 and name_row != -1:
                        break
                if name_row == -1:
                    return (False, f"No {pokemon_name} in this options")
                if cursor_row == name_row:
                    action = 'a'
                    break
                elif cursor_row < name_row:
                    action = 'down'
                else:
                    action = 'up'
                self.dialog_buffer.append(self.state_dict['filtered_screen_text'])
                
                await self._send_action_set([action])
                time.sleep(0.1)
            
            self.dialog_buffer.append(self.state_dict['filtered_screen_text'])
            
            await self._send_action_set([action])
            time.sleep(1.0)
        
        await self._get_current_state()
        success, _ = await self.continue_dialog()
        if success:
            return (True, f"Successfully used {item_name}")

