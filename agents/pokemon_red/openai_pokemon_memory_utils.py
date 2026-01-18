"""
Agent-side memory utilities for Pokemon Red agent.

This module is fully self-contained with no server-side dependencies.
Participants can freely modify these utilities to implement their own
memory management and observation processing strategies.
"""
import re


# =============================================================================
# Game State Parsing
# =============================================================================

def parse_game_state(text):
    """
    Parse game state text into structured dictionary.
    
    Participants can modify this to extract different information
    or parse the state differently for their agent's needs.
    """
    result = {}

    # 1. State
    state_match = re.search(r"State:\s*(\w+)", text)
    result['state'] = state_match.group(1) if state_match else None

    # 2. Filtered Screen Text
    filtered_text = re.search(r"\[Filtered Screen Text\]\n(.*?)(?=\[Selection Box Text\])", text, re.DOTALL)
    text_tmp = filtered_text.group(1).strip() if filtered_text else ""
    result['filtered_screen_text'] = text_tmp if text_tmp != "" else "N/A"

    # 3. Selection Box Text
    selection_box = re.search(r"\[Selection Box Text\]\n(.*?)(?=\[Enemy Pokemon\])", text, re.DOTALL)
    text_tmp = selection_box.group(1).strip() if selection_box else ""
    result['selection_box_text'] = text_tmp if text_tmp != "" else "N/A"

    # 4. Enemy Pokemon
    enemy_pokemon = {}
    enemy_section = re.search(r"\[Enemy Pokemon\]\n(.*?)(?=\[Current Party\])", text, re.DOTALL)
    if enemy_section:
        for line in enemy_section.group(1).splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                enemy_pokemon[key.strip()] = value.strip()
    result['enemy_pokemon'] = enemy_pokemon

    # 5. Your Party
    party_match = re.search(r"\[Current Party\]\n(.*?)(?=\[Badge List\])", text, re.DOTALL)
    result['your_party'] = party_match.group(1).strip() if party_match else ""

    # 6. Badge List
    badge_match = re.search(r"\[Badge List\]\n(.*?)(?=\[Bag\])", text, re.DOTALL)
    result['badge_list'] = badge_match.group(1).strip() if badge_match else ""

    # 7. Inventory
    inventory_match = re.search(r"\[Bag\]\n(.*?)(?=\[Current Money\])", text, re.DOTALL)
    result['inventory'] = inventory_match.group(1).strip() if inventory_match else ""

    # 8. Current Money
    money_match = re.search(r"\[Current Money\]:\s*¥(\d+)", text)
    result['money'] = int(money_match.group(1)) if money_match else 0

    # 9. Map Info
    map_info = {}
    map_section = re.search(r"\[Map Info\]\n(.*)", text, re.DOTALL)
    if map_section:
        map_text = map_section.group(1)
        map_name_match = re.search(r"Map Name:\s*(.*?),", map_text)
        map_info['map_name'] = map_name_match.group(1) if map_name_match else None

        map_type_match = re.search(r"Map type:\s*(.*)", map_text)
        map_info['map_type'] = map_type_match.group(1).strip() if map_type_match else None

        expansion_match = re.search(r"Expansion direction:\s*(.*)", map_text)
        map_info['expansion_direction'] = expansion_match.group(1).strip() if expansion_match else None

        coords_match = re.search(r"\(x_max , y_max\):\s*\((\d+),\s*(\d+)\)", map_text)
        map_info['x_max'] = int(coords_match.group(1)) if coords_match else None
        map_info['y_max'] = int(coords_match.group(2)) if coords_match else None

        pos_match = re.search(r"Your position \(x, y\): \((\d+), (\d+)\)", map_text)
        map_info['player_pos_x'] = int(pos_match.group(1)) if pos_match else None
        map_info['player_pos_y'] = int(pos_match.group(2)) if pos_match else None

        facing_match = re.search(r"Your facing direction:\s*(\w+)", map_text)
        map_info['facing'] = facing_match.group(1) if facing_match else None

        try:
            map_info['map_screen_raw'] = re.search(r"Map on Screen:\n(.+)", map_text, re.DOTALL).group(1).strip()
        except:
            map_info['map_screen_raw'] = None

    result['map_info'] = map_info

    return result


# =============================================================================
# Map Memory Management
# =============================================================================

def construct_init_map(x_max, y_max, map_screen_raw):
    """Create initial map grid from raw map screen data."""
    width, height = x_max + 1, y_max + 1

    # Initialize empty map with unexplored tiles
    maps = [['?' for _ in range(width)] for _ in range(height)]

    # Extract coordinates and symbols from map_screen_raw
    if map_screen_raw:
        map_lines = map_screen_raw.strip().split('\n')
        for line in map_lines:
            tile_matches = re.findall(r"\(\s*(\d+),\s*(\d+)\):\s*([^\s]+)", line)
            for x_str, y_str, val in tile_matches:
                x, y = int(x_str), int(y_str)
                if 0 <= x < width and 0 <= y < height:
                    maps[y][x] = val

    return maps


def refine_current_map(maps, x_max, y_max, map_screen_raw):
    """Update existing map grid with new map screen data."""
    width, height = x_max + 1, y_max + 1

    if map_screen_raw:
        map_lines = map_screen_raw.strip().split('\n')
        sprite_positions = []

        for line in map_lines:
            tile_matches = re.findall(r"\(\s*(\d+),\s*(\d+)\):\s*([^\s]+)", line)
            for x_str, y_str, val in tile_matches:
                x, y = int(x_str), int(y_str)
                if 0 <= x < width and 0 <= y < height:
                    if val.startswith("SPRITE_"):
                        sprite_positions.append((x, y, val))
                    else:
                        maps[y][x] = val

        # Separately process SPRITEs (they move, so clear old positions)
        for x, y, sprite_val in sprite_positions:
            for row in range(height):
                for col in range(width):
                    if maps[row][col] == sprite_val and (col != x or row != y):
                        maps[row][col] = '?'
            maps[y][x] = sprite_val

    return maps


def get_map_memory_dict(state_dict: dict, map_memory_dict: dict) -> dict:
    """
    Updates the map memory dictionary based on the current state.
    
    This is the agent's exploration memory strategy. Participants can modify
    this to implement different memory management approaches.
    """
    current_map = state_dict['map_info']['map_name']
    if state_dict['map_info']['x_max'] is not None:
        if current_map not in map_memory_dict.keys():
            map_memory_dict[current_map] = {
                "explored_map": construct_init_map(
                    state_dict['map_info']['x_max'],
                    state_dict['map_info']['y_max'],
                    state_dict['map_info']['map_screen_raw']
                ),
                "history": [],
            }
        else:
            map_memory_dict[current_map]["explored_map"] = refine_current_map(
                map_memory_dict[current_map]["explored_map"],
                state_dict['map_info']['x_max'],
                state_dict['map_info']['y_max'],
                state_dict['map_info']['map_screen_raw']
            )
    return map_memory_dict


# =============================================================================
# Observation Processing
# =============================================================================

def replace_map_on_screen_with_full_map(state_text: str, map_current: list, warp_annotations: dict = None) -> str:
    """
    Replace the partial map view in observation with the agent's full explored map.
    
    Args:
        state_text: Original observation string.
        map_current: 2D list representing the current map grid.
        warp_annotations: Dictionary mapping (x, y) coordinates to destination map strings.
    """
    # Return original text if map_current is empty or invalid
    if not map_current or not isinstance(map_current, list) or \
        not (all(isinstance(row, list) for row in map_current) if map_current else True):
        return state_text
    if map_current and not map_current[0]:  # handles case like [[]]
        map_current = []  # Treat as empty map

    # --- 0. Remove "Map on Screen" section first ---
    processed_state_text = re.sub(
        r"Map on Screen:(?:\n(?:\(\s*\d+,\s*\d+\): [^\n]+\n*)+)?",
        "", state_text, flags=re.DOTALL
    )

    # --- Then, fill 'N/A' to other specified empty sections ---
    section_names = [
        "Filtered Screen Text",
        "Selection Box Text",
        "Enemy Pokemon",
        "Current Party",
        "Badge List",
        "Bag",
        "Current Money"
    ]
    for section in section_names:
        pattern = rf"(\[{re.escape(section)}\])\n(?=\s*\[|\Z)"
        processed_state_text = re.sub(pattern, r"\1\nN/A\n", processed_state_text, flags=re.MULTILINE)

    # Clean up potentially multiple blank lines
    processed_state_text = re.sub(r"\n\s*\n", "\n\n", processed_state_text).strip()

    # --- 1. Extract player position ---
    player_x, player_y = -1, -1
    player_pos_match = re.search(r"Your position \(x, y\): \((\d+), (\d+)\)", processed_state_text)
    if player_pos_match:
        player_x = int(player_pos_match.group(1))
        player_y = int(player_pos_match.group(2))

    # --- 2. Generate the compact full map text ---
    map_grid_lines = []
    notable_objects = {}

    if not map_current:
        full_map_text_block = "[Full Map]\n(Map data is empty or malformed)\n"
    else:
        num_rows = len(map_current)
        num_cols = len(map_current[0])
        if num_cols == 0:
            full_map_text_block = "[Full Map]\n(Map data has rows but no columns)\n"
            map_current = []
        else:
            actual_x_max = num_cols - 1
            actual_y_max = num_rows - 1

            # --- Calculate paddings and dimensions ---
            y_label_num_width = len(str(actual_y_max)) if actual_y_max >= 0 else 1
            max_y_label_str = f"{actual_y_max:<{y_label_num_width}} | "
            header_left_padding = " " * len(max_y_label_str)

            # --- Map Header Construction ---
            # Column numbers (tens digits if needed, then units)
            col_headers_digits_only_list = []
            if num_cols > 0:
                if num_cols >= 100:
                    col_headers_digits_only_list.append("".join([str(i // 100 % 10) if i >= 100 else ' ' for i in range(num_cols)]))
                if num_cols >= 10:
                    col_headers_digits_only_list.append("".join([str(i // 10 % 10) if i >= 10 else ' ' for i in range(num_cols)]))
                col_headers_digits_only_list.append("".join([str(i % 10) for i in range(num_cols)]))

            for digits_str in col_headers_digits_only_list:
                map_grid_lines.append(f"{header_left_padding}{digits_str}")
            
            # Separator line
            map_grid_lines.append(f"{header_left_padding}+{'-' * num_cols}+")

            # --- Map Rows Construction ---
            for y_coord in range(num_rows):
                current_y_label_str = f"{y_coord:<{y_label_num_width}} | "
                line_content_chars = []
                for x_coord in range(num_cols):
                    val_at_cell = map_current[y_coord][x_coord]
                    original_char_code = '?'

                    if val_at_cell and isinstance(val_at_cell, str):
                        if len(val_at_cell) == 1:
                            original_char_code = val_at_cell
                        else:
                            original_char_code = val_at_cell[0].upper()
                            # Handle WarpPoint defaults
                            if val_at_cell == "WarpPoint" or "WARP" in val_at_cell.upper():
                                is_known_warp = False
                                if warp_annotations and (x_coord, y_coord) in warp_annotations:
                                    is_known_warp = True
                                
                                if not is_known_warp:
                                     notable_objects[(x_coord, y_coord)] = f"{val_at_cell} (Unexplored)"
                                else:
                                     notable_objects[(x_coord, y_coord)] = f"{val_at_cell}"
                            else:
                                notable_objects[(x_coord, y_coord)] = f"{val_at_cell}"
                    elif val_at_cell is None or val_at_cell == "":
                        original_char_code = '?'
                    else:
                        original_char_code = 'E'
                        notable_objects[(x_coord, y_coord)] = f"E: Invalid_Data_Type({type(val_at_cell).__name__})"
                    
                    line_content_chars.append(original_char_code)
                
                map_row_content_str = "".join(line_content_chars)
                map_grid_lines.append(f"{current_y_label_str}{map_row_content_str}")

            # --- Add Annotations manually if not already in notable_objects ---
            if warp_annotations:
                for coord, note_text in warp_annotations.items():
                   if coord not in notable_objects:
                        # If unknown, assume it's a WarpPoint or just use the text
                        notable_objects[coord] = f"WarpPoint -> {note_text}" if "->" not in note_text and ("Map" not in note_text and "SPRITE" not in note_text) else note_text
                   else:
                        # Avoid redundancy if the annotation repeats the object name
                        # e.g. Object: "SPRITE_1", Annotation: "SPRITE_1 (Said: ...)" mechanism
                        current_val = notable_objects[coord]
                        # Clean current value of tags like (Unexplored) for comparison
                        base_val = current_val.split(" (")[0]
                        
                        if note_text.startswith(base_val):
                            notable_objects[coord] = note_text
                        else:
                            notable_objects[coord] += f" -> {note_text}"

            # --- Assemble the [Full Map] and [Notable Objects] blocks ---
            full_map_text_block = "[Full Map]\n" + "\n".join(map_grid_lines)
            if notable_objects:
                notable_list_str = "\n\n[Notable Objects]"
                sorted_notables_coords = sorted(notable_objects.keys(), key=lambda k: (k[1], k[0]))
                for coord_key in sorted_notables_coords:
                    x_obj, y_obj = coord_key
                    notable_list_str += f"\n({x_obj:2}, {y_obj:2}) {notable_objects[coord_key]}"
                full_map_text_block += notable_list_str
    
    # --- 3. Append the full map text block to the end ---
    if processed_state_text:
        final_state_text = processed_state_text + "\n\n" + full_map_text_block
    else:
        final_state_text = full_map_text_block
        
    # Final cleanup
    final_state_text = re.sub(r"\n{3,}", "\n\n", final_state_text).strip()
    
    return final_state_text


def replace_filtered_screen_text(state_text: str, dialog_buffer: list) -> str:
    """
    Insert dialog buffer into observation text.
    
    This is the agent's dialog history handling strategy. Participants can modify
    this to present dialog information differently to their agent.
    """
    if not dialog_buffer:
        return state_text

    # Generate new section
    new_section = f"[Interacted Dialog Buffer]\n" + "\n".join(dialog_buffer) + "\n\n"

    # Find location of [Filtered Screen Text]
    match = re.search(r"(?=\[Filtered Screen Text\])", state_text)
    if match:
        insert_index = match.start()
        new_state_text = state_text[:insert_index] + new_section + state_text[insert_index:]
        return new_state_text
    else:
        return new_section + "\n" + state_text


# =============================================================================
# Helper Utilities (Annotation & Interaction)
# =============================================================================

def get_warp_annotations(warp_memory: dict, map_memory: dict, current_map_name: str, map_current: list) -> dict:
    """Generates annotations for warp points based on memory and grid layout."""
    annotations = {}
    raw_warps = warp_memory.get(current_map_name, {}) if current_map_name else {}
    
    for pos, dest_map in raw_warps.items():
        # Heuristic: Snap landing position to visible WarpPoint if adjacent (usually Up)
        display_pos = pos
        is_valid_warp = True
            
        if map_current:
            x, y = pos
            curr_cell = ""
            if 0 <= y < len(map_current) and 0 <= x < len(map_current[0]):
                curr_cell = str(map_current[y][x])
            
            # Check if current cell is a known warp type
            is_current_warp = "WARP" in curr_cell.upper() or "DOOR" in curr_cell.upper() or curr_cell == "WarpPoint"
            
            if not is_current_warp:
                # If current cell is definitively NOT a warp (e.g. generic walkable), check neighbor
                snapped = False
                # Check Up (y-1)
                if 0 <= y-1 < len(map_current) and 0 <= x < len(map_current[0]):
                    up_cell = str(map_current[y-1][x])
                    if "WARP" in up_cell.upper() or "DOOR" in up_cell.upper() or up_cell == "WarpPoint":
                        display_pos = (x, y-1)
                        snapped = True
                
                # If we didn't snap and the current cell is a generic tile, this is likely a false positive backward link.
                if not snapped:
                     # Check if on map edge (often valid bidirectional warps)
                     width = len(map_current[0]) if len(map_current) > 0 else 0
                     height = len(map_current)
                     is_edge = (x == 0) or (y == 0) or (x == width - 1) or (y == height - 1)

                     # List of characters that are definitely NOT warps (e.g., standard floor, wall, etc.)
                     # If it's a '?' we give benefit of doubt.
                     # However, if it is on the edge, we assume it's a valid map transition warp.
                     if curr_cell in ['O', 'G', 'X', '-', '|', ' '] and not is_edge:
                         is_valid_warp = False

        if is_valid_warp:
            status = " (Unexplored)"
            if dest_map in map_memory:
                grid = map_memory[dest_map].get("explored_map", [])
                if grid:
                    # Check if there's any '?' row by row
                    is_partial = False
                    for row in grid:
                        if '?' in row:
                            is_partial = True
                            break
                    status = " (Partially Explored)" if is_partial else " (Fully Explored)"
            annotations[display_pos] = f"{dest_map}{status}"
        
    return annotations


def get_npc_annotations(map_memory: dict, current_map_name: str, map_current: list) -> dict:
    """Scan the map for known NPCs and create annotations with their summary."""
    annotations = {}
    if current_map_name and "npcs" in map_memory.get(current_map_name, {}):
        npc_memory = map_memory[current_map_name]["npcs"]
        # Scan grid to find current positions of these NPCs
        if map_current:
            for y, row in enumerate(map_current):
                for x, cell in enumerate(row):
                    if isinstance(cell, str) and cell in npc_memory:
                        # Found a sprite with memory
                        mem_item = npc_memory[cell]
                        text_show = "..."
                        if isinstance(mem_item, dict):
                            text_show = mem_item.get("summary", "...")
                        elif isinstance(mem_item, str):
                            text_show = mem_item[:50] + "..." if len(mem_item) > 50 else mem_item
                            
                        annotations[(x, y)] = f"{cell} (Said: '{text_show}')"
    return annotations


def detect_npc_interaction(parsed_state: dict, map_current: list) -> str:
    """
    Determines if the player is currently confirming dialog from a sprite.
    Returns the sprite ID if found, else None.
    """
    screen_text = parsed_state.get("screen_text")
    pos = parsed_state.get("pos")
    facing = parsed_state.get("facing")
    
    if screen_text and pos and facing and map_current:
        # Calculate target position
        tx, ty = pos
        if facing == "up": ty -= 1
        elif facing == "down": ty += 1
        elif facing == "left": tx -= 1
        elif facing == "right": tx += 1
        
        # Check bounds and retrieve object
        if 0 <= ty < len(map_current) and 0 <= tx < len(map_current[0]):
            target_obj = map_current[ty][tx]
            # If it looks like a sprite, return it
            if isinstance(target_obj, str) and target_obj.startswith("SPRITE_"):
                return target_obj
    return None
