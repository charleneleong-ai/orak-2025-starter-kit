import threading
import time
from pyboy import PyBoy
from pyboy.utils import WindowEvent

import os
import re
import glob
import json
import importlib.util

frame_time = 0.01
game_code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_map_module(map_name):
    map_dir = os.path.join(game_code_dir, "game", "processed_map")
    map_filename = f"{map_name}.py"
    path = os.path.join(map_dir, map_filename)

    if not os.path.exists(path):
        path_not_found = True
        for filename in os.listdir(map_dir):
            if filename.lower() == map_filename.lower():
                path = os.path.join(map_dir, filename)
                path_not_found = False
                break
        
        if path_not_found:
            print(f"[WARN] Map module not found: {path}")
            return None, None, None, None

    spec = importlib.util.spec_from_file_location(map_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.tile_type, mod.map_connection, mod.tile_map, mod.coll_map

def _resolve_asm_path(asm_path: str) -> str | None:
    """Find an asm file on disk, tolerating case mismatches.

    pokered's checkout uses CamelCase with capital floor suffixes
    (``RedsHouse1F.asm``, ``Museum2F.asm``) but our ``map_names.json``
    snapshots the runtime map names with lowercase ``f``
    (``RedsHouse1f``). Linux is case-sensitive, so a direct lookup
    misses 76 of 248 maps and the renderer falls back to ``OBJ_X_Y``
    placeholders — which is exactly the failure mode that kept the
    Stage A pokemon agent from telling a Pokeball from a bookshelf
    in Oak's lab.

    Returns the resolved path (which may differ in case from the
    requested ``asm_path``), or ``None`` if no case-insensitive match
    exists in the directory.
    """
    if os.path.exists(asm_path):
        return asm_path
    asm_dir = os.path.dirname(asm_path)
    target = os.path.basename(asm_path).lower()
    if not os.path.isdir(asm_dir):
        return None
    for entry in os.listdir(asm_dir):
        if entry.lower() == target:
            return os.path.join(asm_dir, entry)
    return None


def _require_asm_files(asm_dir: str) -> None:
    """Hard-fail if the pokered asm dir is missing or empty.

    Every Pokemon experiment in this repo before 2026-05-14 silently
    fell back to ``OBJ_n_n`` placeholders that wedged Stage J in
    OaksLab. See ``docs/experiments/pokemon-asm-gap.md``.
    """
    try:
        with os.scandir(asm_dir) as it:
            if any(entry.name.endswith(".asm") for entry in it):
                return
    except (FileNotFoundError, NotADirectoryError):
        pass
    pokered_root = os.path.normpath(os.path.join(asm_dir, "..", "..", ".."))
    raise RuntimeError(
        f"Pokemon Red object-sprite asm files missing at {asm_dir!r}. "
        f"Without these the harness emits OBJ_n_n placeholders instead of "
        f"real sprite names (SPRITE_OAK, SPRITE_POKE_BALL, ...). Run:\n"
        f"  git clone --depth 1 https://github.com/pret/pokered.git {pokered_root}\n"
        f"See docs/experiments/pokemon-asm-gap.md."
    )


def parse_object_sprites(asm_path):
    resolved = _resolve_asm_path(asm_path)
    if resolved is None:
        print(f"[WARN] asm not found: {asm_path}")
        return []

    sprite_names = []
    pattern = re.compile(r"object_event\s+\d+,\s*\d+,\s*([A-Z0-9_]+)")

    with open(resolved, encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                sprite = match.group(1)
                sprite_names.append(sprite)
    return sprite_names

class PyBoyRunner:
    def __init__(self, rom_path, state_path=None):
        self.pyboy = PyBoy(rom_path, window="null")  # Run without GUI
        self.running = True
        self.lock = threading.Lock()
        
        # Load state file if provided or if it exists with same name as ROM
        if state_path is None:
            state_path = rom_path + ".state"
        
        if os.path.exists(state_path):
            print(f"[PyBoy] Loading saved state from: {state_path}")
            with open(state_path, "rb") as f:
                self.pyboy.load_state(f)
        else:
            print(f"[PyBoy] No saved state found at: {state_path}")

        self.json_dir = os.path.join(game_code_dir, "game", "mapping_json")
        self.asm_dir = os.path.join(game_code_dir, "game", "pokered", "data", "maps", "objects")
        _require_asm_files(self.asm_dir)

        self.species_names = load_json(os.path.join(self.json_dir, "species_names.json"))
        self.type_names = load_json(os.path.join(self.json_dir, "type_names.json"))
        self.map_names = load_json(os.path.join(self.json_dir, "map_names.json"))
        self.charmap = load_json(os.path.join(self.json_dir, "charmap.json"))
        self.item_names = load_json(os.path.join(self.json_dir, "item_names.json"))
        self.move_names = load_json(os.path.join(self.json_dir, "move_names.json"))

        self.quit_flag = False

    def tick_loop(self):
        frame_counter = 0
        while self.running:
            with self.lock:
                self.pyboy.tick()
            time.sleep(frame_time)
            frame_counter += 1
        self.pyboy.stop()

    def press_and_release(self, event_down, event_up, hold_frames=10):
        self.pyboy.send_input(event_down)
        for _ in range(hold_frames):
            self.pyboy.tick()
            time.sleep(frame_time)
        self.pyboy.send_input(event_up)
        for _ in range(2):
            self.pyboy.tick()
            time.sleep(frame_time)

    def send_input(self, action):
        action = action.lower()
        with self.lock:
            if action == "a":
                self.press_and_release(WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A)
            elif action == "b":
                self.press_and_release(WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B)
            elif action == "start":
                self.press_and_release(WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START)
            elif action == "select":
                self.press_and_release(WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT)
            elif action == "up":
                self.press_and_release(WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP)
            elif action == "down":
                self.press_and_release(WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)
            elif action == "left":
                self.press_and_release(WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT)
            elif action == "right":
                self.press_and_release(WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)
            elif action == "quit":
                self.quit_flag = True
                return
            elif action == "pass":
                pass
            else:
                print(f"[WARN] Unknown action: '{action}'")

            # Time to progress the game
            for _ in range(100):
                self.pyboy.tick()
                time.sleep(frame_time)

    def take_screenshot(self, dir_name, img_name):

        os.makedirs(dir_name, exist_ok=True)

        # Ensure filename ends with .png
        if not img_name.endswith('.png'):
            img_name += '.png'

        # Save the new screenshot
        img = self.pyboy.screen.image
        img.save(os.path.join(dir_name, img_name))
        return img
    
    def decode_tilemap(self):
        mem = self.pyboy.memory
        TILEMAP_ADDR = 0xC3A0
        SCREEN_WIDTH = 20
        SCREEN_HEIGHT = 18
        charmap = self.charmap

        tile_lines = []
        for row in range(SCREEN_HEIGHT):
            line = []
            for col in range(SCREEN_WIDTH):
                addr = TILEMAP_ADDR + row * SCREEN_WIDTH + col
                byte = mem[addr]
                ch = charmap.get(str(byte), " ")
                if ch == "<NULL>":
                    ch = " "
                line.append(ch)
            tile_lines.append(line)
        return tile_lines

    def get_filtered_screen_text(self, tile_lines):
        lines = []
        for line in tile_lines:
            line_str = "".join(line).strip() # Without blank line
            if "QRSTUVWXYZ():;[]" in line_str:
                return "N/A"
            if line_str:
                lines.append(line_str)
        return "\n".join(lines) if lines else "N/A"

    def find_selection_box(self, tile_lines):
        height = len(tile_lines)
        width = len(tile_lines[0]) if tile_lines else 0

        # 1. Find ▶ cursor
        cursor_y, cursor_x = None, None
        for y_idx, row_chars in enumerate(tile_lines):
            for x_idx, char_val in enumerate(row_chars):
                if char_val == "▶":
                    cursor_y, cursor_x = y_idx, x_idx
                    break
            if cursor_y is not None:
                break
        if cursor_y is None:
            return None

        # 2. Find vertical │ borders for the line containing the cursor
        def find_vertical_border_char_x(line_chars, start_x_coord, direction_step):
            current_x = start_x_coord
            while 0 <= current_x < len(line_chars):
                if line_chars[current_x] == "│":
                    return current_x
                current_x += direction_step
            return None

        x1 = find_vertical_border_char_x(tile_lines[cursor_y], cursor_x, -1)
        x2 = find_vertical_border_char_x(tile_lines[cursor_y], cursor_x, 1)

        if x1 is None or x2 is None or x2 <= x1:
            return None

        # 3. Find horizontal borders (y1 and y2) using the Lua-like logic
        
        # Find y1 (upwards)
        found_y1 = None
        for y_scan in range(cursor_y - 1, -1, -1):
            # Check if tile_lines[y_scan] is long enough
            if len(tile_lines[y_scan]) <= x1 or len(tile_lines[y_scan]) <= x2-1 :
                continue # Line is too short

            is_line_with_dash = False
            for x_scan in range(x1 + 1, x2):
                if tile_lines[y_scan][x_scan] == "─":
                    is_line_with_dash = True
                    break
            if is_line_with_dash:
                found_y1 = y_scan
                break
        
        # Find y2 (downwards)
        found_y2 = None
        for y_scan in range(cursor_y + 1, height):
            # Check if tile_lines[y_scan] is long enough
            if len(tile_lines[y_scan]) <= x1 or len(tile_lines[y_scan]) <= x2-1:
                continue # Line is too short
                
            is_line_with_dash = False
            for x_scan in range(x1 + 1, x2):
                if tile_lines[y_scan][x_scan] == "─":
                    is_line_with_dash = True
                    break
            if is_line_with_dash:
                found_y2 = y_scan
                break

        if found_y1 is None or found_y2 is None:
            # This can happen if the box is at the very edge of the screen
            # or if the horizontal borders don't use '─'
            # print(f"Debug: Lua-like horizontal borders not found. y1={found_y1}, y2={found_y2}")
            
            # Fallback to check for explicit corners if Lua-like borders fail
            # This part is from the previous Pythonic suggestion
            # Check for y1 with corners
            y1_corner_fallback = None
            for y_s in range(cursor_y - 1, -1, -1):
                line = tile_lines[y_s]
                if len(line) > x2 and line[x1] in ['┌','├'] and line[x2] in ['┐','┤'] and all(c=='─' for c in line[x1+1:x2]):
                    y1_corner_fallback = y_s
                    break
            if found_y1 is None: found_y1 = y1_corner_fallback
            if found_y1 is None and len(tile_lines[0]) > x2 and tile_lines[0][x1] in ['┌','├'] and tile_lines[0][x2] in ['┐','┤'] and all(c=='─' for c in tile_lines[0][x1+1:x2]):
                found_y1 = 0


            # Check for y2 with corners
            y2_corner_fallback = None
            for y_s in range(cursor_y + 1, height):
                line = tile_lines[y_s]
                if len(line) > x2 and line[x1] in ['└','├'] and line[x2] in ['┘','┤'] and all(c=='─' for c in line[x1+1:x2]):
                    y2_corner_fallback = y_s
                    break
            if found_y2 is None: found_y2 = y2_corner_fallback
            if found_y2 is None and len(tile_lines[height-1]) > x2 and tile_lines[height-1][x1] in ['└','├'] and tile_lines[height-1][x2] in ['┘','┤'] and all(c=='─' for c in tile_lines[height-1][x1+1:x2]):
                found_y2 = height -1

            if found_y1 is None or found_y2 is None:
                 return None # Still not found after fallbacks

        return {"x1": x1, "x2": x2, "y1": found_y1, "y2": found_y2}

    def extract_selection_box_text(self, tile_lines, box):
        lines = []
        for y in range(box["y1"] + 1, box["y2"]):
            line = tile_lines[y][box["x1"] + 1 : box["x2"]]
            line_str = "".join(line).strip()
            if line_str:
                lines.append(line_str)
        return lines

    def get_dialog(self):
        tile_lines = self.decode_tilemap()

        # [Filtered Screen Text]
        text = "[Filtered Screen Text]\n"
        text += self.get_filtered_screen_text(tile_lines) + "\n"

        # [Selection Box Text]
        text += "\n[Selection Box Text]\n"
        box = self.find_selection_box(tile_lines)
        if box:
            lines = self.extract_selection_box_text(tile_lines, box)
            text += "----------------\n"
            for line in lines:
                text += line + "\n"
            text += "----------------\n"
        else:
            text += "N/A\n"

        return text

    def get_inventory(self):
        mem = self.pyboy.memory
        item_names = self.item_names

        item_count = mem[0xD31D]
        base = 0xD31E

        text = "[Bag]\n"
        text += f"({item_count} items):\n" if item_count > 0 else "N/A\n"

        for i in range(item_count):
            addr = base + i * 2
            item_id = mem[addr]
            quantity = mem[addr + 1]

            if item_id == 0:
                continue  # Empty slot

            name = item_names.get(str(item_id), f"Unknown Item (ID:{item_id:02X})")
            text += f"- {name} × {quantity}\n"

        return text
    
    def get_map_visual(self, coll_map, player_x, player_y, object_coords):
        warp_dests = self.get_warp_destinations()
        lines = []
        for sy in range(max(0, player_y - 4), player_y + 5):
            line = ""
            for sx in range(max(0, player_x - 4), player_x + 5):
                key = (sx, sy)
                if key in object_coords:
                    cell = object_coords[key]
                elif coll_map and sy < len(coll_map) and sx < len(coll_map[sy]):
                    cell = self._enrich_warp_label(coll_map[sy][sx], key, warp_dests)
                else:
                    cell = "?"
                line += f"({sx:2d},{sy:2d}): {cell}\t"
            lines.append(line)
        return "\n".join(lines)

    def get_warp_destinations(self) -> dict[tuple[int, int], str]:
        """Map every warp tile on the current map to its destination map name.

        Reads the active wWarpEntries table from PyBoy memory:

          wNumberOfWarps  @ 0xD3AE  (1 byte, count, max 32)
          wWarpEntries    @ 0xD3AF  (4 bytes per entry: y, x, dest_warp_id, dest_map_id)

        Returns ``{(x, y): dest_map_name}``. Resolves the trick value
        ``dest_map_id == 0xFF`` ("use last entered map", i.e. door warps that
        return the player to wherever they came from) to ``"PrevMap"``.

        Without this enrichment, all warp tiles render as the bare string
        ``WarpPoint``, leaving the agent unable to distinguish a staircase
        (``Warp→RedsHouse2f``) from an exit door (``Warp→PalletTown``) — see
        the audit in commit f98ed6b for the failure-mode that prompted this.
        """
        mem = self.pyboy.memory
        n_warps = mem[0xD3AE]
        # Game Boy memory occasionally returns garbage for uninitialised
        # regions; clamp defensively.
        if n_warps > 32:
            n_warps = 0
        out: dict[tuple[int, int], str] = {}
        for i in range(n_warps):
            base = 0xD3AF + i * 4
            wy = mem[base]
            wx = mem[base + 1]
            # mem[base + 2] is dest_warp_id, unused for the label
            dest_map_id = mem[base + 3]
            if dest_map_id == 0xFF:
                dest_name = "PrevMap"
            else:
                dest_name = self.map_names.get(
                    str(dest_map_id), f"UNKNOWN_{dest_map_id}"
                )
            out[(wx, wy)] = dest_name
        return out

    @staticmethod
    def _enrich_warp_label(cell: str, coord: tuple[int, int],
                           warp_dests: dict[tuple[int, int], str]) -> str:
        """If ``cell`` is the bare 'WarpPoint' marker and ``coord`` resolves to
        a destination via ``warp_dests``, return ``f"Warp→{dest}"`` instead.
        Otherwise return ``cell`` unchanged."""
        if cell == "WarpPoint" and coord in warp_dests:
            return f"Warp→{warp_dests[coord]}"
        return cell

    def get_object_coords(self, player_x, player_y):
        mem = self.pyboy.memory
        object_coords = {}

        map_id = mem[0xD35E]
        map_name = self.map_names.get(str(map_id), f"UNKNOWN_{map_id}")

        asm_path = os.path.join(self.asm_dir, f"{map_name}.asm")
        sprite_list = parse_object_sprites(asm_path)

        for i in range(1, 16):
            base = 0xC100 + i * 16
            if mem[base + 2] == 0xFF:
                continue

            y_delta = ((mem[base + 4] + 4) % 256 - (mem[0xC100 + 4] + 4)) // 16
            x_delta = (mem[base + 6] - mem[0xC100 + 6]) // 16

            obj_x = player_x + x_delta
            obj_y = player_y + y_delta

            sprite_name = sprite_list[i-1] if i-1 < len(sprite_list) else f"OBJ_{i}"
            object_coords[(obj_x, obj_y)] = f"{sprite_name}_{i}"

        return object_coords

    def get_player_pos(self):
        mem = self.pyboy.memory
        map_id = mem[0xD35E]
        map_name = self.map_names.get(str(map_id), f"UNKNOWN_{map_id}")
        player_x = mem[0xD362]
        player_y = mem[0xD361]
        return (player_x, player_y, map_name)


    def get_map_info(self):
        mem = self.pyboy.memory
        map_id = mem[0xD35E]
        map_name = self.map_names.get(str(map_id), f"UNKNOWN_{map_id}")

        max_width = mem[0xD369] * 2 - 1
        max_height = mem[0xD368] * 2 - 1
        player_x = mem[0xD362]
        player_y = mem[0xD361]
        direction_code = mem[0xC109]

        direction_map = {0: "down", 4: "up", 8: "left", 12: "right"}
        facing = direction_map.get(direction_code, "None")

        # Load map module
        tile_type, map_connection, tile_map, coll_map = load_map_module(map_name)

        # Extract object's location
        object_coords = self.get_object_coords(player_x, player_y)

        text = "[Map Info]\n"
        text += f"Map Name: {map_name}, (x_max , y_max): ({max_width}, {max_height})\n"
        text += f"Map type: {tile_type or 'UNKNOWN'}\n"
        text += f"Expansion direction: {map_connection or 'None'}\n"
        text += f"Your position (x, y): ({player_x}, {player_y})\n"
        text += f"Your facing direction: {facing}\n"
        text += "Action instruction\n"
        text += " - up: (x, y) -> (x, y-1)\n"
        text += " - down: (x, y) -> (x, y+1)\n"
        text += " - left: (x, y) -> (x-1, y)\n"
        text += " - right: (x, y) -> (x+1, y)\n"

        text += "\nMap on Screen:\n"
        if self.get_battle_state() != 'Field':
            text += "Not in Field State"
            return text
        warp_dests = self.get_warp_destinations()
        for sy in range(max(0, player_y - 4), min(player_y + 4, max_height) + 1):
            for sx in range(max(0, player_x - 4), min(player_x + 5, max_width) + 1):
                key = (sx, sy)
                if key in object_coords:
                    cell = object_coords[key]
                elif coll_map and sy < len(coll_map) and sx < len(coll_map[sy]):
                    cell = self._enrich_warp_label(coll_map[sy][sx], key, warp_dests)
                else:
                    cell = "?"
                text += f"({sx:2d}, {sy:2d}): {cell}\t"
            text += "\n"

        return text
    
    # def get_battle_state(self):
    #     mem = self.pyboy.memory
    #     wIsInBattle = mem[0xD057]
    #     wLinkState = mem[0xD72E]
    #     title_check = mem[0xC0EF]
    #     map_id = mem[0xD35C]

    #     if (title_check == 0x1F and map_id == 0x00) or title_check is None:
    #         return "Title"
    #     elif wLinkState == 0x05:
    #         return "LinkBattle"
    #     elif wIsInBattle == 0x01:
    #         return "WildBattle"
    #     elif wIsInBattle == 0x02:
    #         return "TrainerBattle"
    #     else:
    #         return "Field"
        
    def get_battle_state(self):
        mem = self.pyboy.memory
        wIsInBattle = mem[0xD057]
        wLinkState = mem[0xD72E]
        title_check = mem[0xC0EF]
        map_id = mem[0xD35C]

        if (title_check == 0x1F and map_id == 0x00) or title_check is None:
            battle_state = "Title"
        elif wLinkState == 0x05:
            battle_state = "LinkBattle"
        elif wIsInBattle == 0x01:
            battle_state = "WildBattle"
        elif wIsInBattle == 0x02:
            battle_state = "TrainerBattle"
        else:
            battle_state = "Field"
        
        dialog_text = self.get_dialog()
        has_dialog = "[Filtered Screen Text]\nN/A" not in dialog_text

        if has_dialog and battle_state == "Field":
            if 'CONTINUE' in dialog_text and 'NEW GAME' in dialog_text:
                battle_state = "Title"
            else:
                battle_state = "Dialog"
        
        return battle_state

    def get_enemy_info(self):
        mem = self.pyboy.memory
        species_id = mem[0xCFE5]
        level = mem[0xCFF3]
        hp = (mem[0xCFE6] << 8) + (mem[0xCFE7])
        max_hp = (mem[0xCFF4] << 8) + (mem[0xCFF5])
        status = mem[0xCFE9]

        battle_state = self.get_battle_state()
        text = "\n[Enemy Pokemon]\n"

        if battle_state in ["WildBattle", "TrainerBattle", "LinkBattle"]:
            name = self.species_names.get(str(species_id), f"UNKNOWN_{species_id}")
            text += f"Name: {name}\n"
            text += f"Level: {level}\n"

            if max_hp > 0:
                hp_pct = int((hp / max_hp) * 100)
                text += f"HP_percentage: {hp_pct}%\n"
            else:
                text += f"HP_percentage: Unknown\n"

            status_text = {
                0: "Normal state",
                1: "Sleep", 2: "Sleep", 3: "Sleep", 4: "Sleep", 5: "Sleep", 6: "Sleep", 7: "Sleep",
                8: "Poisoned",
                16: "Burned",
                32: "Frozen",
                64: "Paralyzed"
            }.get(status, "Normal state")
            text += f"Status: {status_text}\n"

        else:
            text += "- Not in battle\n"

        return text
    
    def get_active_pokemon_name(self):
        mem = self.pyboy.memory
        active_name = ""
        for i in range(11):
            b = mem[0xD009 + i]
            ch = self.charmap.get(str(b), "")
            if ch in ["<NULL>", "@"]:
                break
            active_name += ch
        return active_name

    def get_party_info(self):
        mem = self.pyboy.memory
        active_name = self.get_active_pokemon_name()
        text = "\n[Current Party]\n"

        for i in range(6):
            base = 0xD16B + i * 0x2C
            species_id = mem[base]
            if species_id == 0:
                text += "No more Pokemons\n"
                break

            level = mem[base + 0x21]
            hp = (mem[base + 1] << 8) + mem[base + 2]
            max_hp = (mem[base + 0x22] << 8) + mem[base + 0x23]

            # nickname
            name_bytes = mem[0xD2B5 + i * 11 : 0xD2B5 + (i + 1) * 11]
            nickname = ""
            for b in name_bytes:
                ch = self.charmap.get(str(b), "")
                if ch in ["<NULL>", "@"]:
                    break
                nickname += ch

            species_name = self.species_names.get(str(species_id), f"UNKNOWN_{species_id}")
            type1 = self.type_names.get(str(mem[base + 0x05]), f"UNKNOWN_{mem[base + 0x05]}")
            type2 = self.type_names.get(str(mem[base + 0x06]), f"UNKNOWN_{mem[base + 0x06]}")

            # moves and PP
            moves = []
            for j in range(4):
                move_id = mem[0xD173 + i * 0x2C + j]
                pp = mem[0xD188 + i * 0x2C + j]
                move_name = self.move_names.get(str(move_id), "Not Learned")
                moves.append(f"{move_name}(pp={pp})")
            
            status = mem[base + 4]
            status_text = {
                0: "Normal state",
                1: "Sleep", 2: "Sleep", 3: "Sleep", 4: "Sleep", 5: "Sleep", 6: "Sleep", 7: "Sleep",
                8: "Poisoned",
                16: "Burned",
                32: "Frozen",
                64: "Paralyzed"
            }.get(status, "Normal state")

            prefix = "[In-battle] Name:" if nickname == active_name and 'Battle' in self.get_battle_state() else "Name: "
            if type1 != type2:
                text += f"{prefix}{nickname}, Species: {species_name}, Level: {level}, Status: {status_text}, Type: {type1}/{type2}, HP: {hp}/{max_hp}"
            else:
                text += f"{prefix}{nickname}, Species: {species_name}, Level: {level}, Status: {status_text}, Type: {type1}, HP: {hp}/{max_hp}, Moves: "
            text += ", ".join(moves) + "\n"

        return text

    def get_badge_info(self):
        mem = self.pyboy.memory
        badge_mask = mem[0xD356]
        badge_names = [
            "Boulder", "Cascade", "Thunder", "Rainbow",
            "Soul", "Marsh", "Volcano", "Earth"
        ]
        badges = [badge_names[i] for i in range(8) if badge_mask & (1 << i)]

        text = "\n[Badge List]\n"
        text += ", ".join(badges) if badges else "N/A"
        text += "\n"
        return text

    def get_money_info(self):
        mem = self.pyboy.memory
        def bcd_to_int(bcd):
            return (bcd >> 4) * 10 + (bcd & 0xF)
        money = (
            bcd_to_int(mem[0xD347]) * 10000 +
            bcd_to_int(mem[0xD348]) * 100 +
            bcd_to_int(mem[0xD349])
        )
        return f"\n[Current Money]: ¥{money}\n"

    def get_state(self):
        dialog_text = self.get_dialog()
        has_dialog = "[Filtered Screen Text]\nN/A" not in dialog_text

        battle_state = self.get_battle_state()
        if has_dialog and battle_state == "Field":
            battle_state = "Dialog"
        
        parts = []
        parts.append("State: " + battle_state + "\n")
        parts.append(self.get_dialog())
        parts.append(self.get_enemy_info())
        parts.append(self.get_party_info())
        parts.append(self.get_badge_info())
        parts.append(self.get_inventory())
        parts.append(self.get_money_info())
        parts.append(self.get_map_info())

        return "\n".join(parts)
    
    def save_sav_file(self, path):
        with open(path, "wb") as f:
            f.write(self.pyboy.cartridge.savefile)
        print(f"[INFO] .sav File Saved: {path}")
