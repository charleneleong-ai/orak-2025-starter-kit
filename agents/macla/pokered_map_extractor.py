"""Auto-extract MAP_GRAPH + EXIT_TILES from pokered's .asm map metadata.

The hand-authored ``MAP_GRAPH`` in :mod:`agents.macla.macla_lib` only
covers M1-M6 territory (14 maps out of pokered's 224 headers). This
module parses the .asm source-of-truth so later-game stages get
adjacency for free.

Two .asm conventions are consumed:

1. ``headers/<Map>.asm`` — outdoor map-to-map connections plus the
   direction agent must walk::

       map_header Route1, ROUTE_1, OVERWORLD, NORTH | SOUTH
       connection north, ViridianCity, VIRIDIAN_CITY, -5
       connection south, PalletTown, PALLET_TOWN, 0
       end_map_header

2. ``objects/<Map>.asm`` — indoor warps (doors/stairs) plus the exact
   ``(x, y)`` exit tile::

       def_warp_events
       warp_event  5,  5, REDS_HOUSE_1F, 1
       warp_event 13,  5, BLUES_HOUSE, 1

The two files use different naming conventions for the same map.
Header CamelCase: ``RedsHouse1F``. Warp SCREAMING_SNAKE:
``REDS_HOUSE_1F``. Game observation: ``RedsHouse1f`` (lowercase floor
suffix). All references resolve through :func:`snake_to_canonical`
which targets the game-observation format so the auto graph composes
cleanly with the runtime ``_extract_map_name`` output.

Not wired into the runtime yet. The current ``MAP_GRAPH`` constant
stays the source-of-truth until the Stage P n=5 verdict determines
whether expanding coverage is the right intervention.
"""

from __future__ import annotations

import re
from pathlib import Path

_CONNECTION_RE = re.compile(
    r"^\s*connection\s+(\w+)\s*,\s*\w+\s*,\s*(\w+)\s*,\s*-?\d+",
    re.MULTILINE,
)
_WARP_RE = re.compile(
    r"^\s*warp_event\s+(\d+)\s*,\s*(\d+)\s*,\s*(\w+)\s*,\s*\d+",
    re.MULTILINE,
)
_MAP_HEADER_RE = re.compile(
    r"^\s*map_header\s+\w+\s*,\s*(\w+)\s*,",
    re.MULTILINE,
)


def snake_to_canonical(snake: str) -> str | None:
    """Convert SCREAMING_SNAKE_CASE → game-observation CamelCase.

    Returns ``None`` for ``LAST_MAP``, the runtime sentinel meaning
    "whichever map you entered from" (no static edge to capture).

    Floor suffixes like ``1F``/``2F`` lowercase to ``1f``/``2f`` to
    match the game's observation format (see ``map_names.json``)::

        >>> snake_to_canonical("REDS_HOUSE_1F")
        'RedsHouse1f'
        >>> snake_to_canonical("ROUTE_22")
        'Route22'
        >>> snake_to_canonical("LAST_MAP") is None
        True
    """
    if snake == "LAST_MAP":
        return None
    parts = []
    for part in snake.split("_"):
        # Floor suffixes (digit + single letter, eg "1F", "2F", "3F")
        if len(part) == 2 and part[0].isdigit() and part[1].isalpha():
            parts.append(part.lower())
        else:
            parts.append(part.title())
    return "".join(parts)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _map_name_from_header(path: Path) -> str | None:
    """Read the canonical map name from a ``headers/<Map>.asm`` file.

    Falls back to the filename stem if the header omits ``map_header``
    (which happens for stub/empty files)."""
    text = _read(path)
    m = _MAP_HEADER_RE.search(text)
    if m:
        return snake_to_canonical(m.group(1))
    # Fallback: convert the filename stem (CamelCase already) by
    # lowercasing floor suffixes.
    stem = path.stem
    # Match a trailing single-digit + capital-letter floor suffix
    return re.sub(r"(\d)([A-Z])$", lambda m: m.group(1) + m.group(2).lower(), stem)


def parse_connections(header_path: Path) -> dict[str, str]:
    """Parse outdoor connections from a header .asm file.

    Returns ``{direction: neighbour_canonical_name}``."""
    text = _read(header_path)
    out: dict[str, str] = {}
    for m in _CONNECTION_RE.finditer(text):
        direction = m.group(1).lower()
        neighbour = snake_to_canonical(m.group(2))
        if neighbour:
            out[direction] = neighbour
    return out


def parse_warps(object_path: Path) -> list[tuple[int, int, str]]:
    """Parse indoor warp events from an objects .asm file.

    Returns a list of ``(x, y, target_canonical_name)``. LAST_MAP
    sentinels are dropped — they're runtime back-references, not
    static edges."""
    text = _read(object_path)
    out: list[tuple[int, int, str]] = []
    for m in _WARP_RE.finditer(text):
        x, y = int(m.group(1)), int(m.group(2))
        target = snake_to_canonical(m.group(3))
        if target:
            out.append((x, y, target))
    return out


def build_map_graph(pokered_root: Path) -> dict[str, set[str]]:
    """Build the full pokemon_red MAP_GRAPH from .asm files.

    Edges are bidirectional — for every parsed A→B, the B→A reverse
    is added so the dict is symmetric (matching the hand-authored
    convention)."""
    headers_dir = pokered_root / "data/maps/headers"
    objects_dir = pokered_root / "data/maps/objects"

    graph: dict[str, set[str]] = {}

    def _add(src: str, dst: str) -> None:
        graph.setdefault(src, set()).add(dst)
        graph.setdefault(dst, set()).add(src)

    for header in sorted(headers_dir.glob("*.asm")):
        src = _map_name_from_header(header)
        if not src:
            continue
        for _direction, neighbour in parse_connections(header).items():
            _add(src, neighbour)

        # Indoor warps live in the matching objects file. Filenames
        # match the header filename exactly.
        objects = objects_dir / header.name
        if objects.is_file():
            for _x, _y, target in parse_warps(objects):
                _add(src, target)

    return graph


def build_exit_tiles(
    pokered_root: Path,
) -> dict[tuple[str, str], tuple[int, int] | str]:
    """Build a ``(src_map, dst_map) → exit_info`` dict.

    Exit info is either:
        * ``(x, y)`` tile — for indoor warps (door/stair coordinates)
        * direction string (``"north"``, ``"south"``, ...) — for
          outdoor connections where the agent has to walk off the
          map edge

    Indoor warps take precedence over outdoor connections when both
    exist for the same (src, dst) pair (rare; agent prefers the
    precise tile when given a choice).
    """
    headers_dir = pokered_root / "data/maps/headers"
    objects_dir = pokered_root / "data/maps/objects"

    exits: dict[tuple[str, str], tuple[int, int] | str] = {}

    for header in sorted(headers_dir.glob("*.asm")):
        src = _map_name_from_header(header)
        if not src:
            continue
        # Outdoor connections first (lower precedence).
        for direction, neighbour in parse_connections(header).items():
            exits.setdefault((src, neighbour), direction)
        # Indoor warps overwrite (higher precedence — precise tile).
        objects = objects_dir / header.name
        if objects.is_file():
            for x, y, target in parse_warps(objects):
                exits[(src, target)] = (x, y)

    return exits
