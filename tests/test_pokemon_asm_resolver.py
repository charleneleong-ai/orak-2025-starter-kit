"""Pin the case-insensitive asm-file resolver.

pokered's repo uses CamelCase with capital floor suffixes
(``RedsHouse1F.asm``, ``Museum2F.asm``, ``MtMoonB1F.asm``), but our
``map_names.json`` snapshots the runtime map names with lowercase ``f``
(``RedsHouse1f``). On Linux the direct lookup misses 76 of 248 maps,
so ``parse_object_sprites`` returned ``[]`` and the renderer fell
back to placeholder ``OBJ_X_Y`` cell labels — see the Stage A pokemon
audit (game_logs/pokemon_red/20260506_221856/) where the agent saw
``OBJ_1_1``/``OBJ_2_2``/``OBJ_3_3`` instead of
``SPRITE_POKE_BALL_2/3/4`` and spammed all of them indiscriminately.

This pins the resolver so the asm checkout's casing convention is
decoupled from the map_names.json convention.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# pyboy_runner does ``from pyboy import PyBoy`` at module top, which we
# don't want to pay for here. Stub the pyboy bindings before loading
# the module so the pure-stdlib helpers can be imported normally.
_REPO = Path(__file__).resolve().parent.parent
_pyboy_stub = types.ModuleType("pyboy")
_pyboy_stub.PyBoy = object
sys.modules.setdefault("pyboy", _pyboy_stub)
_pyboy_utils_stub = types.ModuleType("pyboy.utils")
_pyboy_utils_stub.WindowEvent = object
sys.modules.setdefault("pyboy.utils", _pyboy_utils_stub)

_spec = importlib.util.spec_from_file_location(
    "pyboy_runner_under_test",
    _REPO / "evaluation_utils/mcp_game_servers/pokemon_red/game/pyboy_runner.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_resolve = _module._resolve_asm_path
_require_asm_files = _module._require_asm_files


def test_returns_path_unchanged_when_exact_match_exists(tmp_path):
    target = tmp_path / "PalletTown.asm"
    target.write_text("# stub\n")
    assert _resolve(str(target)) == str(target)


def test_finds_case_mismatched_floor_suffix(tmp_path):
    """The 76-map mismatch — pokered uses ``RedsHouse1F.asm``, our
    map_names.json says ``RedsHouse1f``. Pre-fix this returned None."""
    actual = tmp_path / "RedsHouse1F.asm"
    actual.write_text("object_event 5, 4, SPRITE_MOM, ...\n")
    requested = tmp_path / "RedsHouse1f.asm"
    assert _resolve(str(requested)) == str(actual)


def test_finds_case_mismatched_basement_suffix(tmp_path):
    """``MtMoonB1F.asm`` vs ``MtMoonB1f`` — the basement variant of
    the same casing trap. Same handling."""
    actual = tmp_path / "MtMoonB1F.asm"
    actual.write_text("# stub\n")
    requested = tmp_path / "MtMoonB1f.asm"
    assert _resolve(str(requested)) == str(actual)


def test_returns_none_when_no_case_variant_exists(tmp_path):
    """Truly absent file — caller emits the [WARN] line and returns []."""
    requested = tmp_path / "NotARealMap.asm"
    assert _resolve(str(requested)) is None


def test_returns_none_when_directory_does_not_exist(tmp_path):
    """Defensive: asm_dir wasn't ever set up. Don't crash on listdir."""
    requested = tmp_path / "missing_subdir" / "PalletTown.asm"
    assert _resolve(str(requested)) is None


def test_picks_first_match_when_multiple_case_variants_exist(tmp_path):
    """Edge case: someone has both ``Foo.asm`` and ``foo.asm`` in the
    same dir (impossible on case-insensitive filesystems but possible
    on Linux). Both are valid matches — we just need to pick one
    deterministically and not crash."""
    (tmp_path / "Foo.asm").write_text("variant 1\n")
    (tmp_path / "foo.asm").write_text("variant 2\n")
    requested = tmp_path / "FOO.asm"
    resolved = _resolve(str(requested))
    assert resolved is not None
    assert resolved.endswith((".asm",))
    assert Path(resolved).read_text() in ("variant 1\n", "variant 2\n")


# ── End-to-end against the actual map_names.json registry ──────────────


# ── Hard-fail check when pokered/ wasn't cloned ────────────────────────
#
# Pre-2026-05-14 every pokemon run silently fell back to OBJ_n_n
# placeholders because pokered/ was empty; the hard-fail keeps that
# from re-occurring. See docs/experiments/pokemon-asm-gap.md.


@pytest.mark.parametrize(
    "setup",
    [
        lambda d: d / "does_not_exist",  # missing dir
        lambda d: d,  # empty dir
        lambda d: (d / "README.md").write_text("hi") or d,  # only non-asm files
    ],
    ids=["missing", "empty", "no_asm_files"],
)
def test_require_asm_files_raises(tmp_path, setup):
    target = setup(tmp_path)
    with pytest.raises(RuntimeError, match="OBJ_n_n placeholders"):
        _require_asm_files(str(target))


def test_require_asm_files_error_message_is_self_fixing(tmp_path):
    with pytest.raises(RuntimeError) as exc:
        _require_asm_files(str(tmp_path))
    msg = str(exc.value)
    assert "git clone" in msg
    assert "pret/pokered.git" in msg
    assert "docs/experiments/pokemon-asm-gap.md" in msg


def test_require_asm_files_passes_when_asm_present(tmp_path):
    (tmp_path / "OaksLab.asm").write_text("# stub\n")
    _require_asm_files(str(tmp_path))


def test_resolver_handles_every_floor_suffix_pattern(tmp_path):
    """Build a synthetic asm dir using pokered's CamelCase convention
    (Capital F, capital B-prefix) and verify our lowercase-f lookups
    all resolve. Mirrors the 76-file mismatch in the real checkout."""
    cases = [
        ("RedsHouse1F.asm", "RedsHouse1f.asm"),
        ("RedsHouse2F.asm", "RedsHouse2f.asm"),
        ("Museum1F.asm", "Museum1f.asm"),
        ("Museum2F.asm", "Museum2f.asm"),
        ("MtMoon1F.asm", "MtMoon1f.asm"),
        ("MtMoonB1F.asm", "MtMoonB1f.asm"),
        ("MtMoonB2F.asm", "MtMoonB2f.asm"),
        ("RockTunnel1F.asm", "RockTunnel1f.asm"),
        ("Route11Gate1F.asm", "Route11Gate1f.asm"),
        ("Route11Gate2F.asm", "Route11Gate2f.asm"),
    ]
    for actual, _ in cases:
        (tmp_path / actual).write_text("# stub\n")

    for actual, requested in cases:
        resolved = _resolve(str(tmp_path / requested))
        assert resolved == str(tmp_path / actual), f"failed to resolve {requested} → {actual}"
