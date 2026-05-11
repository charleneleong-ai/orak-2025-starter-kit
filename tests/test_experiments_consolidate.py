"""Tests for the experiments consolidator + scoreboard-from-index CLI.

The consolidator aggregates per-sweep ``results.jsonl`` files into a single
canonical index. The new ``scoreboard-from-index`` subcommand renders the
cross-game scoreboard from that index, so we no longer need to split each
chain's results into one tag-dir per bar.
"""

from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ── consolidator ───────────────────────────────────────────────────────


def test_consolidate_aggregates_multiple_sweep_dirs(tmp_path):
    """Given N per-sweep results.jsonl files, output a single jsonl with
    all rows + a `_source_path` field showing where each came from."""
    from experiments._consolidate import consolidate

    _write_jsonl(
        tmp_path / "pr31_pokemon/gemma_26b/results.jsonl",
        [
            {"variant": "stage_a", "game": "pokemon_red", "evaluation_score": 28.57},
            {"variant": "stage_d", "game": "pokemon_red", "evaluation_score": 57.14},
        ],
    )
    _write_jsonl(
        tmp_path / "pr31_2048/gemma_26b/results.jsonl",
        [
            {"variant": "stage_a", "game": "twenty_fourty_eight", "evaluation_score": 54.55},
        ],
    )

    out = tmp_path / "consolidated.jsonl"
    n = consolidate(roots=[tmp_path], output=out)
    assert n == 3
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert {r["variant"] for r in rows} == {"stage_a", "stage_d"}
    assert {r["game"] for r in rows} == {"pokemon_red", "twenty_fourty_eight"}
    assert all("_source_path" in r for r in rows)


def test_consolidate_handles_missing_or_empty_dirs(tmp_path):
    """Empty dirs and missing results.jsonl are skipped silently."""
    from experiments._consolidate import consolidate

    (tmp_path / "empty_sweep" / "gemma_26b").mkdir(parents=True)
    _write_jsonl(
        tmp_path / "real_sweep/gemma_26b/results.jsonl",
        [{"variant": "x", "game": "g", "evaluation_score": 1.0}],
    )

    out = tmp_path / "consolidated.jsonl"
    n = consolidate(roots=[tmp_path], output=out)
    assert n == 1


def test_consolidate_filters_by_tag_pattern(tmp_path):
    """``include_tags`` keeps only sweeps whose dir name matches one of the
    given prefixes — useful for "only consolidate pr31_* sweeps"."""
    from experiments._consolidate import consolidate

    _write_jsonl(
        tmp_path / "pr31_alpha/gemma_26b/results.jsonl",
        [{"variant": "a", "game": "g", "evaluation_score": 1.0}],
    )
    _write_jsonl(
        tmp_path / "legacy_dir/gemma_26b/results.jsonl",
        [{"variant": "b", "game": "g", "evaluation_score": 2.0}],
    )

    out = tmp_path / "consolidated.jsonl"
    n = consolidate(roots=[tmp_path], output=out, include_prefixes=["pr31_"])
    assert n == 1
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert rows[0]["variant"] == "a"


# ── scoreboard-from-index CLI ──────────────────────────────────────────


def test_scoreboard_from_index_renders_variant_filtered_bars(tmp_path):
    """Given an index file, the new CLI subcommand renders one bar per
    --variant within each --game panel."""
    from typer.testing import CliRunner

    from experiments.plot_comparisons import app

    # Build a minimal index file
    index = tmp_path / "results.jsonl"
    rows = [
        {"variant": "stage_a", "game": "pokemon_red", "evaluation_score": 28.57},
        {"variant": "stage_d", "game": "pokemon_red", "evaluation_score": 57.14},
        {"variant": "stage_a", "game": "twenty_fourty_eight", "evaluation_score": 54.55},
        {"variant": "stage_b", "game": "twenty_fourty_eight", "evaluation_score": 63.64},
    ]
    _write_jsonl(index, rows)

    out_png = tmp_path / "scoreboard.png"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scoreboard-from-index",
            "--from-file",
            str(index),
            "--game",
            "pokemon_red",
            "--variant",
            "stage_a",
            "--label",
            "Stage A",
            "--variant",
            "stage_d",
            "--label",
            "Stage D",
            "--sep",
            "2",
            "--game",
            "twenty_fourty_eight",
            "--variant",
            "stage_a",
            "--label",
            "Stage A",
            "--variant",
            "stage_b",
            "--label",
            "Stage B",
            "--sep",
            "2",
            "--out",
            str(out_png),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
    assert out_png.exists()
    assert out_png.stat().st_size > 1000  # non-trivial PNG


def test_scoreboard_from_index_takes_best_score_per_variant(tmp_path):
    """When multiple rows share (game, variant), pick the max score."""
    from typer.testing import CliRunner

    from experiments.plot_comparisons import app

    index = tmp_path / "results.jsonl"
    rows = [
        # Multiple stage_a rows on pokemon_red — best is 28.57
        {"variant": "stage_a", "game": "pokemon_red", "evaluation_score": 14.29},
        {"variant": "stage_a", "game": "pokemon_red", "evaluation_score": 28.57},
        {"variant": "stage_a", "game": "pokemon_red", "evaluation_score": 0.0},
    ]
    _write_jsonl(index, rows)

    out_png = tmp_path / "scoreboard.png"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scoreboard-from-index",
            "--from-file",
            str(index),
            "--game",
            "pokemon_red",
            "--variant",
            "stage_a",
            "--label",
            "Stage A",
            "--sep",
            "1",
            "--out",
            str(out_png),
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
    # Verify the rendered value via a re-import of the helper
    from experiments.plot_comparisons import _best_score_for_variant_from_index

    best = _best_score_for_variant_from_index(rows, game="pokemon_red", variant="stage_a")
    assert best == 28.57
