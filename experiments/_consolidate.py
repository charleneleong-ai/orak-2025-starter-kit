"""Aggregate per-sweep ``results.jsonl`` files into a single canonical index.

The autoresearch contract (CLAUDE.md) puts each sweep's results at
``experiments/<tag>/<config_name>/results.jsonl``. Over a long-running PR
that accumulates dozens of those dirs, and downstream plotting CLIs end up
needing to either iterate every dir or split chain results into one-tag-
per-bar shadow dirs (see the PR #31 ``_stage_a/b/c/d`` shadows).

This module flattens all those per-sweep files into a single
``experiments/<topic>_results.jsonl`` index. Each row gains a
``_source_path`` field pointing back at the originating file so the
provenance survives the merge. The CLI plotting code can then read from
the index and filter by ``game`` / ``variant`` without needing the shadow
dirs.

Usage:
    python -m experiments._consolidate \\
        --root experiments \\
        --include-prefix pr31_ \\
        --output experiments/pr31_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from loguru import logger


def _iter_sweep_dirs(roots: Iterable[Path], include_prefixes: list[str] | None) -> Iterable[Path]:
    """Yield ``<tag>/<config>/results.jsonl`` paths under each root."""
    prefixes = include_prefixes or []
    for root in roots:
        if not root.exists():
            continue
        for tag_dir in sorted(root.iterdir()):
            if not tag_dir.is_dir():
                continue
            if prefixes and not any(tag_dir.name.startswith(p) for p in prefixes):
                continue
            for cfg_dir in sorted(tag_dir.iterdir()):
                if not cfg_dir.is_dir():
                    continue
                results = cfg_dir / "results.jsonl"
                if results.exists() and results.stat().st_size > 0:
                    yield results


def consolidate(
    *,
    roots: list[Path],
    output: Path,
    include_prefixes: list[str] | None = None,
) -> int:
    """Read all ``results.jsonl`` under each root, merge, write ``output``.

    Returns the number of rows written. Each row in the output gets a
    ``_source_path`` field so consumers can trace it back. Symlinked
    files are read through; symlinks pointing at the same target are
    deduplicated (resolved-path basis).
    """
    seen_resolved: set[Path] = set()
    rows_out: list[dict] = []

    for results_path in _iter_sweep_dirs(roots, include_prefixes):
        resolved = results_path.resolve()
        if resolved in seen_resolved:
            continue
        seen_resolved.add(resolved)
        rel = str(results_path)
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"[consolidate] skipping malformed line in {rel}: {e}")
                continue
            row["_source_path"] = rel
            rows_out.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    logger.info(f"[consolidate] wrote {len(rows_out)} rows → {output}")
    return len(rows_out)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        action="append",
        default=[],
        help="Root dir to scan (repeat for multiple). Default: experiments/",
    )
    ap.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Tag-dir name prefix to include (repeat). Default: all dirs.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output path for the consolidated jsonl (e.g. experiments/pr31_results.jsonl).",
    )
    args = ap.parse_args()
    roots = [Path(r) for r in (args.root or ["experiments"])]
    consolidate(
        roots=roots,
        output=Path(args.output),
        include_prefixes=args.include_prefix or None,
    )


if __name__ == "__main__":
    _cli()
