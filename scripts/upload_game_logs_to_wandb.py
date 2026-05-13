"""Upload curated game_logs dirs as wandb Artifacts, linked to the original runs.

Why: companion to Runner._archive_game_logs (auto-attach for *future* runs).
This script backfills *past* runs whose game_logs were written but never
logged as an Artifact.

The wandb run id is `<launcher_run_id>_<project>` (the runner appends the
project name as a suffix). The script reconstructs that, resumes via
`wandb.init(resume='must', id=...)`, and logs an Artifact named
`game_logs_<launcher_run_id>` (same naming as the inline auto-archive).

Usage
-----
    python scripts/upload_game_logs_to_wandb.py --game pokemon_red --run-id <id>
    python scripts/upload_game_logs_to_wandb.py --game pokemon_red             # all curated for one game
    python scripts/upload_game_logs_to_wandb.py --all-curated                  # everything
    python scripts/upload_game_logs_to_wandb.py --all-curated --include-checkpoints --dry-run
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import wandb
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED_YAML = REPO_ROOT / "scripts" / "curated_runs_to_upload.yaml"
ENTITY = "chaleong"

app = typer.Typer(pretty_exceptions_enable=False)


def find_run_dir(game: str, run_id: str, roots: list[str]) -> Path | None:
    for r in roots:
        p = Path(r) / game / run_id
        if p.is_dir():
            return p
    return None


def upload_one(
    game: str, run_id: str, project: str, run_dir: Path, include_checkpoints: bool, dry_run: bool
) -> tuple[str, float]:
    """Return (status, size_mb). status ∈ {'uploaded', 'dry-run', 'skip:<reason>'}."""
    # Collect files: everything except checkpoints/ unless opted in
    files = [
        f
        for f in run_dir.rglob("*")
        if f.is_file() and (include_checkpoints or "checkpoints" not in f.parts)
    ]
    size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)

    if dry_run:
        return "dry-run", size_mb

    # wandb id == <launcher_run_id>_<project> (runner appends project suffix)
    with wandb.init(
        entity=ENTITY, project=project, id=f"{run_id}_{project}", resume="must", reinit=True
    ) as run:
        art = wandb.Artifact(
            name=f"game_logs_{run_id}",
            type="game_logs",
            metadata={"source_path": str(run_dir), "include_checkpoints": include_checkpoints},
        )
        art.add_dir(str(run_dir), name=run_id)
        run.log_artifact(art)
    return "uploaded", size_mb


@app.command()
def main(
    game: Annotated[
        str | None, typer.Option(help="Game (pokemon_red / super_mario / twenty_fourty_eight)")
    ] = None,
    run_id: Annotated[str | None, typer.Option(help="Single run id (requires --game)")] = None,
    all_curated: Annotated[
        bool, typer.Option("--all-curated", help="All runs in curated YAML")
    ] = False,
    include_checkpoints: Annotated[
        bool, typer.Option(help="Also upload checkpoints/*.pkl (~10x bigger)")
    ] = False,
    dry_run: Annotated[bool, typer.Option(help="Print what would upload, don't upload")] = False,
):
    load_dotenv(REPO_ROOT / ".env")
    curated = yaml.safe_load(CURATED_YAML.read_text())
    projects = curated["projects"]
    data_roots = curated["data_roots"]

    # Resolve targets: single run-id, single game, or all curated
    if run_id:
        if not game:
            raise typer.BadParameter("--game required with --run-id")
        targets = [(game, run_id)]
    elif all_curated:
        targets = [
            (g, r)
            for g, stages in curated.items()
            if g not in ("projects", "data_roots")
            for rids in stages.values()
            for r in rids
        ]
    elif game:
        targets = [(game, r) for rids in curated.get(game, {}).values() for r in rids]
    else:
        raise typer.BadParameter("Pass --run-id, --game, or --all-curated")

    n_uploaded = n_dry = n_skipped = 0
    total_mb = 0.0
    for g, rid in targets:
        project = projects.get(g)
        run_dir = find_run_dir(g, rid, data_roots) if project else None
        if not project or not run_dir:
            n_skipped += 1
            print(f"  [skip      ] {g}/{rid}")
            continue
        status, size_mb = upload_one(g, rid, project, run_dir, include_checkpoints, dry_run)
        total_mb += size_mb
        if status == "uploaded":
            n_uploaded += 1
        elif status == "dry-run":
            n_dry += 1
        print(f"  [{status:10s}] {g}/{rid}  ({size_mb:.1f} MB)")

    print()
    print(
        f"  {len(targets)} target(s) | uploaded: {n_uploaded} | dry-run: {n_dry} | skipped: {n_skipped} | total: {total_mb:.1f} MB"
    )


if __name__ == "__main__":
    app()
