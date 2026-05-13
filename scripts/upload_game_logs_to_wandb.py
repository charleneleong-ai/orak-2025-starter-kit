"""Upload curated game_logs directories as wandb Artifacts, linked to their
original wandb runs.

Per-run-dir basename == wandb run_id (run.py sets settings.wandb.run_id from
the launcher's --run-id). So matching is straightforward: locate the run dir,
resume the wandb run with the matching id, log_artifact, finish.

Examples
--------
    # Single run, dry-run (no upload)
    python scripts/upload_game_logs_to_wandb.py \\
        --game pokemon_red \\
        --run-id pr_procesc_stage_g_pokemon_iter3_20260513T082840Z \\
        --dry-run

    # Real upload of a single run
    python scripts/upload_game_logs_to_wandb.py \\
        --game pokemon_red \\
        --run-id pr_procesc_stage_g_pokemon_iter3_20260513T082840Z

    # All curated runs for a game, dry-run
    python scripts/upload_game_logs_to_wandb.py --game pokemon_red --all-curated --dry-run

    # Everything in the curated YAML
    python scripts/upload_game_logs_to_wandb.py --all-curated
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

# Files to bundle into the artifact (skip massive ones unless --include-checkpoints)
DEFAULT_INCLUDE_FILES = {
    "config.yaml",
    "evaluation.log",
    "evaluation_summary.json",
    "game_server.log",
    "game_states.jsonl",
    "model_declaration.json",
}
DEFAULT_INCLUDE_DIRS = {"logs"}
CHECKPOINTS_DIR = "checkpoints"

app = typer.Typer(pretty_exceptions_enable=False)


def load_curated() -> dict:
    with CURATED_YAML.open() as f:
        return yaml.safe_load(f)


def find_run_dir(game: str, run_id: str, data_roots: list[str]) -> Path | None:
    for root in data_roots:
        candidate = Path(root) / game / run_id
        if candidate.is_dir():
            return candidate
    return None


def collect_files_for_artifact(run_dir: Path, include_checkpoints: bool) -> list[Path]:
    files: list[Path] = []
    for name in DEFAULT_INCLUDE_FILES:
        p = run_dir / name
        if p.exists():
            files.append(p)
    for d in DEFAULT_INCLUDE_DIRS:
        d_path = run_dir / d
        if d_path.is_dir():
            files.extend(p for p in d_path.rglob("*") if p.is_file())
    if include_checkpoints:
        cp_dir = run_dir / CHECKPOINTS_DIR
        if cp_dir.is_dir():
            files.extend(p for p in cp_dir.rglob("*") if p.is_file())
    return files


def upload_one(
    *,
    game: str,
    run_id: str,
    project: str,
    run_dir: Path,
    include_checkpoints: bool,
    dry_run: bool,
) -> dict:
    files = collect_files_for_artifact(run_dir, include_checkpoints=include_checkpoints)
    total_bytes = sum(f.stat().st_size for f in files)
    total_mb = total_bytes / (1024 * 1024)

    info = {
        "game": game,
        "run_id": run_id,
        "project": f"{ENTITY}/{project}",
        "run_dir": str(run_dir),
        "n_files": len(files),
        "total_mb": round(total_mb, 1),
        "include_checkpoints": include_checkpoints,
    }

    if dry_run:
        info["status"] = "dry-run"
        return info

    # Resume the existing wandb run by id
    run = wandb.init(
        entity=ENTITY,
        project=project,
        id=run_id,
        resume="must",
        reinit=True,
    )
    try:
        artifact_name = f"game_logs_{run_id}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="game_logs",
            description=f"game_logs dir from {run_dir}",
            metadata={
                "n_files": len(files),
                "size_mb": round(total_mb, 1),
                "include_checkpoints": include_checkpoints,
                "source_path": str(run_dir),
            },
        )
        # add_dir preserves the relative structure; we add the run_dir as the root
        artifact.add_dir(str(run_dir), name=run_id)
        run.log_artifact(artifact)
    finally:
        run.finish()

    info["status"] = "uploaded"
    info["artifact_name"] = f"{artifact_name}:latest"
    return info


@app.command()
def main(
    game: Annotated[
        str | None,
        typer.Option("--game", help="Game name (e.g. pokemon_red, super_mario, twenty_fourty_eight)"),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Single run id to upload (e.g. pr_procesc_stage_g_pokemon_iter3_...)"),
    ] = None,
    all_curated: Annotated[
        bool, typer.Option("--all-curated", help="Upload every run in scripts/curated_runs_to_upload.yaml")
    ] = False,
    include_checkpoints: Annotated[
        bool, typer.Option("--include-checkpoints", help="Also upload checkpoints/*.pkl (~10x bigger artifacts)")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print what would be uploaded, don't actually upload")] = False,
):
    load_dotenv(REPO_ROOT / ".env")
    curated = load_curated()

    targets: list[tuple[str, str]] = []
    if run_id is not None:
        if game is None:
            raise typer.BadParameter("--game required with --run-id")
        targets.append((game, run_id))
    elif all_curated:
        for game_name, stages in curated.items():
            if game_name in ("projects", "data_roots"):
                continue
            if game and game_name != game:
                continue
            for _stage, run_ids in stages.items():
                for rid in run_ids:
                    targets.append((game_name, rid))
    elif game is not None:
        # Just the curated runs for this one game
        stages = curated.get(game, {})
        for _stage, run_ids in stages.items():
            for rid in run_ids:
                targets.append((game, rid))
    else:
        raise typer.BadParameter("Pass --run-id (single), --game (curated for that game), or --all-curated")

    projects = curated["projects"]
    data_roots = curated["data_roots"]

    results: list[dict] = []
    for g, rid in targets:
        project = projects.get(g)
        if project is None:
            results.append({"game": g, "run_id": rid, "status": "skip:unknown-project"})
            continue
        run_dir = find_run_dir(g, rid, data_roots)
        if run_dir is None:
            results.append({"game": g, "run_id": rid, "status": "skip:no-run-dir"})
            continue
        info = upload_one(
            game=g,
            run_id=rid,
            project=project,
            run_dir=run_dir,
            include_checkpoints=include_checkpoints,
            dry_run=dry_run,
        )
        results.append(info)
        print(f"  [{info['status']:10s}] {g}/{rid}  ({info.get('total_mb','?')} MB, {info.get('n_files','?')} files)")

    print()
    print("=" * 60)
    print(f"  {len(results)} target(s)")
    print(f"  uploaded:   {sum(1 for r in results if r.get('status') == 'uploaded')}")
    print(f"  dry-run:    {sum(1 for r in results if r.get('status') == 'dry-run')}")
    print(f"  skipped:    {sum(1 for r in results if r.get('status', '').startswith('skip:'))}")
    total_mb = sum(r.get("total_mb", 0) for r in results if r.get("status") in ("uploaded", "dry-run"))
    print(f"  total size: {total_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    app()
