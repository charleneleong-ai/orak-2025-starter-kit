"""GSPO/GRPO offline training entrypoint — skeleton.

Pipeline:

    sweep results.jsonl + game_states/*  →  collate_sweep  →  jsonl of GSPOSamples
                                                              ↓
                                                    compute_group_advantages
                                                              ↓
                                                  (prompt, completion, advantage)
                                                              ↓
                                              GRPO/GSPO loop over LoRA adapter
                                                              ↓
                                                  adapter saved → vLLM re-serve

This module owns the offline-runnable parts (data load + advantage attach +
trainer setup). The gradient step itself is marked TODO(gpu) since it
requires the model weights resident on GPU — currently we share with vLLM
on a single 40 GB card, so training has to wait for the sweep daemon to
release the memory.

Three CLI commands:
  * ``info``   — load a samples jsonl, report dataset stats, dry-run-safe.
  * ``prepare`` — load + compute advantages + emit a trainer-ready jsonl
                  with the reward field replaced by the advantage.
  * ``train``  — full pipeline (TODO(gpu): the actual loop).

Why a skeleton: separates the data-shaping work (which we can validate
end-to-end on the existing Stage R + Stage S trajectories *now*) from the
gradient step (which needs GPU room we don't have while the v1 sweep runs).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import typer

from experiments.gspo.advantages import (
    attach_advantage,
    compute_group_advantages,
    zero_variance_group_ids,
)
from experiments.gspo.collate import GSPOSample

app = typer.Typer(add_completion=False, no_args_is_help=True)


# ── jsonl I/O ─────────────────────────────────────────────────────────


def load_samples(jsonl_path: Path) -> list[GSPOSample]:
    """Load a collated samples jsonl as a list of ``GSPOSample``."""
    out: list[GSPOSample] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(GSPOSample(**d))
    return out


def iter_advantage_records(samples: list[GSPOSample]) -> Iterator[dict[str, object]]:
    """Yield ``(prompt, completion, advantage, group_id, run_id, iter_step)``
    dicts ready for the trainer's batch loader."""
    for sample, advantage in compute_group_advantages(samples):
        yield {
            **asdict(attach_advantage(sample, advantage)),
            # `reward` field now holds advantage; alias for clarity downstream.
            "advantage": advantage,
        }


# ── CLI commands ──────────────────────────────────────────────────────


@app.command()
def info(
    samples_jsonl: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Report dataset shape + flag zero-variance groups.

    The default ``group_id=run_id`` collation gives one trajectory per
    group — every group has zero variance, so the trainer would refuse
    to run on this data alone. This command surfaces that state."""
    samples = load_samples(samples_jsonl)
    n_groups = len({s.group_id for s in samples})
    bad_groups = zero_variance_group_ids(samples)
    rewards = [s.reward for s in samples]
    typer.echo(f"samples:                 {len(samples)}")
    typer.echo(f"groups:                  {n_groups}")
    typer.echo(f"zero-variance groups:    {len(bad_groups)} / {n_groups}")
    if rewards:
        typer.echo(f"reward range:            [{min(rewards):.4f}, {max(rewards):.4f}]")
        typer.echo(f"reward mean:             {sum(rewards) / len(rewards):.4f}")
    if bad_groups and len(bad_groups) == n_groups:
        typer.echo(
            "WARNING: every group has zero variance — train.py would refuse to run. "
            "Need a re-roll launcher that fixes group_id across K trajectories from "
            "the same checkpoint state to produce a meaningful gradient signal."
        )


@app.command()
def prepare(
    samples_jsonl: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(
        Path("gspo_advantages.jsonl"), "--out", "-o", help="trainer-ready jsonl"
    ),
) -> None:
    """Compute group-relative advantages + emit a trainer-ready jsonl.

    Output rows: ``{run_id, iter_step, prompt, completion, advantage, group_id}``.
    The ``reward`` field of the underlying dataclass is overwritten with
    the advantage value (`reward` field doubles as advantage in the
    flattened format)."""
    samples = load_samples(samples_jsonl)
    n_groups = len({s.group_id for s in samples})
    bad = zero_variance_group_ids(samples)
    if bad and len(bad) == n_groups:
        typer.secho(
            "Every group has zero variance; emitting anyway but trainer will refuse.",
            fg=typer.colors.YELLOW,
        )
    with out.open("w") as f:
        for record in iter_advantage_records(samples):
            f.write(json.dumps(record) + "\n")
    typer.echo(f"wrote {len(samples)} advantage records from {n_groups} groups -> {out}")


@app.command()
def train(
    samples_jsonl: Path = typer.Argument(..., exists=True, dir_okay=False),
    base_model: str = typer.Option(
        "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "--base-model",
        help="HF id / path. AWQ-quantized models need a non-quantized variant for training.",
    ),
    out_dir: Path = typer.Option(
        Path("artifacts/gspo_lora"), "--out-dir", help="LoRA adapter output dir"
    ),
    epochs: int = typer.Option(1, "--epochs"),
    lora_r: int = typer.Option(16, "--lora-r"),
    lora_alpha: int = typer.Option(32, "--lora-alpha"),
    lr: float = typer.Option(5e-5, "--lr"),
    clip_eps: float = typer.Option(0.2, "--clip-eps", help="PPO/GSPO clip range"),
    dry_run: bool = typer.Option(False, "--dry-run", help="skip the gradient loop"),
) -> None:
    """Full GSPO training run.

    Refuses to start if every group has zero variance — that's the
    default ``group_id=run_id`` state and produces no learning signal.
    Run the re-roll launcher first (TODO: separate command) to build
    K-trajectory groups.

    On --dry-run, loads samples, computes advantages, prints a batch
    sample, exits without touching the model. Useful for validating the
    data pipeline end-to-end while GPU is busy with sweep work.
    """
    samples = load_samples(samples_jsonl)
    n_groups = len({s.group_id for s in samples})
    bad = zero_variance_group_ids(samples)
    if bad and len(bad) == n_groups:
        typer.secho(
            "Every group has zero variance — no gradient signal. Run the re-roll launcher first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    records = list(iter_advantage_records(samples))
    typer.echo(f"loaded {len(records)} samples / {n_groups} groups")
    typer.echo(f"first record: {records[0] if records else '(empty)'}")

    if dry_run:
        typer.echo("--dry-run: skipping model load + gradient loop")
        return

    # TODO(gpu): the rest of this function is the GSPO/GRPO loop.
    # Reference shape (left as TODOs since the GPU is currently full):
    #
    #   from transformers import AutoModelForCausalLM, AutoTokenizer
    #   from peft import LoraConfig, get_peft_model
    #   from trl import GRPOTrainer, GRPOConfig
    #
    #   tok   = AutoTokenizer.from_pretrained(base_model)
    #   model = AutoModelForCausalLM.from_pretrained(
    #       base_model, torch_dtype="bfloat16", device_map="auto"
    #   )
    #   model = get_peft_model(
    #       model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, ...)
    #   )
    #
    #   cfg = GRPOConfig(
    #       output_dir=str(out_dir), num_train_epochs=epochs,
    #       learning_rate=lr, beta=0.0,           # no KL (pure GSPO)
    #       cliprange=clip_eps,
    #       # GSPO-specific: sequence-level importance ratio. trl >= 0.20
    #       # exposes this as a flag; pre-0.20 it's a one-line monkeypatch
    #       # of the loss function.
    #   )
    #   trainer = GRPOTrainer(model, args=cfg, train_dataset=...)
    #   trainer.train()
    #   model.save_pretrained(out_dir)
    typer.echo("training loop is TODO(gpu) — waiting on vLLM to release the 37 GB it's holding.")
    typer.echo(f"would have trained against {len(records)} advantages and saved to {out_dir}")


if __name__ == "__main__":
    app()
