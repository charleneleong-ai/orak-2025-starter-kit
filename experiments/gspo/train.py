"""GSPO offline training entrypoint.

Pipeline:

    sweep results.jsonl + game_states/*  →  collate_sweep  →  jsonl of GSPOSamples
                                                              ↓
                                                    compute_group_advantages
                                                              ↓
                                                  (prompt, completion, advantage)
                                                              ↓
                                              GSPO loop over LoRA adapter (Unsloth)
                                                              ↓
                                                  adapter saved → vLLM re-serve

This module owns the offline-runnable parts (data load + advantage attach +
LoRA training step). Three CLI commands:

  * ``info``   — load a samples jsonl, report dataset stats, dry-run-safe.
  * ``prepare`` — load + compute advantages + emit a trainer-ready jsonl
                  with the reward field replaced by the advantage.
  * ``train``  — full pipeline: load Gemma-4-26B with Unsloth, run GSPO
                  loop, save LoRA adapter. ``--dry-run`` exits before model
                  load so the data path can be validated without GPU.

Online-vs-offline note: the GSPO loss math (``advantages.py``) is
identical in both modes — what differs is how ``pi_old`` is obtained.
This module's offline mode reconstructs ``pi_old`` by toggling the
trainable LoRA adapter off via ``model.disable_adapter()`` (iter 1:
``pi_old`` = frozen base; iter 2+: load prior LoRA as the "old" adapter
first). For online operation, a future trainer would feed the same
helpers from an in-process snapshot or a vLLM HTTP scoring path.

Precompute optimization: ``pi_old`` is frozen for the entire training
cycle, so we forward-pass it ONCE upfront, cache per-sample logprobs in
memory, then training only forwards through ``pi_new``. Halves the
per-step compute vs the naive two-forward-per-step shape.
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
    """Report dataset shape + flag zero-variance groups."""
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
            "Run the re-roll launcher (experiments/gspo/reroll.sh) first."
        )


@app.command()
def prepare(
    samples_jsonl: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(
        Path("gspo_advantages.jsonl"), "--out", "-o", help="trainer-ready jsonl"
    ),
) -> None:
    """Compute group-relative advantages + emit a trainer-ready jsonl."""
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
        "unsloth/gemma-4-26B-A4B-it",
        "--base-model",
        help="HF model id loaded via Unsloth's FastLanguageModel + 4-bit quant.",
    ),
    out_dir: Path = typer.Option(
        Path("artifacts/gspo_lora"), "--out-dir", help="LoRA adapter output dir"
    ),
    epochs: int = typer.Option(1, "--epochs"),
    lora_r: int = typer.Option(16, "--lora-r"),
    lora_alpha: int = typer.Option(32, "--lora-alpha"),
    lr: float = typer.Option(5e-5, "--lr"),
    clip_eps: float = typer.Option(
        3e-4,
        "--clip-eps",
        help="GSPO importance-ratio clip (paper v2 default 3e-4; tune per task).",
    ),
    batch_size: int = typer.Option(2, "--batch-size"),
    grad_accum: int = typer.Option(4, "--grad-accum"),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    dry_run: bool = typer.Option(False, "--dry-run", help="skip the model load + train"),
) -> None:
    """Full GSPO training run.

    Refuses to start if every group has zero variance — that's the
    default ``group_id=run_id`` state and produces no learning signal.
    Run the re-roll launcher (``experiments/gspo/reroll.sh``) first to
    build K-trajectory groups.
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

    # One training cycle reconstructs ONE pi_old. A dataset mixing
    # rollouts from different policies (e.g. some under "base", some
    # under "lora_v1") would need separate pi_old loads per sample —
    # not supported. Refuse early, before any model load.
    policy_ids = {s.policy_id for s in samples}
    if len(policy_ids) > 1:
        typer.secho(
            f"Mixed policy_id values in dataset: {sorted(policy_ids)}. "
            "Train cycles must use rollouts from a single policy (the same "
            "pi_old). Split the jsonl by policy_id and train each shard "
            "separately, or re-roll under one policy.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=3)
    policy_id = next(iter(policy_ids)) if policy_ids else "base"

    records = list(iter_advantage_records(samples))
    typer.echo(f"loaded {len(records)} samples / {n_groups} groups (policy_id={policy_id})")
    typer.echo(f"first record: {records[0] if records else '(empty)'}")

    if dry_run:
        typer.echo("--dry-run: skipping model load + gradient loop")
        return

    _run_gspo_training(
        records=records,
        base_model=base_model,
        out_dir=out_dir,
        epochs=epochs,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lr=lr,
        clip_eps=clip_eps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_seq_length=max_seq_length,
    )


# ── GSPO training loop (heavy deps; only imported when train runs) ────


def _run_gspo_training(
    *,
    records: list[dict[str, object]],
    base_model: str,
    out_dir: Path,
    epochs: int,
    lora_r: int,
    lora_alpha: int,
    lr: float,
    clip_eps: float,
    batch_size: int,
    grad_accum: int,
    max_seq_length: int,
) -> None:
    """Local-import the heavy stack (unsloth + torch) inside this fn so
    ``info`` / ``prepare`` / ``--dry-run`` keep working without the
    ``gspo-training`` extra installed. Documented exception to the
    "hoist imports" rule."""
    # ruff: noqa: PLC0415 — intentional local imports
    import torch
    from unsloth import FastLanguageModel

    from experiments.gspo.advantages import (
        gather_completion_logprobs,
        gspo_clipped_loss,
        length_normalized_log_ratio_batch,
    )

    typer.echo(f"loading {base_model} via Unsloth FastLanguageModel (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto: bfloat16 on Ampere+
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        # 30% less VRAM, 2x batch — see CLAUDE.md note on Unsloth perf.
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    FastLanguageModel.for_training(model)
    device = next(model.parameters()).device
    typer.echo(f"model on {device}; LoRA r={lora_r}, alpha={lora_alpha}")

    # Tokenize each record once. Per-row tensors (variable length) — we
    # pad at batch-collate time. Cheap memory-wise (just int ids).
    typer.echo(f"tokenizing {len(records)} records...")
    tokenized = [_tokenize_record(r, tokenizer, max_seq_length) for r in records]

    # ── precompute pi_old logprobs (one-time per cycle) ─────────────
    #
    # pi_old is frozen for the whole training cycle (iter 1: base; iter
    # 2+: prior LoRA adapter, also frozen). Forwarding through pi_old
    # every step would double compute. Precompute once and cache the
    # per-token logprobs on CPU, gather per-batch at train time.
    typer.echo("precomputing pi_old logprobs (one forward pass through frozen policy)...")
    model.eval()
    old_logp_cache: list[torch.Tensor] = []
    with torch.no_grad(), model.disable_adapter():
        for tok in tokenized:
            input_ids = tok["input_ids"].to(device)
            mask = tok["completion_mask"].to(device)
            logits = model(input_ids=input_ids).logits
            logp = gather_completion_logprobs(logits, input_ids, mask)
            old_logp_cache.append(logp.detach().cpu())

    # ── train ────────────────────────────────────────────────────────
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    optimizer.zero_grad()

    n_batches = (len(tokenized) + batch_size - 1) // batch_size
    total_loss = 0.0
    accum_count = 0
    for epoch in range(epochs):
        for batch_idx in range(n_batches):
            indices = list(
                range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(tokenized)))
            )
            batch = _collate_batch(
                [tokenized[i] for i in indices],
                [old_logp_cache[i] for i in indices],
                [float(records[i]["advantage"]) for i in indices],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                device=device,
            )

            # pi_new forward (adapter active)
            new_logits = model(input_ids=batch["input_ids"]).logits
            new_logp = gather_completion_logprobs(
                new_logits, batch["input_ids"], batch["completion_mask"]
            )

            # Shifted mask matches the [T-1] output of gather_completion_logprobs.
            shifted_mask = batch["completion_mask"][:, 1:]
            log_ratio = length_normalized_log_ratio_batch(new_logp, batch["old_logp"], shifted_mask)
            loss = gspo_clipped_loss(log_ratio, batch["advantages"], epsilon=clip_eps)
            loss = loss / grad_accum
            loss.backward()
            accum_count += 1
            total_loss += loss.item() * grad_accum

            if accum_count % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                typer.echo(
                    f"epoch {epoch + 1}/{epochs} batch {batch_idx + 1}/{n_batches} "
                    f"loss={loss.item() * grad_accum:.4f}"
                )

    # Flush any tail-end accumulated gradients
    if accum_count % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    avg_loss = total_loss / max(accum_count, 1)
    typer.echo(f"saved LoRA adapter -> {out_dir}  (avg loss = {avg_loss:.4f})")


def _tokenize_record(record: dict, tokenizer, max_seq_length: int) -> dict:
    """Tokenize (prompt, completion) via Gemma chat template; build mask
    that is 1.0 on assistant (completion) tokens, 0.0 elsewhere.

    Limitation: ``prompt`` is just the env obs_str — the full prompt vLLM
    saw (system + history + active subgoal) lives only in Weave traces.
    For first-cycle parity this truncation is documented; full-fidelity
    requires Weave join, tracked as a follow-up."""
    # ruff: noqa: PLC0415
    import torch

    prompt = str(record["prompt"])
    completion = str(record["completion"])

    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    full = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        add_generation_prompt=False,
        return_tensors="pt",
    )

    # Truncate to max_seq_length from the LEFT (keep the completion intact
    # so the gradient signal is preserved; drop early prompt tokens if needed).
    if full.shape[1] > max_seq_length:
        overflow = full.shape[1] - max_seq_length
        full = full[:, overflow:]
        # The completion-region start shifts by the same amount.
        completion_start = max(0, user_only.shape[1] - overflow)
    else:
        completion_start = user_only.shape[1]

    completion_mask = torch.zeros_like(full, dtype=torch.float32)
    completion_mask[:, completion_start:] = 1.0
    return {"input_ids": full, "completion_mask": completion_mask}


def _collate_batch(
    tokenized_rows: list[dict],
    old_logp_rows: list,
    advantages: list[float],
    *,
    pad_token_id: int,
    device,
) -> dict:
    """Right-pad variable-length sequences to the batch max, stack into
    [B, T] tensors. ``old_logp`` rows are [1, T-1] each — we pad those
    on the same axis with zeros (which the mask suppresses anyway)."""
    # ruff: noqa: PLC0415
    import torch

    max_len = max(t["input_ids"].shape[1] for t in tokenized_rows)
    batch_size = len(tokenized_rows)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    completion_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    # old_logp is shape [1, T-1] per row → batched [B, max_len - 1]
    old_logp = torch.zeros((batch_size, max_len - 1), dtype=torch.float32)

    for i, (tok, old) in enumerate(zip(tokenized_rows, old_logp_rows)):
        L = tok["input_ids"].shape[1]
        input_ids[i, :L] = tok["input_ids"][0]
        completion_mask[i, :L] = tok["completion_mask"][0]
        old_logp[i, : L - 1] = old[0]

    return {
        "input_ids": input_ids.to(device),
        "completion_mask": completion_mask.to(device),
        "old_logp": old_logp.to(device),
        "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
    }


if __name__ == "__main__":
    app()
