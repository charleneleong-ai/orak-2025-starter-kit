"""GSPO train entrypoint — offline-runnable parts.

Skeleton only; the gradient loop is TODO(gpu). What's testable now: the
data-pipeline glue (jsonl I/O, advantage attachment, dry-run path,
zero-variance refusal)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from experiments.gspo.collate import GSPOSample
from experiments.gspo.train import (
    app,
    iter_advantage_records,
    load_samples,
)


def _sample(reward: float, group_id: str = "g", run_id: str | None = None) -> GSPOSample:
    return GSPOSample(
        run_id=run_id or f"r_{group_id}_{reward}",
        iter_step=1,
        prompt="x",
        completion="y",
        reward=reward,
        group_id=group_id,
    )


def _write_jsonl(path: Path, samples: list[GSPOSample]) -> None:
    path.write_text("\n".join(json.dumps(asdict(s)) for s in samples) + "\n")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def multi_group_jsonl(tmp_path: Path) -> Path:
    """A 2-group dataset where both groups have real reward variance —
    the trainer should accept this."""
    p = tmp_path / "multi.jsonl"
    _write_jsonl(
        p,
        [
            _sample(reward=0.2, group_id="ga", run_id="ga_r1"),
            _sample(reward=0.6, group_id="ga", run_id="ga_r2"),
            _sample(reward=0.3, group_id="gb", run_id="gb_r1"),
            _sample(reward=0.9, group_id="gb", run_id="gb_r2"),
        ],
    )
    return p


@pytest.fixture
def degenerate_jsonl(tmp_path: Path) -> Path:
    """Singletons — every group has variance 0. Trainer must refuse."""
    p = tmp_path / "degen.jsonl"
    _write_jsonl(
        p,
        [
            _sample(reward=0.5, group_id="iter1"),
            _sample(reward=0.7, group_id="iter2"),
            _sample(reward=0.3, group_id="iter3"),
        ],
    )
    return p


class TestLoadSamples:
    def test_roundtrip(self, tmp_path: Path):
        samples = [_sample(reward=0.4), _sample(reward=0.8)]
        p = tmp_path / "x.jsonl"
        _write_jsonl(p, samples)
        out = load_samples(p)
        assert out == samples

    def test_skips_blank_lines(self, tmp_path: Path):
        p = tmp_path / "x.jsonl"
        s = _sample(reward=0.5)
        p.write_text(json.dumps(asdict(s)) + "\n\n\n")
        assert load_samples(p) == [s]


class TestIterAdvantageRecords:
    def test_emits_one_per_sample_with_advantage_field(self):
        samples = [_sample(reward=0.4, group_id="g"), _sample(reward=0.8, group_id="g")]
        records = list(iter_advantage_records(samples))
        assert len(records) == 2
        # advantage = ±1.0 (population std on n=2 with rewards 0.4, 0.8)
        assert records[0]["advantage"] == pytest.approx(-1.0)
        assert records[1]["advantage"] == pytest.approx(+1.0)

    def test_reward_field_holds_advantage(self):
        """The flattened dict has both ``reward`` (overwritten) and
        ``advantage`` (explicit alias). Downstream consumers may use
        either — both point at the same value."""
        samples = [_sample(reward=0.4, group_id="g"), _sample(reward=0.8, group_id="g")]
        records = list(iter_advantage_records(samples))
        for record in records:
            assert record["reward"] == record["advantage"]


class TestInfoCommand:
    def test_reports_counts_and_stats(self, runner: CliRunner, multi_group_jsonl: Path):
        result = runner.invoke(app, ["info", str(multi_group_jsonl)])
        assert result.exit_code == 0
        assert "samples:" in result.stdout
        assert "4" in result.stdout  # 4 samples
        assert "groups:" in result.stdout
        assert "zero-variance groups:    0" in result.stdout

    def test_flags_all_zero_variance_dataset(self, runner: CliRunner, degenerate_jsonl: Path):
        result = runner.invoke(app, ["info", str(degenerate_jsonl)])
        assert result.exit_code == 0
        assert "WARNING" in result.stdout
        assert "re-roll launcher" in result.stdout


class TestPrepareCommand:
    def test_writes_advantage_records(
        self, runner: CliRunner, multi_group_jsonl: Path, tmp_path: Path
    ):
        out = tmp_path / "advantages.jsonl"
        result = runner.invoke(app, ["prepare", str(multi_group_jsonl), "--out", str(out)])
        assert result.exit_code == 0
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(rows) == 4
        for row in rows:
            assert "advantage" in row and "prompt" in row and "completion" in row


class TestTrainCommand:
    def test_dry_run_succeeds_on_multi_group(self, runner: CliRunner, multi_group_jsonl: Path):
        result = runner.invoke(app, ["train", str(multi_group_jsonl), "--dry-run"])
        assert result.exit_code == 0
        assert "loaded 4 samples / 2 groups" in result.stdout
        assert "dry-run" in result.stdout

    def test_refuses_degenerate_dataset(self, runner: CliRunner, degenerate_jsonl: Path):
        """Every group has variance 0 → no gradient signal → exit code 2."""
        result = runner.invoke(app, ["train", str(degenerate_jsonl), "--dry-run"])
        assert result.exit_code == 2
        assert "zero variance" in result.stdout or "no gradient signal" in result.stdout

    def test_full_path_not_run_says_todo_gpu(self, runner: CliRunner, multi_group_jsonl: Path):
        """Without --dry-run, the function emits the TODO(gpu) marker
        rather than actually starting a training loop."""
        result = runner.invoke(app, ["train", str(multi_group_jsonl)])
        assert result.exit_code == 0
        assert "TODO(gpu)" in result.stdout
