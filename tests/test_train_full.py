import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_full import build_parser, newest_checkpoint, validate_args


def test_full_finetuning_defaults_are_the_one_run_configuration():
    args = build_parser().parse_args(
        ["--train_files", "train.jsonl", "--output_dir", "run"]
    )

    assert args.epochs == 1.0
    assert args.lr == 5e-6
    assert args.batch_size == 1
    assert args.grad_accum == 16
    assert args.max_len == 1536
    assert args.warmup_ratio == 0.05
    assert args.weight_decay == 0.01
    assert args.save_steps == 25
    assert args.save_total_limit == 3
    assert not hasattr(args, "mixed_country_batches")
    assert not hasattr(args, "oversample_si_negation_3x")
    validate_args(args)


def test_save_steps_must_be_positive():
    args = build_parser().parse_args(
        [
            "--train_files",
            "train.jsonl",
            "--output_dir",
            "run",
            "--save_steps",
            "0",
        ]
    )

    with pytest.raises(ValueError, match="save_steps"):
        validate_args(args)


def test_newest_checkpoint_uses_numeric_step_order(tmp_path):
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert newest_checkpoint(tmp_path).endswith("checkpoint-100")
