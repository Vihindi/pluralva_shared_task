"""Single-GPU full-parameter SFT for the chat JSONL built by build_sft_data.py.

Unlike train_lora.py, this script does not create or load a PEFT adapter and
does not quantize the model weights. Every base-model parameter is trainable.
The model and gradients use BF16 while AdamW optimizer states use bitsandbytes
8-bit storage to make full fine-tuning an 8B model practical on a 96GB GPU.

All supplied rows are combined into one ordinary map-style dataset. Hugging
Face Trainer's RandomSampler shuffles that combined dataset naturally; there
is no country balancing, stratification, or oversampling.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from mixed_microbatching import load_schedule_rows


class EpochLossCallback(TrainerCallback):
    """Print an average of the losses logged during each epoch."""

    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.losses:
            average = sum(self.losses) / len(self.losses)
            print(
                f"=== epoch {state.epoch:.2f} finished | avg train loss: "
                f"{average:.4f} ({len(self.losses)} logged steps) ==="
            )
            self.losses = []
        return control


class PadCollator:
    """Pad inputs dynamically and keep prompt tokens masked from the loss."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for feature in features:
            length = len(feature["input_ids"])
            padding = max_length - length
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_id] * padding
            )
            batch["labels"].append(
                feature["labels"] + [-100] * padding
            )
            batch["attention_mask"].append(
                [1] * length + [0] * padding
            )
        return {
            name: torch.tensor(values, dtype=torch.long)
            for name, values in batch.items()
        }


def template_ids(tokenizer, messages):
    """Render a generation prompt with the model's own chat template."""

    kwargs = {"add_generation_prompt": True, "tokenize": True}
    try:
        ids = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def encode_example(tokenizer, messages, max_length):
    """Tokenize one chat and calculate loss on assistant tokens only."""

    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("every SFT row must end with an assistant message")
    prompt = template_ids(tokenizer, messages[:-1])
    target = tokenizer.encode(
        messages[-1]["content"], add_special_tokens=False
    )
    target.append(tokenizer.eos_token_id)
    input_ids = prompt + target
    if len(input_ids) > max_length:
        return None
    return {
        "input_ids": input_ids,
        "labels": [-100] * len(prompt) + target,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full-parameter BF16 SFT with natural shuffled sampling."
    )
    parser.add_argument(
        "--base_model", default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--train_files", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=1536)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--save_steps",
        type=int,
        default=175,
        help="save a resumable checkpoint every N optimizer steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="maximum number of step checkpoints retained",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attn_implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="SDPA needs no extra package; Flash Attention 2 may be faster",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the newest checkpoint-* inside --output_dir",
    )
    return parser


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.lr <= 0:
        raise ValueError("--lr must be greater than zero")
    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("--batch_size and --grad_accum must be positive")
    if args.max_len <= 0:
        raise ValueError("--max_len must be positive")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup_ratio must be in [0, 1)")
    if args.save_steps <= 0:
        raise ValueError("--save_steps must be positive")
    if args.save_total_limit <= 0:
        raise ValueError("--save_total_limit must be positive")


def newest_checkpoint(output_dir):
    checkpoints = []
    for path in Path(output_dir).glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return str(max(checkpoints)[1]) if checkpoints else None


def main():
    args = build_parser().parse_args()
    validate_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_schedule_rows(args.train_files)
    features = []
    skipped = 0
    dataset_counts = {}
    for row in rows:
        encoded = encode_example(tokenizer, row["messages"], args.max_len)
        if encoded is None:
            skipped += 1
            continue
        features.append(encoded)
        dataset = row.get("dataset", "unknown")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    if not features:
        raise SystemExit("no training rows remain after tokenization")

    # Trainer uses RandomSampler for this ordinary map-style train dataset.
    # Do not call Dataset.shuffle here: a second fixed shuffle is unnecessary.
    train_dataset = Dataset.from_list(features)
    effective_batch = args.batch_size * args.grad_accum
    updates_per_epoch = math.ceil(len(features) / effective_batch)
    estimated_updates = math.ceil(updates_per_epoch * args.epochs)
    print(f"training rows: {len(features)} ({skipped} over-length skipped)")
    print(f"natural dataset distribution: {dataset_counts}")
    print(
        f"physical batch={args.batch_size}, accumulation={args.grad_accum}, "
        f"effective batch={effective_batch}"
    )
    print(
        f"estimated optimizer updates: {estimated_updates}; "
        f"checkpoint interval: {args.save_steps} updates"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    trainable_percent = 100 * trainable_parameters / total_parameters
    print(
        f"trainable parameters: {trainable_parameters:,} / "
        f"{total_parameters:,} ({trainable_percent:.6f}%)"
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError(
            "full fine-tuning requires every model parameter to be trainable"
        )
    if getattr(model, "is_loaded_in_4bit", False) or getattr(
        model, "is_loaded_in_8bit", False
    ):
        raise RuntimeError(
            "the model itself must not be loaded in 4-bit or 8-bit for full "
            "fine-tuning"
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        optim="adamw_bnb_8bit",
        bf16=True,
        tf32=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[EpochLossCallback()],
    )

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = newest_checkpoint(args.output_dir)
        if resume_checkpoint:
            print(f"resuming from {resume_checkpoint}")
        else:
            print("--resume requested, but no checkpoint-* exists; starting fresh")

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    output_path = Path(args.output_dir)
    with open(output_path / "log_history.json", "w", encoding="utf-8") as file:
        json.dump(trainer.state.log_history, file, indent=2)
    with open(
        output_path / "training_summary.json", "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "training_type": "full_parameter_finetuning",
                "base_model": args.base_model,
                "train_files": args.train_files,
                "training_rows": len(features),
                "skipped_over_length": skipped,
                "dataset_counts": dataset_counts,
                "natural_random_sampling": True,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler": "cosine",
                "batch_size": args.batch_size,
                "gradient_accumulation": args.grad_accum,
                "effective_batch_size": effective_batch,
                "optimizer": "adamw_bnb_8bit",
                "model_dtype": "bfloat16",
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "max_len": args.max_len,
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
                "seed": args.seed,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
