"""Stage-1 LoRA SFT on the chat-format files from build_sft_data.py.

Loss is computed on assistant tokens only (the "Answer: X" line and, when
present, the bootstrapped rationale). Instead of trl's assistant_only_loss —
which requires {% generation %} markers in the chat template and therefore
fails on Llama-3.1 — examples are pre-tokenized here: the prompt is rendered
with the model's own chat template (add_generation_prompt=True), the assistant
target + EOS is appended, and prompt positions are masked to -100 in labels.
This works with any template and matches evaluate.py's scoring layout exactly
(template prompt + "Answer: X" continuation).

Per-country adapter: pass one file. Joint multi-country adapter: pass several
files (country conditioning lives in the system prompts).

Runs on a single 16-24GB GPU with --load_4bit (QLoRA), or 40GB+ in bf16.
  pip install torch transformers peft datasets accelerate bitsandbytes
  # per-country adapter, CV fold 0:
  python src/train_lora.py --train_files sft_data/zh_train_fold0.jsonl \
      --output_dir runs/zh_fold0
  # joint fold-0 adapter with validation-loss tracking:
  python src/train_lora.py --train_files sft_data/zh_train_full.jsonl \
      sft_data/id_train_full.jsonl sft_data/si_train_full.jsonl \
      --eval_fold 0 --output_dir runs/joint_fold0
  # final submission model (all dev data, no holdout):
  python src/train_lora.py --train_files sft_data/zh_train_full.jsonl \
      sft_data/id_train_full.jsonl sft_data/si_train_full.jsonl \
      --output_dir runs/joint_full
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

ROOT = Path(__file__).resolve().parent.parent


def load_examples(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.append({"messages": r["messages"],
                                 "fold": r.get("meta", {}).get("fold", -1)})
    return rows


def template_ids(tok, messages):
    """Prompt token ids via the model's chat template (thinking disabled when
    the template supports it), normalized to a flat list of ints."""
    kwargs = {"add_generation_prompt": True, "tokenize": True}
    try:
        ids = tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tok.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def encode_example(tok, messages, max_len):
    """-> {input_ids, labels} with prompt masked, or None if too long."""
    assert messages[-1]["role"] == "assistant", "last message must be assistant"
    prompt_ids = template_ids(tok, messages[:-1])
    target_ids = tok.encode(messages[-1]["content"], add_special_tokens=False)
    target_ids.append(tok.eos_token_id)
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in feats:
            n = len(f["input_ids"])
            pad = maxlen - n
            batch["input_ids"].append(list(f["input_ids"]) + [self.pad_id] * pad)
            batch["labels"].append(list(f["labels"]) + [-100] * pad)
            batch["attention_mask"].append([1] * n + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA (16-24GB GPUs)")
    ap.add_argument("--eval_fold", type=int, default=None,
                    help="hold out this CV fold as the validation set for loss "
                         "curves; pass the *_train_full.jsonl files with this")
    ap.add_argument("--eval_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=0,
                    help="save a checkpoint every N optimizer steps (0 = only at "
                         "epoch end). Use e.g. 50 on Colab to survive disconnects.")
    ap.add_argument("--save_total_limit", type=int, default=2,
                    help="keep only the newest N step-checkpoints (saves disk)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint-* in --output_dir")
    ap.add_argument("--logging_strategy", choices=["epoch", "steps"], default="epoch",
                    help="'epoch' (default) prints one loss line per epoch; "
                         "'steps' prints every --logging_steps optimizer steps")
    ap.add_argument("--logging_steps", type=int, default=10,
                    help="only used when --logging_strategy steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = load_examples(args.train_files)
    eval_rows = []
    if args.eval_fold is not None:
        eval_rows = [r for r in rows if r["fold"] == args.eval_fold]
        rows = [r for r in rows if r["fold"] != args.eval_fold]
        if not eval_rows:
            raise SystemExit(
                f"--eval_fold {args.eval_fold} matched no examples. Pass the "
                f"*_train_full.jsonl files (fold-filtered *_train_fold{args.eval_fold} "
                f"files already exclude that fold).")

    def encode_all(rws):
        feats, skipped = [], 0
        for r in rws:
            enc = encode_example(tok, r["messages"], args.max_len)
            if enc is None:
                skipped += 1
            else:
                feats.append(enc)
        return feats, skipped

    feats, skipped = encode_all(rows)
    print(f"{len(feats)} training examples from {len(args.train_files)} file(s)"
          f" ({skipped} skipped as longer than {args.max_len} tokens)")
    ds = Dataset.from_list(feats).shuffle(seed=args.seed)
    eval_ds = None
    if eval_rows:
        eval_feats, _ = encode_all(eval_rows)
        eval_ds = Dataset.from_list(eval_feats)
        print(f"{len(eval_feats)} validation examples (held-out fold {args.eval_fold})")

    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto",
                    "attn_implementation": "sdpa"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        model_kwargs.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False

    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    else:
        model.enable_input_require_grads()  # needed with gradient checkpointing

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,  # keep pre-tokenized columns with PeftModel
        eval_strategy="steps" if eval_rows else "no",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=max(2, args.batch_size),
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=ds,
                      eval_dataset=eval_ds,
                      data_collator=PadCollator(tok.pad_token_id))

    # resume from the newest checkpoint-* in output_dir if asked and one exists
    resume_ckpt = None
    if args.resume:
        ckpts = sorted(Path(args.output_dir).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            resume_ckpt = str(ckpts[-1])
            print(f"resuming from {resume_ckpt}")
        else:
            print(f"--resume set but no checkpoint-* in {args.output_dir}; "
                  f"starting fresh")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    # full loss history for plot_training.py
    hist_path = Path(args.output_dir) / "log_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"adapter saved to {args.output_dir}")
    print(f"loss history -> {hist_path}")


if __name__ == "__main__":
    main()
