"""Stage-1 LoRA SFT on the chat-format files from build_sft_data.py.

Loss is computed on assistant tokens only (the "Answer: X" line and, when
present, the bootstrapped rationale) — the standard letter-likelihood setup for
MCQ fine-tuning. Per-country adapter: pass one file. Joint multi-country
adapter: pass several files (country conditioning lives in the system prompts).

Runs on a single 16-24GB GPU with --load_4bit (QLoRA), or 40GB+ in bf16.
  pip install torch transformers peft trl datasets accelerate bitsandbytes
  # per-country adapter, CV fold 0:
  python src/train_lora.py --train_files sft_data/zh_train_fold0.jsonl \
      --output_dir runs/zh_fold0
  # joint adapter on everything (final submission model):
  python src/train_lora.py --train_files sft_data/zh_train_full.jsonl \
      sft_data/id_train_full.jsonl sft_data/si_train_full.jsonl \
      --output_dir runs/joint_full
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent


def load_examples(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
<<<<<<< Updated upstream
                    rows.append({"messages": json.loads(line)["messages"]})
=======
                    r = json.loads(line)
                    rows.append({"messages": r["messages"],
                                 "fold": r.get("meta", {}).get("fold", -1)})
>>>>>>> Stashed changes
    return rows


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
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_examples(args.train_files)
<<<<<<< Updated upstream
    print(f"{len(rows)} training examples from {len(args.train_files)} file(s)")
    ds = Dataset.from_list(rows).shuffle(seed=args.seed)
=======
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
>>>>>>> Stashed changes

    tok = AutoTokenizer.from_pretrained(args.base_model)
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

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        max_length=args.max_len,
        assistant_only_loss=True,   # loss on assistant tokens only
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
<<<<<<< Updated upstream
    )

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=ds,
                         processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.output_dir)
=======
        remove_unused_columns=False,  # keep pre-tokenized columns with PeftModel
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=max(2, args.batch_size),
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=ds,
                      eval_dataset=eval_ds,
                      data_collator=PadCollator(tok.pad_token_id))
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    # full loss history for plot_training.py
    hist_path = Path(args.output_dir) / "log_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
>>>>>>> Stashed changes
    print(f"adapter saved to {args.output_dir}")
    print(f"loss history -> {hist_path}")


if __name__ == "__main__":
    main()
