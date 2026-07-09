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
                    rows.append(json.loads(line)["messages"])
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
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = load_examples(args.train_files)
    feats, skipped = [], 0
    for messages in rows:
        enc = encode_example(tok, messages, args.max_len)
        if enc is None:
            skipped += 1
        else:
            feats.append(enc)
    print(f"{len(feats)} training examples from {len(args.train_files)} file(s)"
          f" ({skipped} skipped as longer than {args.max_len} tokens)")
    ds = Dataset.from_list(feats).shuffle(seed=args.seed)

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
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,  # keep pre-tokenized columns with PeftModel
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=ds,
                      data_collator=PadCollator(tok.pad_token_id))
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
