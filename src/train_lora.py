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
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainerCallback, TrainingArguments)

ROOT = Path(__file__).resolve().parent.parent

# Architecture name fragments that indicate a multimodal/VLM checkpoint (e.g.
# Qwen3.5's "Qwen3_5ForConditionalGeneration") rather than a plain text
# AutoModelForCausalLM. Extend this list as new families show up.
_MULTIMODAL_ARCH_HINTS = ("ConditionalGeneration", "ForVision2Seq", "VLFor")


def is_multimodal_checkpoint(base_model):
    """Best-effort detection from config.json alone (no weights downloaded
    yet): true if the declared architecture or presence of vision_config
    indicates a vision-language checkpoint rather than a plain text LM."""
    try:
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=False)
    except Exception:
        return False  # unreadable config -> assume plain text LM, let the
                      # normal AutoModelForCausalLM path raise if that's wrong
    archs = getattr(cfg, "architectures", None) or []
    if any(hint in a for a in archs for hint in _MULTIMODAL_ARCH_HINTS):
        return True
    return hasattr(cfg, "vision_config")


def load_tokenizer(base_model, multimodal):
    """AutoTokenizer for plain text LMs; AutoProcessor's .tokenizer for
    multimodal checkpoints (falls back to AutoTokenizer if that also works)."""
    if not multimodal:
        return AutoTokenizer.from_pretrained(base_model)
    try:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(base_model).tokenizer
    except Exception:
        return AutoTokenizer.from_pretrained(base_model)


def load_base_model(base_model, model_kwargs, multimodal):
    """AutoModelForCausalLM for plain text LMs; AutoModelForMultimodalLM for
    multimodal checkpoints like Qwen3.5 (used purely as a text backbone here —
    we only ever train on input_ids/attention_mask/labels, no pixel_values)."""
    if not multimodal:
        return AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    print(f"{base_model!r} looks like a multimodal checkpoint "
         f"(architecture/vision_config detected) -> loading via "
         f"AutoModelForMultimodalLM, text-only usage.")
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as e:
        raise SystemExit(
            f"{base_model!r} requires AutoModelForMultimodalLM, which isn't "
            f"in your installed transformers version ({e}). Qwen's own model "
            f"card says this needs the git-main build:\n"
            f"  pip install -U git+https://github.com/huggingface/transformers"
        ) from e
    return AutoModelForMultimodalLM.from_pretrained(base_model, **model_kwargs)


# The 7 projections shared by all standard dense transformers (Llama-3.x,
# Qwen3-8B/4B, Mistral, ...). These are the ONLY modules LoRA adapts here, for
# every model. On a hybrid model like Qwen3.5 that deliberately leaves the
# linear-attention projections (in_proj_*/out_proj) unadapted — only the
# full-attention and MLP layers are trained.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def resolve_target_modules(model):
    """Always the fixed 7 dense projections, intersected with what this model
    actually has (so a missing name can't silently break LoraConfig). Reports
    any adaptable modules that are being skipped, e.g. Qwen3.5's
    linear-attention projections."""
    present = find_lora_target_modules(model)
    chosen = [m for m in TARGET_MODULES if m in present]
    missing = [m for m in TARGET_MODULES if m not in present]
    if missing:
        print(f"  WARNING: these standard modules aren't in this model, "
              f"skipping: {missing}")
    skipped = [m for m in present if m not in chosen]
    if skipped:
        print(f"  note: not adapting {len(skipped)} other module(s) this "
              f"model has: {skipped}")
    if not chosen:
        raise SystemExit(
            f"none of {TARGET_MODULES} exist in this model — its projections "
            f"are named: {present}")
    return chosen


def find_lora_target_modules(model, exclude_substrings=("vision", "visual")):
    """Discover Linear (incl. bitsandbytes-quantized) leaf module names to
    LoRA-adapt, instead of a hardcoded list tied to one architecture. Works
    unchanged for dense models (recovers the usual q/k/v/o/gate/up/down set)
    and for hybrid/linear-attention architectures like Qwen3.5, whose exact
    internal projection names we don't hardcode-guess. lm_head and anything
    under a path containing exclude_substrings (default: the vision tower, so
    a multimodal checkpoint is only ever text-adapted here) are skipped."""
    names = set()
    for full_name, module in model.named_modules():
        cls_name = module.__class__.__name__
        if "Linear" not in cls_name:  # nn.Linear, bnb Linear4bit/Linear8bitLt, ...
            continue
        leaf = full_name.split(".")[-1]
        if leaf == "lm_head":
            continue
        if any(s in full_name.lower() for s in exclude_substrings):
            continue
        names.add(leaf)
    if not names:
        raise SystemExit("find_lora_target_modules found no Linear layers to "
                         "adapt — inspect the model structure manually.")
    return sorted(names)


class EpochLossCallback(TrainerCallback):
    """Prints a loss line every --logging_steps steps (via the Trainer's usual
    on_log printing) AND an averaged summary whenever an epoch finishes, so
    both granularities are visible regardless of --logging_strategy."""

    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.losses:
            avg = sum(self.losses) / len(self.losses)
            print(f"=== epoch {state.epoch:.2f} finished | avg train loss: "
                  f"{avg:.4f} (over {len(self.losses)} logged steps) ===")
            self.losses = []
        return control


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
    ap.add_argument("--warmup_ratio", type=float, default=0.03,
                    help="fraction of total steps spent ramping the LR from 0 "
                         "to --lr before cosine decay begins (default 0.03)")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument(
    "--load_4bit",
    action="store_true",
    help="Load the base model in 4-bit"
    )
    ap.add_argument(
    "--load_8bit",
    action="store_true",
    help="Load the base model in 8-bit"
    )
    ap.add_argument("--eval_fold", type=int, default=None,
                    help="hold out this single CV fold as the validation set for "
                         "loss curves; pass the *_train_full.jsonl files with this")
    ap.add_argument("--holdout_folds", nargs="+", type=int, default=None,
                    help="hold out these folds from training (e.g. --holdout_folds "
                         "3 4 for the 60/40 experiment split: trains on folds "
                         "0,1,2 + aux(fold=-1), holds out 3,4 for validation loss). "
                         "Overrides --eval_fold. Pass the *_train_full.jsonl files.")
    ap.add_argument("--eval_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=0,
                    help="save a checkpoint every N optimizer steps (0 = only at "
                         "epoch end). Use e.g. 50 on Colab to survive disconnects.")
    ap.add_argument("--save_total_limit", type=int, default=2,
                    help="keep only the newest N step-checkpoints (saves disk)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint-* in --output_dir")
    ap.add_argument("--logging_strategy", choices=["epoch", "steps"], default="steps",
                    help="'steps' (default) prints a loss line every "
                         "--logging_steps optimizer steps; 'epoch' prints only "
                         "once per epoch. Either way, an averaged per-epoch "
                         "summary is ALWAYS printed too (see EpochLossCallback)")
    ap.add_argument("--logging_steps", type=int, default=100,
                    help="print training loss every N steps (default 100)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    multimodal = is_multimodal_checkpoint(args.base_model)
    tok = load_tokenizer(args.base_model, multimodal)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = load_examples(args.train_files)
    eval_rows = []
    holdout = None
    if args.holdout_folds is not None:
        holdout = set(args.holdout_folds)
    elif args.eval_fold is not None:
        holdout = {args.eval_fold}
    if holdout is not None:
        eval_rows = [r for r in rows if r["fold"] in holdout]
        rows = [r for r in rows if r["fold"] not in holdout]
        if not eval_rows:
            raise SystemExit(
                f"holdout folds {sorted(holdout)} matched no examples. Pass the "
                f"*_train_full.jsonl files (the *_train_fold*.jsonl files already "
                f"exclude a fold).")
        print(f"holding out fold(s) {sorted(holdout)}: "
              f"{len(rows)} train / {len(eval_rows)} validation examples")

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

    if args.load_4bit and args.load_8bit:
        raise ValueError(
            "Choose only one quantization mode: "
            "--load_4bit or --load_8bit"
        )

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }

    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.pop("torch_dtype")

    elif args.load_8bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model_kwargs["torch_dtype"] = torch.float16

    model = load_base_model(args.base_model, model_kwargs, multimodal)

    model.config.use_cache = False

    if args.load_4bit or args.load_8bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        model.enable_input_require_grads()

    target_modules = resolve_target_modules(model)
    print(f"LoRA target modules ({len(target_modules)}): {target_modules}")
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=not args.load_8bit,
        fp16=args.load_8bit,
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
                      data_collator=PadCollator(tok.pad_token_id),
                      callbacks=[EpochLossCallback()])

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
