"""Train three independent country-specific LoRA adapters on one frozen model.

This is the Method-3 training entry point for PlurVA-LLM Track 1:

    zh_train_full.jsonl -> Chinese adapter
    id_train_full.jsonl -> Indonesian adapter
    si_train_full.jsonl -> Sri Lankan binary or four-way adapter

The input files must be produced by build_sft_data.py. The base checkpoint is
loaded once. For every country, a fresh LoRA is attached, trained only on that
country, saved, and unloaded before the next fresh LoRA is created. No adapter
weights are shared between countries.

Examples:

  python src/country_lora_pipeline.py train \
      --base_model Qwen/Qwen3.5-4B \
      --train_dir sft_data \
      --output_dir runs/country_lora/qwen35_4b_full \
      --load_4bit

  python src/country_lora_pipeline.py train \
      --base_model meta-llama/Llama-3.1-8B-Instruct \
      --train_dir sft_data \
      --output_dir runs/country_lora/llama31_8b_full \
      --load_4bit
"""

import argparse
import gc
import importlib.metadata
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COUNTRIES = (
    ("chinese", "zh", "zh_train_full.jsonl"),
    ("indonesian", "id", "id_train_full.jsonl"),
    ("sri_lankan", "si", "si_train_full.jsonl"),
)

TARGET_PATTERNS = {
    "chinese": re.compile(r"(?:^|\n)Answer: ([ABCD])\s*$"),
    "indonesian": re.compile(r"(?:^|\n)Answer: ([ABCD])\s*$"),
}
SI_TARGET_PATTERNS = {
    "binary": re.compile(r"(?:^|\n)Answer: (Yes|No)\s*$"),
    "4way": re.compile(r"(?:^|\n)Answer: (A|B|Both|0)\s*$"),
}


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Train independent Chinese, Indonesian, and Sri Lankan LoRAs")
    ap.add_argument("mode", choices=["train"])
    ap.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--train_dir", default=str(ROOT / "sft_data"))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--countries",
        nargs="+",
        choices=[dataset for dataset, _, _ in COUNTRIES],
        default=[dataset for dataset, _, _ in COUNTRIES],
        help="country adapters to train (default: all three); for example, "
             "--countries sri_lankan",
    )
    ap.add_argument(
        "--si_mode",
        choices=["binary", "4way"],
        default="binary",
        help="expected Sri Lankan SFT target format; must match the "
             "--si_mode used by build_sft_data.py",
    )
    ap.add_argument(
        "--oversample_si_3x",
        action="store_true",
        help="repeat the encoded Sri Lankan training rows to 3x their original "
             "count before shuffling. Default: disabled. Other countries are "
             "never oversampled by this option.",
    )
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--zh_epochs", type=float, default=None)
    ap.add_argument("--id_epochs", type=float, default=None)
    ap.add_argument("--si_epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--zh_lr", type=float, default=None)
    ap.add_argument("--id_lr", type=float, default=None)
    ap.add_argument("--si_lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=0,
                    help="0 saves at each epoch; otherwise save every N steps")
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume each country from its newest checkpoint-*")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip a country whose final adapter already exists")
    return ap


def validate_args(args):
    if args.load_4bit and args.load_8bit:
        raise ValueError("Choose only one of --load_4bit or --load_8bit")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora_dropout must be in [0, 1)")
    if args.epochs <= 0 or args.lr <= 0:
        raise ValueError("--epochs and --lr must be positive")
    for prefix in ("zh", "id", "si"):
        epochs = getattr(args, f"{prefix}_epochs")
        learning_rate = getattr(args, f"{prefix}_lr")
        if epochs is not None and epochs <= 0:
            raise ValueError(f"--{prefix}_epochs must be positive")
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError(f"--{prefix}_lr must be positive")
    if args.batch_size <= 0 or args.grad_accum <= 0 or args.max_len <= 0:
        raise ValueError("batch size, gradient accumulation, and max length "
                         "must be positive")
    if args.logging_steps <= 0 or args.save_steps < 0:
        raise ValueError("--logging_steps must be positive and --save_steps "
                         "cannot be negative")


def load_and_validate_sft(path, expected_dataset, si_mode="binary"):
    """Load one build_sft_data.py full file and validate its chat contract."""
    path = Path(path)
    if path.name != dict((ds, fn) for ds, _, fn in COUNTRIES)[expected_dataset]:
        raise ValueError(
            f"{expected_dataset}: expected its full-data file, got {path.name!r}")
    if not path.exists():
        raise FileNotFoundError(f"missing SFT file: {path}")

    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e

            messages = row.get("messages")
            meta = row.get("meta")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(
                    f"{path}:{line_no}: expected at least system/user/assistant "
                    "messages from build_sft_data.py")
            if not isinstance(meta, dict):
                raise ValueError(f"{path}:{line_no}: missing meta object")
            if meta.get("dataset") != expected_dataset:
                raise ValueError(
                    f"{path}:{line_no}: dataset {meta.get('dataset')!r}, expected "
                    f"{expected_dataset!r}")
            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"{path}:{line_no}: final message must be assistant")
            roles = {m.get("role") for m in messages[:-1]}
            if "system" not in roles or "user" not in roles:
                raise ValueError(
                    f"{path}:{line_no}: system/user messages are required")
            target = messages[-1].get("content")
            target_pattern = (
                SI_TARGET_PATTERNS[si_mode]
                if expected_dataset == "sri_lankan"
                else TARGET_PATTERNS[expected_dataset]
            )
            if not isinstance(target, str) or not target_pattern.search(target):
                if expected_dataset == "sri_lankan":
                    legal = ("Answer: Yes/No" if si_mode == "binary"
                             else "Answer: A/B/Both/0")
                else:
                    legal = "Answer: A/B/C/D"
                raise ValueError(
                    f"{path}:{line_no}: assistant target must end with {legal}")
            rows.append({"messages": messages, "uid": meta.get("uid")})

    if not rows:
        raise ValueError(f"{path}: no training examples")
    return rows


def package_versions():
    names = ("torch", "transformers", "peft", "datasets", "accelerate",
             "bitsandbytes")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_tokenizer_and_processor(model_name, trust_remote_code=False):
    from transformers import AutoProcessor, AutoTokenizer

    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
    except Exception:
        pass

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
    except Exception:
        pass

    if tok is None:
        tok = getattr(processor, "tokenizer", None)
    if tok is None:
        raise RuntimeError(f"could not load a tokenizer for {model_name!r}")
    return tok, processor


def load_base_model(model_name, model_kwargs):
    """Support text CausalLMs and multimodal Qwen 3.5 used text-only."""
    import transformers

    class_names = (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
    )
    errors = []
    for name in class_names:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model_name, **model_kwargs)
        except (ValueError, KeyError, OSError, TypeError) as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError(
        f"no Transformers auto-model class could load {model_name!r}. "
        "Qwen3.5 requires a recent Transformers release.\n" +
        "\n".join(errors))


def discover_lora_targets(model):
    """Find text-side linear leaf names while excluding vision and lm_head."""
    names = set()
    for full_name, module in model.named_modules():
        if "Linear" not in module.__class__.__name__:
            continue
        lower = full_name.lower()
        if "vision" in lower or "visual" in lower:
            continue
        leaf = full_name.split(".")[-1]
        if leaf == "lm_head":
            continue
        names.add(leaf)
    if not names:
        raise RuntimeError("no text Linear modules were found for LoRA")
    return sorted(names)


def merge_system_into_user(messages):
    if not messages or messages[0].get("role") != "system":
        return messages
    system_text = messages[0]["content"]
    rest = messages[1:]
    for i, message in enumerate(rest):
        if message.get("role") == "user":
            merged = dict(message)
            merged["content"] = f"{system_text}\n\n{message['content']}"
            return rest[:i] + [merged] + rest[i + 1:]
    return rest


def normalize_ids(ids):
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def template_prompt_ids(tok, processor, messages):
    kwargs = {"add_generation_prompt": True, "tokenize": True,
              "return_tensors": None}
    for obj in (tok, processor):
        if obj is None or not hasattr(obj, "apply_chat_template"):
            continue
        for candidate_messages in (messages, merge_system_into_user(messages)):
            try:
                try:
                    ids = obj.apply_chat_template(
                        candidate_messages, enable_thinking=False, **kwargs)
                except TypeError:
                    ids = obj.apply_chat_template(candidate_messages, **kwargs)
                return normalize_ids(ids)
            except Exception as e:
                if "system" in str(e).lower():
                    continue
                break
    raise RuntimeError(
        "neither tokenizer nor processor could render the chat template")


def encode_rows(tok, processor, rows, max_len):
    features = []
    skipped = []
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer has no eos_token_id")

    for row in rows:
        messages = row["messages"]
        prompt_ids = template_prompt_ids(tok, processor, messages[:-1])
        target_ids = list(tok.encode(
            messages[-1]["content"], add_special_tokens=False))
        target_ids.append(eos_id)
        input_ids = prompt_ids + target_ids
        if len(input_ids) > max_len:
            skipped.append(row.get("uid"))
            continue
        features.append({
            "input_ids": input_ids,
            "labels": [-100] * len(prompt_ids) + target_ids,
        })
    return features, skipped


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch

        max_len = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for feature in features:
            size = len(feature["input_ids"])
            padding = max_len - size
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_id] * padding)
            batch["labels"].append(
                feature["labels"] + [-100] * padding)
            batch["attention_mask"].append([1] * size + [0] * padding)
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in batch.items()}


def newest_checkpoint(output_dir):
    checkpoints = []
    for path in Path(output_dir).glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    return str(max(checkpoints)[1]) if checkpoints else None


def country_setting(args, prefix, name):
    value = getattr(args, f"{prefix}_{name}")
    return value if value is not None else getattr(args, name)


def adapter_is_complete(path):
    path = Path(path)
    return ((path / "adapter_config.json").exists() and
            ((path / "adapter_model.safetensors").exists() or
             (path / "adapter_model.bin").exists()))


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def run_train(args):
    validate_args(args)
    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)
    selected = set(args.countries)
    selected_countries = [
        country for country in COUNTRIES if country[0] in selected
    ]

    loaded_rows = {}
    output_paths = {}
    for dataset, prefix, filename in selected_countries:
        path = train_dir / filename
        loaded_rows[dataset] = load_and_validate_sft(
            path, dataset, si_mode=args.si_mode)
        output_paths[dataset] = output_dir / dataset
        if adapter_is_complete(output_paths[dataset]) and not (
                args.skip_existing or args.resume):
            raise FileExistsError(
                f"final adapter already exists: {output_paths[dataset]}. "
                "Use a new --output_dir or pass --skip_existing.")

    to_train = [
        dataset for dataset, _, _ in selected_countries
        if not ((args.skip_existing or args.resume) and adapter_is_complete(
            output_paths[dataset]))
    ]
    if not to_train:
        print("all selected final adapters already exist; nothing to train")
        return

    # Heavy dependencies are intentionally imported only after CLI/data checks.
    import torch
    from datasets import Dataset
    from peft import (LoraConfig, get_peft_model,
                      prepare_model_for_kbit_training)
    from transformers import Trainer, TrainingArguments

    has_cuda = torch.cuda.is_available()
    use_bf16 = bool(has_cuda and torch.cuda.is_bf16_supported())
    compute_dtype = (torch.bfloat16 if use_bf16 else
                     torch.float16 if has_cuda else torch.float32)
    if (args.load_4bit or args.load_8bit) and not has_cuda:
        raise RuntimeError("4-bit/8-bit training requires a CUDA GPU")

    tokenizer, processor = load_tokenizer_and_processor(
        args.base_model, args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": compute_dtype,
    }
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.pop("torch_dtype")
    elif args.load_8bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True)

    print(f"loading frozen base model once: {args.base_model}")
    base_model = load_base_model(args.base_model, model_kwargs)
    base_model.config.use_cache = False

    if args.load_4bit or args.load_8bit:
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True)
    else:
        base_model.enable_input_require_grads()

    target_modules = discover_lora_targets(base_model)
    print(f"LoRA target modules ({len(target_modules)}): {target_modules}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "independent_country_lora",
        "base_model": args.base_model,
        "quantization": ("4bit" if args.load_4bit else
                         "8bit" if args.load_8bit else
                         "bf16" if use_bf16 else "fp16"),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        },
        "seed": args.seed,
        "max_len": args.max_len,
        "selected_countries": [dataset for dataset, _, _ in selected_countries],
        "si_mode": args.si_mode,
        "oversample_si_3x": args.oversample_si_3x,
        "adapters": {},
        "package_versions": package_versions(),
    }

    for dataset, prefix, filename in selected_countries:
        country_dir = output_paths[dataset]
        if dataset not in to_train:
            print(f"[{dataset}] complete adapter exists; skipping")
            manifest["adapters"][dataset] = {
                "path": str(country_dir),
                "training_file": str(train_dir / filename),
                "skipped_existing": True,
            }
            continue

        epochs = country_setting(args, prefix, "epochs")
        learning_rate = country_setting(args, prefix, "lr")
        rows = loaded_rows[dataset]
        base_features, skipped = encode_rows(
            tokenizer, processor, rows, args.max_len)
        if not base_features:
            raise RuntimeError(
                f"{dataset}: every example exceeded --max_len {args.max_len}")
        oversample_factor = (
            3 if dataset == "sri_lankan" and args.oversample_si_3x else 1
        )
        features = base_features * oversample_factor

        print(f"\n[{dataset}] creating a fresh independent LoRA")
        row_note = (
            f"{len(base_features)} encoded -> {len(features)} training rows "
            f"(Sinhala 3x oversampling)"
            if oversample_factor == 3
            else f"{len(features)} encoded"
        )
        print(f"[{dataset}] {row_note} / {len(skipped)} skipped; "
              f"epochs={epochs}, lr={learning_rate}")

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()

        dataset_obj = Dataset.from_list(features).shuffle(seed=args.seed)
        country_dir.mkdir(parents=True, exist_ok=True)
        train_args = TrainingArguments(
            output_dir=str(country_dir),
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,
            bf16=use_bf16,
            fp16=has_cuda and not use_bf16,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            save_strategy="steps" if args.save_steps > 0 else "epoch",
            save_steps=args.save_steps if args.save_steps > 0 else 500,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset_obj,
            data_collator=PadCollator(tokenizer.pad_token_id),
        )

        resume_checkpoint = newest_checkpoint(country_dir) if args.resume else None
        if args.resume and resume_checkpoint:
            print(f"[{dataset}] resuming from {resume_checkpoint}")
        elif args.resume:
            print(f"[{dataset}] no checkpoint found; starting fresh")

        trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model(str(country_dir))
        tokenizer.save_pretrained(str(country_dir))
        write_json(country_dir / "log_history.json",
                   trainer.state.log_history)
        summary = {
            "dataset": dataset,
            "base_model": args.base_model,
            "training_file": str(train_dir / filename),
            "source_rows": len(rows),
            "encoded_rows": len(features),
            "encoded_rows_before_oversampling": len(base_features),
            "oversample_factor": oversample_factor,
            "skipped_rows": len(skipped),
            "skipped_uids": skipped,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "max_len": args.max_len,
            "seed": args.seed,
        }
        if dataset == "sri_lankan":
            summary["si_mode"] = args.si_mode
        write_json(country_dir / "training_summary.json", summary)
        manifest["adapters"][dataset] = {
            "path": str(country_dir),
            **summary,
        }
        write_json(output_dir / "manifest.json", manifest)
        print(f"[{dataset}] adapter saved to {country_dir}")

        # Drop optimizer/scheduler state, remove this LoRA without merging it,
        # and recover the exact same frozen base for the next fresh adapter.
        del trainer, dataset_obj, features
        base_model = model.unload()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(output_dir / "manifest.json", manifest)
    print(f"\nall requested country adapters complete -> {output_dir}")


def main():
    args = build_arg_parser().parse_args()
    if args.mode == "train":
        run_train(args)


if __name__ == "__main__":
    main()
