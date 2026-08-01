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
      --output_dir runs/joint_full --load_4bit

Pass --mixed_country_batches to mix all countries in every optimizer update:
2 physical rows x 10 gradient-accumulation passes = 20 rows, composed of
9 Indonesian, 7 Chinese, and 4 Sri Lankan rows. Without that option, joint
training uses the default row-level shuffle.
"""
import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainerCallback, TrainingArguments)

from mixed_microbatching import (OPTIMIZER_BLOCK_ROWS,
                                 build_mixed_optimizer_blocks,
                                 load_schedule_rows,
                                 mixed_optimizer_step_count,
                                 mixed_schedule_stats)

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


class DatasetWeightedTrainer(Trainer):
    """Apply per-token Sinhala loss weights during training only.

    The loss is recomputed without reduction so each row's weight applies to
    all and only its supervised assistant tokens. Held-out evaluation remains
    ordinary, unweighted cross-entropy so CV losses stay comparable.
    """

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        token_loss_weights = inputs.pop("token_loss_weights")
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = token_loss_weights[:, 1:].contiguous()
        batch_size, sequence_length = shift_labels.shape
        token_losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(batch_size, sequence_length)
        supervised = shift_labels.ne(-100)

        # Do not weight validation loss: it should retain its usual meaning
        # and remain comparable across folds and older runs.
        if model.training:
            weights = shift_weights.to(
                device=token_losses.device, dtype=token_losses.dtype)
        else:
            weights = torch.ones_like(token_losses)
        loss = (token_losses * supervised * weights).sum()
        loss = loss / supervised.sum().clamp_min(1)
        return (loss, outputs) if return_outputs else loss


def load_examples(paths):
    return load_schedule_rows(paths)


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


def final_answer_token_mask(tok, target_text, target_ids):
    """Boolean mask for token pieces in the final value after ``Answer:``."""
    match = re.search(
        r"(?:^|\n)Answer:\s*(?P<value>[^\r\n]*?\S)\s*\Z",
        target_text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            "answer-token weighting requires the assistant response to end "
            "with a line such as 'Answer: Yes'"
        )
    try:
        encoded = tok(
            target_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        offset_ids = encoded["input_ids"]
    except (KeyError, TypeError, NotImplementedError) as exc:
        raise ValueError(
            "--sinhala_answer_loss_weight requires a fast tokenizer that "
            "supports return_offsets_mapping"
        ) from exc
    if offset_ids != target_ids or len(offsets) != len(target_ids):
        raise RuntimeError(
            "token IDs from offset mapping do not match target tokenization"
        )
    value_start, value_end = match.span("value")
    mask = [
        start < value_end and end > value_start
        for start, end in offsets
    ]
    if not any(mask):
        raise RuntimeError(
            "the final answer value did not overlap any tokenizer token"
        )
    return mask


def encode_example(tok, messages, max_len, mark_answer_tokens=False):
    """-> {input_ids, labels} with prompt masked, or None if too long."""
    assert messages[-1]["role"] == "assistant", "last message must be assistant"
    prompt_ids = template_ids(tok, messages[:-1])
    target_text = messages[-1]["content"]
    target_ids = tok.encode(target_text, add_special_tokens=False)
    if mark_answer_tokens:
        target_answer_mask = final_answer_token_mask(
            tok, target_text, target_ids)
    else:
        target_answer_mask = [False] * len(target_ids)
    target_ids.append(tok.eos_token_id)
    target_answer_mask.append(False)  # EOS is not part of the answer value.
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "answer_token_mask": (
            [False] * len(prompt_ids) + target_answer_mask
        ),
    }


class MixedCountryDataset(torch.utils.data.IterableDataset):
    """Rebuild the deterministic 9-ID/7-ZH/4-SI schedule every epoch."""

    def __init__(self, features, seed):
        super().__init__()
        self.features = features
        self.seed = seed
        self.epoch = 0
        self.schedule_rows = (
            mixed_optimizer_step_count(features) * OPTIMIZER_BLOCK_ROWS)

    def __len__(self):
        return self.schedule_rows

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            raise RuntimeError(
                "MixedCountryDataset requires dataloader_num_workers=0")
        blocks = build_mixed_optimizer_blocks(
            self.features, seed=self.seed + self.epoch)
        self.epoch += 1
        for block in blocks:
            for index in block:
                yield self.features[index]


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        batch = {"input_ids": [], "labels": [], "attention_mask": [],
                 "token_loss_weights": []}
        for f in feats:
            n = len(f["input_ids"])
            pad = maxlen - n
            batch["input_ids"].append(list(f["input_ids"]) + [self.pad_id] * pad)
            batch["labels"].append(list(f["labels"]) + [-100] * pad)
            batch["attention_mask"].append([1] * n + [0] * pad)
            batch["token_loss_weights"].append(
                list(f["token_loss_weights"]) + [1.0] * pad)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(
                batch["attention_mask"], dtype=torch.long),
            "token_loss_weights": torch.tensor(
                batch["token_loss_weights"], dtype=torch.float32),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument(
        "--init_adapter",
        default=None,
        help="optional LoRA adapter used as the trainable initialization. "
             "The source adapter is read-only and training starts with a "
             "fresh optimizer/scheduler. Omit to create a new LoRA from the "
             "base model.",
    )
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--sinhala_loss_weight",
        type=float,
        default=1.0,
        help="training-loss multiplier for supervised assistant tokens in "
             "sri_lankan rows (default 1.0 = disabled; use 1.5 for the "
             "proposed joint-training weighting). Chinese and Indonesian "
             "always remain at 1.0.",
    )
    ap.add_argument(
        "--sinhala_answer_loss_weight",
        type=float,
        default=1.0,
        help="training-loss multiplier only for token piece(s) representing "
             "the final value after 'Answer:' in sri_lankan rows (default "
             "1.0 = disabled; recommended 1.5). All other assistant tokens "
             "remain at 1.0. Cannot be combined with a non-default "
             "--sinhala_loss_weight.",
    )
    ap.add_argument("--warmup_ratio", type=float, default=0.05,
                    help="fraction of total steps spent ramping the LR from 0 "
                         "to --lr before cosine decay begins (default 0.05)")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=10)
    ap.add_argument(
        "--zh_extra_file",
        default=str(ROOT / "sft_data" /
                    "zh_train_remaining_permutations.jsonl"),
        help="20 non-cyclic permutations per Chinese UID, used only after "
             "the primary Chinese rows are exhausted",
    )
    ap.add_argument(
        "--mixed_country_batches",
        action="store_true",
        help="joint training only: explicitly enable 9-ID/7-ZH/4-SI optimizer "
             "blocks. Default: disabled; use normal row-level shuffling.",
    )
    ap.add_argument(
        "--no_mixed_country_batches",
        action="store_true",
        help="deprecated compatibility flag; mixed-country batching is now "
             "disabled by default",
    )
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
    ap.add_argument(
        "--oversample_si_negation_3x",
        action="store_true",
        help="Sinhala-only training: repeat only rows whose UIDs occur in "
             "--si_negation_data to 3x their original count. Default: "
             "disabled.",
    )
    ap.add_argument(
        "--si_negation_data",
        default=str(ROOT / "negation_sinhala_data.jsonl"),
        help="JSONL file containing the negation UIDs used by "
             "--oversample_si_negation_3x",
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
    ap.add_argument("--save_total_limit", type=int, default=3,
                    help="keep only the newest N step-checkpoints (saves disk)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint-* in --output_dir")
    ap.add_argument("--logging_strategy", choices=["epoch", "steps"], default="steps",
                    help="'steps' (default) prints a loss line every "
                         "--logging_steps optimizer steps; 'epoch' prints only "
                         "once per epoch. Either way, an averaged per-epoch "
                         "summary is ALWAYS printed too (see EpochLossCallback)")
    ap.add_argument("--logging_steps", type=int, default=10,
                    help="print training loss every N steps (default 10)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.sinhala_loss_weight <= 0:
        raise ValueError("--sinhala_loss_weight must be greater than zero")
    if args.sinhala_answer_loss_weight <= 0:
        raise ValueError(
            "--sinhala_answer_loss_weight must be greater than zero")
    if (args.sinhala_loss_weight != 1.0 and
            args.sinhala_answer_loss_weight != 1.0):
        raise ValueError(
            "choose either whole-response --sinhala_loss_weight or "
            "--sinhala_answer_loss_weight, not both")

    if args.init_adapter and args.resume:
        raise ValueError(
            "--init_adapter and --resume are mutually exclusive: use "
            "--init_adapter for a fresh optimizer over existing adapter "
            "weights, or --resume to restore an output checkpoint including "
            "optimizer/scheduler state")
    if args.init_adapter:
        init_path = Path(args.init_adapter)
        output_path = Path(args.output_dir)
        if init_path.exists():
            init_resolved = init_path.resolve()
            output_resolved = output_path.resolve()
            if (init_resolved == output_resolved or
                    output_resolved.is_relative_to(init_resolved) or
                    init_resolved.is_relative_to(output_resolved)):
                raise ValueError(
                    "--init_adapter and --output_dir must be separate, "
                    "non-nested directories so the source adapter is never "
                    "modified or overwritten")
        elif str(init_path) == str(output_path):
            raise ValueError(
                "--init_adapter and --output_dir must be different")

    multimodal = is_multimodal_checkpoint(args.base_model)
    tok = load_tokenizer(args.base_model, multimodal)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = load_examples(args.train_files)
    datasets_present = {row["dataset"] for row in rows}
    weighting_enabled = (
        args.sinhala_loss_weight != 1.0 or
        args.sinhala_answer_loss_weight != 1.0
    )
    if weighting_enabled:
        if "sri_lankan" not in datasets_present:
            print("WARNING: Sinhala loss weighting has no effect because no "
                  "sri_lankan rows were supplied")
        elif (datasets_present == {"sri_lankan"} and
              args.sinhala_loss_weight != 1.0):
            print("WARNING: all rows are Sinhala, so weighting every loss by "
                  f"{args.sinhala_loss_weight} changes the overall gradient "
                  "scale rather than its share relative to other countries")
    if args.mixed_country_batches and args.no_mixed_country_batches:
        raise ValueError(
            "Choose only one of --mixed_country_batches or "
            "--no_mixed_country_batches")
    if (args.oversample_si_negation_3x and
            datasets_present != {"sri_lankan"}):
        raise ValueError(
            "--oversample_si_negation_3x is supported only when "
            "--train_files contains monolingual Sinhala SFT data")
    mixed_batches = (
        args.mixed_country_batches and
        {"chinese", "indonesian", "sri_lankan"}.issubset(datasets_present)
    )
    extra_rows = []
    if mixed_batches:
        if args.batch_size * args.grad_accum != OPTIMIZER_BLOCK_ROWS:
            raise ValueError(
                "mixed batching requires --batch_size * --grad_accum == "
                f"{OPTIMIZER_BLOCK_ROWS} ({args.batch_size} * "
                f"{args.grad_accum} != {OPTIMIZER_BLOCK_ROWS})")
        extra_path = Path(args.zh_extra_file)
        if not extra_path.exists():
            raise SystemExit(
                f"Chinese remaining-permutation file not found: {extra_path}\n"
                "Generate it with:\n"
                "  python src/build_sft_data.py --n_perms 4 "
                "--only_zh_remaining_permutations")
        extra_rows = load_examples([extra_path])
    elif args.mixed_country_batches:
        print("not all three countries were supplied -> using normal "
              "row-level shuffle")
    else:
        print("mixed-country batching disabled (default) -> using normal "
              "row-level shuffle")

    eval_rows = []
    holdout = None
    if args.holdout_folds is not None:
        holdout = set(args.holdout_folds)
    elif args.eval_fold is not None:
        holdout = {args.eval_fold}
    if holdout is not None:
        eval_rows = [r for r in rows if r["fold"] in holdout]
        rows = [r for r in rows if r["fold"] not in holdout]
        extra_rows = [r for r in extra_rows if r["fold"] not in holdout]
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
            use_answer_weight = (
                r["dataset"] == "sri_lankan" and
                args.sinhala_answer_loss_weight != 1.0
            )
            enc = encode_example(
                tok,
                r["messages"],
                args.max_len,
                mark_answer_tokens=use_answer_weight,
            )
            if enc is None:
                skipped += 1
            else:
                token_loss_weights = [1.0] * len(enc["labels"])
                if r["dataset"] == "sri_lankan":
                    if args.sinhala_loss_weight != 1.0:
                        token_loss_weights = [
                            args.sinhala_loss_weight if label != -100 else 1.0
                            for label in enc["labels"]
                        ]
                    elif args.sinhala_answer_loss_weight != 1.0:
                        token_loss_weights = [
                            args.sinhala_answer_loss_weight if is_answer else 1.0
                            for is_answer in enc["answer_token_mask"]
                        ]
                enc.update({
                    "uid": r["uid"],
                    "dataset": r["dataset"],
                    "messages": r["messages"],
                    "augmentation_source": r.get(
                        "augmentation_source", "primary"),
                    "permutation_order": r.get("permutation_order"),
                    "token_loss_weights": token_loss_weights,
                })
                enc.pop("answer_token_mask")
                feats.append(enc)
        return feats, skipped

    feats, skipped = encode_all(rows)
    print(f"{len(feats)} training examples from {len(args.train_files)} file(s)"
          f" ({skipped} skipped as longer than {args.max_len} tokens)")
    if args.sinhala_answer_loss_weight != 1.0:
        print("training loss mode: Sinhala final answer value token(s)="
              f"{args.sinhala_answer_loss_weight}; every other token=1.0")
    elif args.sinhala_loss_weight != 1.0:
        print("training loss mode: every supervised Sinhala token="
              f"{args.sinhala_loss_weight}; Chinese/Indonesian=1.0")
    else:
        print("training loss mode: standard unweighted loss (all tokens=1.0)")
    if eval_rows:
        print("held-out evaluation loss is unweighted for comparability")
    if args.oversample_si_negation_3x:
        negation_path = Path(args.si_negation_data)
        if not negation_path.exists():
            raise FileNotFoundError(
                f"missing Sinhala negation data: {negation_path}")
        negation_uids = set()
        with open(negation_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    negation_row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{negation_path}:{line_no}: invalid JSON: {e}") from e
                uid = negation_row.get("uid")
                if not isinstance(uid, str) or not uid:
                    raise ValueError(
                        f"{negation_path}:{line_no}: missing string uid")
                if uid in negation_uids:
                    raise ValueError(
                        f"{negation_path}:{line_no}: duplicate uid {uid!r}")
                negation_uids.add(uid)
        negation_feats = [
            feature for feature in feats
            if feature.get("uid") in negation_uids
        ]
        if not negation_feats:
            raise RuntimeError(
                "none of the UIDs in --si_negation_data occur in the Sinhala "
                "SFT rows; rebuild SFT with --si_dev_aug_files first")
        original_count = len(feats)
        feats = feats + negation_feats * 2
        print(
            f"Sinhala negation-only 3x oversampling: {len(negation_feats)} "
            f"matching rows repeated to 3x; {original_count} -> "
            f"{len(feats)} total training rows")
    extra_feats, extra_skipped = encode_all(extra_rows)
    if mixed_batches:
        print(
            f"{len(extra_feats)} Chinese remaining-permutation fallback "
            f"examples ({extra_skipped} skipped)")
    if mixed_batches and (skipped or extra_skipped):
        raise SystemExit(
            "mixed batching cannot silently drop over-length rows because "
            "coverage accounting would become ambiguous. Increase --max_len.")
    if mixed_batches:
        schedule_feats = feats + extra_feats
        preview_blocks = build_mixed_optimizer_blocks(
            schedule_feats, seed=args.seed)
        stats = mixed_schedule_stats(schedule_feats, preview_blocks)
        print("mixed optimizer blocks (9 Indonesian / 7 Chinese / 4 Sinhala):")
        print(f"  rows/epoch: {stats['rows']}")
        print(f"  Chinese sources: {stats['chinese_sources']}")
        print(
            f"  total: {stats['optimizer_blocks']} optimizer blocks/epoch; "
            f"physical batch={args.batch_size}, accumulation={args.grad_accum}")
        ds = MixedCountryDataset(schedule_feats, args.seed)
    else:
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

    if args.init_adapter:
        init_config = PeftConfig.from_pretrained(args.init_adapter)
        peft_type = str(getattr(init_config, "peft_type", "")).upper()
        if "LORA" not in peft_type:
            raise ValueError(
                f"--init_adapter must be a LoRA adapter, got "
                f"{getattr(init_config, 'peft_type', None)!r}")
        configured_base = getattr(
            init_config, "base_model_name_or_path", None)
        if configured_base:
            expected = str(configured_base).replace("\\", "/").rstrip("/").casefold()
            supplied = str(args.base_model).replace("\\", "/").rstrip("/").casefold()
            if expected != supplied:
                raise ValueError(
                    "base-model mismatch: --init_adapter was trained on "
                    f"{configured_base!r}, but --base_model is "
                    f"{args.base_model!r}")
        adapter_r = getattr(init_config, "r", None)
        adapter_alpha = getattr(init_config, "lora_alpha", None)
        adapter_dropout = getattr(init_config, "lora_dropout", None)
        mismatches = []
        if adapter_r is not None and adapter_r != args.lora_r:
            mismatches.append(f"rank {adapter_r} != --lora_r {args.lora_r}")
        if adapter_alpha is not None and adapter_alpha != args.lora_alpha:
            mismatches.append(
                f"alpha {adapter_alpha} != --lora_alpha {args.lora_alpha}")
        if (adapter_dropout is not None and
                abs(float(adapter_dropout) - args.lora_dropout) > 1e-12):
            mismatches.append(
                f"dropout {adapter_dropout} != --lora_dropout "
                f"{args.lora_dropout}")
        if mismatches:
            raise ValueError(
                "LoRA configuration mismatch for --init_adapter: " +
                "; ".join(mismatches))
        print(f"loading trainable LoRA initialization: {args.init_adapter}")
        print(
            "starting a fresh optimizer and scheduler; source adapter remains "
            "unchanged")
        model = PeftModel.from_pretrained(
            model, args.init_adapter, is_trainable=True)
        target_modules = sorted(getattr(init_config, "target_modules", []) or [])
    else:
        target_modules = resolve_target_modules(model)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    print(f"LoRA target modules ({len(target_modules)}): {target_modules}")
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
        max_grad_norm=1.0,
        weight_decay=0.0,
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
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        eval_strategy="steps" if eval_rows else "no",
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=max(2, args.batch_size),
    )

    trainer = DatasetWeightedTrainer(
        model=model, args=train_args, train_dataset=ds,
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
    summary_path = Path(args.output_dir) / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_model": args.base_model,
            "init_adapter": args.init_adapter,
            "fresh_optimizer": not args.resume,
            "train_files": args.train_files,
            "datasets": sorted(datasets_present),
            "training_rows": len(feats),
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "loss_function": "token_weighted_assistant_cross_entropy",
            "sinhala_loss_mode": (
                "answer_value_tokens"
                if args.sinhala_answer_loss_weight != 1.0 else
                "whole_assistant_response"
                if args.sinhala_loss_weight != 1.0 else
                "disabled"
            ),
            "dataset_loss_weights": {
                "chinese": 1.0,
                "indonesian": 1.0,
                "sri_lankan_whole_response": args.sinhala_loss_weight,
                "sri_lankan_answer_value":
                    args.sinhala_answer_loss_weight,
            },
            "evaluation_loss_weighted": False,
            "warmup_ratio": args.warmup_ratio,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "max_len": args.max_len,
            "mixed_country_batches": mixed_batches,
            "oversample_si_negation_3x":
                args.oversample_si_negation_3x,
            "seed": args.seed,
        }, f, ensure_ascii=False, indent=2)
    print(f"adapter saved to {args.output_dir}")
    print(f"loss history -> {hist_path}")
    print(f"training summary -> {summary_path}")


if __name__ == "__main__":
    main()
