"""Stage-2 DPO on top of a Stage-1 SFT adapter (§2.5; Rafailov et al. 2023).

The policy is the base model + the SFT LoRA adapter (loaded trainable and
updated in place). No separate reference model is materialized: when the policy
is a PeftModel and ref_model=None, trl computes reference logprobs with the
adapter disabled — i.e. the frozen base — which keeps memory at 1x model.
If you prefer the SFT checkpoint itself as the reference (tighter KL anchor),
merge it first and pass --sft_adapter on the merged model with a fresh LoRA.

  # Chinese-only preference tuning on CV fold 0, starting from the joint SFT adapter:
  python src/train_dpo.py --train_files dpo_data/zh_dpo_fold0.jsonl \
      --sft_adapter runs/joint_fold0 --output_dir runs/dpo_zh_fold0
  # final: all countries, all data
  python src/train_dpo.py --train_files dpo_data/zh_dpo_full.jsonl \
      dpo_data/id_dpo_full.jsonl dpo_data/si_dpo_full.jsonl \
      --sft_adapter runs/joint_full --output_dir runs/dpo_joint_full
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parent.parent


def resolve_base_model(model_name, adapter):
    """A LoRA adapter only loads onto the exact base it was trained on; if the
    adapter records a different base than --base_model, trust the adapter."""
    if adapter:
        cfg_path = Path(adapter) / "adapter_config.json"
        if cfg_path.exists():
            recorded = json.load(open(cfg_path, encoding="utf-8")).get(
                "base_model_name_or_path")
            if recorded and recorded != model_name:
                print(f"WARNING: adapter was trained on {recorded!r}, not "
                      f"{model_name!r} — loading {recorded!r} instead.")
                return recorded
    return model_name


def load_pairs(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.append({"prompt": r["prompt"], "chosen": r["chosen"],
                                 "rejected": r["rejected"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--sft_adapter", default=None,
                    help="Stage-1 LoRA adapter to continue from (recommended)")
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO KL strength")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--save_steps", type=int, default=0,
                    help="save a checkpoint every N optimizer steps (0 = only at "
                         "epoch end). Use e.g. 50 on Colab to survive disconnects.")
    ap.add_argument("--save_total_limit", type=int, default=2,
                    help="keep only the newest N step-checkpoints (saves disk)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint-* in --output_dir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.sft_adapter:
        args.base_model = resolve_base_model(args.base_model, args.sft_adapter)

    rows = load_pairs(args.train_files)
    print(f"{len(rows)} preference pairs from {len(args.train_files)} file(s)")
    ds = Dataset.from_list(rows).shuffle(seed=args.seed)

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

    peft_config = None
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM")

    # desired config — filtered below to whatever this TRL version's DPOConfig
    # actually accepts (arg names like max_prompt_length drift across versions)
    want = dict(
        output_dir=args.output_dir,
        beta=args.beta,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        max_length=args.max_len,
        max_prompt_length=args.max_len - 16,
        logging_steps=10,
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to="none",
    )
    import inspect
    supported = set(inspect.signature(DPOConfig.__init__).parameters)
    dropped = [k for k in want if k not in supported]
    if dropped:
        print(f"note: DPOConfig in this TRL version ignores {dropped}")
    dpo_config = DPOConfig(**{k: v for k, v in want.items() if k in supported})

    trainer = DPOTrainer(model=model, ref_model=None, args=dpo_config,
                         train_dataset=ds, processing_class=tok,
                         peft_config=peft_config)

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
    print(f"DPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
