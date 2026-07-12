"""Stage-2 GRPO with a verifiable exact-match reward (§2.5; Shao et al. 2024).

Policy samples CoT completions for value-conditioned prompts; reward is
computed from the parsed final "Answer: X" line:
    +1.0  correct (Chinese/SI: exact; Indonesian: lands in the consensus set)
    +0.1  parseable-format bonus (teaches the output contract)
     0.0  otherwise
Group-relative advantages (GRPO) need no value model, so an 8B policy + LoRA
fits on a single A100-40GB (or 24GB with --load_4bit and fewer generations).

Length-hacking guard (methodology §5): the ZH gold correlates with option
length, and RL can drift toward always-longest. The trainer logs the rate at
which sampled answers equal the longest option; if it climbs well above the
dev rate (~52%), lower --epochs / raise --beta.

  python src/train_grpo.py --datasets chinese --sft_adapter runs/joint_fold0 \
      --fold 0 --output_dir runs/grpo_zh_fold0
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from prompts import build_messages
from bootstrap_rationales import parse_answer
from build_sft_data import load_jsonl

ROOT = Path(__file__).resolve().parent.parent
LETTERS4 = ["A", "B", "C", "D"]


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


def build_rows(processed_dir, datasets, fold):
    """One prompt row per training target; gold_info drives the reward."""
    rows = []
    for ds in datasets:
        for rec in load_jsonl(Path(processed_dir) / f"{ds}.jsonl"):
            if fold is not None and rec["fold"] == fold:
                continue  # held-out CV fold never seen in training
            if ds == "sri_lankan":
                for stmt, ok in (("A", rec["stmt_A_ok"]), ("B", rec["stmt_B_ok"])):
                    rows.append({
                        "prompt": build_messages(rec, mode="cot", si_statement=stmt),
                        "gold_info": json.dumps({"accept": ["Yes" if ok else "No"]}),
                        "longest": "",
                    })
            else:
                accept = rec["consensus"] if ds == "indonesian" else [rec["gold"]]
                longest = max(LETTERS4, key=lambda l: len(rec["options"][l]))
                rows.append({
                    "prompt": build_messages(rec, mode="cot"),
                    "gold_info": json.dumps({"accept": accept}),
                    "longest": longest,
                })
    return rows


class RewardStats:
    """Running counters for the length-hacking guard."""
    def __init__(self):
        self.n = self.n_longest = 0


STATS = RewardStats()


def make_reward_fn():
    def reward(completions, gold_info, longest, **kwargs):
        rewards = []
        for comp, gi, lg in zip(completions, gold_info, longest):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            ans = parse_answer(text)
            accept = set(json.loads(gi)["accept"])
            r = 0.0
            if ans is not None:
                r += 0.1
                if ans in accept:
                    r += 1.0
                if lg:  # MCQ only: track drift toward the longest option
                    STATS.n += 1
                    STATS.n_longest += ans == lg
            rewards.append(r)
        if STATS.n and STATS.n % 512 < len(completions):
            print(f"  [guard] longest-option pick rate: "
                  f"{STATS.n_longest / STATS.n:.3f} over {STATS.n} answers")
        return rewards
    return reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B")
    ap.add_argument("--sft_adapter", default=None,
                    help="Stage-1 adapter to continue from (recommended)")
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--datasets", nargs="+", default=["chinese"],
                    help="GRPO is most useful for chinese; id/si also supported")
    ap.add_argument("--fold", type=int, default=None, help="CV fold to hold out")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="prompts per step x num_generations completions")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_completion", type=int, default=400)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--save_steps", type=int, default=0,
                    help="save a checkpoint every N optimizer steps (0 = only at "
                         "epoch end). Use e.g. 20 on Colab to survive disconnects.")
    ap.add_argument("--save_total_limit", type=int, default=2,
                    help="keep only the newest N step-checkpoints (saves disk)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint-* in --output_dir")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.sft_adapter:
        args.base_model = resolve_base_model(args.base_model, args.sft_adapter)

    rows = build_rows(args.processed_dir, args.datasets, args.fold)
    print(f"{len(rows)} GRPO prompts ({args.datasets}, fold held out: {args.fold})")
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

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        temperature=0.9,
        logging_steps=5,
        save_strategy="steps" if args.save_steps > 0 else "epoch",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        report_to="none",
    )

    trainer = GRPOTrainer(model=model, reward_funcs=make_reward_fn(),
                          args=grpo_config, train_dataset=ds,
                          processing_class=tok, peft_config=peft_config)

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
    print(f"GRPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
