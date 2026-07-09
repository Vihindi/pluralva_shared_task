"""Stage-3 chain-of-thought self-consistency (§2.6; Wang et al. 2022).

For each item, sample n CoT chains at temperature t and aggregate the parsed
final answers into a probability distribution (vote shares). Position debiasing
is folded into the sampling: sample s uses cyclic option ordering s % n_perms,
and its vote is mapped back to the original letters — so the n chains are
spread evenly across orderings at no extra cost.

Sri Lankan items: n chains per binary judgment; p_yes = share of "Yes" votes,
composed with --si_threshold exactly like evaluate.py.

Outputs the same details format as evaluate.py (probs / p_yes per item), so
runs can be averaged with ensemble.py against constrained-scoring runs and
other adapters, and calibrated offline with tune_calibration.py.

  python src/self_consistency.py eval --model Qwen/Qwen3-8B \
      --adapter runs/joint_fold0 --fold 0 --n_samples 8
  python src/self_consistency.py predict --model Qwen/Qwen3-8B \
      --adapter runs/joint_full --n_samples 16 \
      --test_files test/chinese_test.jsonl test/indonesian_test.jsonl \
      test/sri_lankan_test.jsonl --out sc_predictions.jsonl
"""
import argparse
import collections
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import build_messages
from bootstrap_rationales import parse_answer
from build_sft_data import permute_record, CYCLIC_SHIFTS, load_jsonl
from evaluate import (is_correct, compute_prior, load_test_records,
                      TEST_PREPROCESSORS)

ROOT = Path(__file__).resolve().parent.parent
LETTERS4 = ["A", "B", "C", "D"]


class Sampler:
    def __init__(self, model_name, adapter=None, load_4bit=False,
                 max_new_tokens=400):
        from evaluate import resolve_base_model
        model_name = resolve_base_model(model_name, adapter)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
            kwargs.pop("torch_dtype")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def sample(self, messages, n, temperature):
        kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        try:
            encoded = self.tok.apply_chat_template(messages, enable_thinking=False,
                                                   **kwargs)
        except TypeError:
            encoded = self.tok.apply_chat_template(messages, **kwargs)
        input_ids = encoded["input_ids"] if hasattr(encoded, "input_ids") else encoded
        input_ids = input_ids.to(self.model.device)
        out = self.model.generate(
            input_ids, do_sample=True, temperature=temperature, top_p=0.95,
            num_return_sequences=n, max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.eos_token_id)
        plen = input_ids.shape[1]
        return [self.tok.decode(seq[plen:], skip_special_tokens=True) for seq in out]


def sc_mcq(sampler, rec, n_samples, temperature, n_perms):
    """Vote-share distribution over original letters; None-safe."""
    votes = collections.Counter()
    n_parsed = 0
    shifts = CYCLIC_SHIFTS[:max(1, n_perms)]
    per_shift = [n_samples // len(shifts) + (1 if s < n_samples % len(shifts) else 0)
                 for s in range(len(shifts))]
    for shift, n in zip(shifts, per_shift):
        if n == 0:
            continue
        p, old_to_new = permute_record(rec, shift)
        new_to_old = {v: k for k, v in old_to_new.items()}
        for text in sampler.sample(build_messages(p, mode="cot"), n, temperature):
            a = parse_answer(text)
            if a in LETTERS4:
                votes[new_to_old[a]] += 1
                n_parsed += 1
    if n_parsed == 0:
        return {l: 0.25 for l in LETTERS4}, 0
    return {l: votes.get(l, 0) / n_parsed for l in LETTERS4}, n_parsed


def sc_si(sampler, rec, n_samples, temperature):
    """p_yes per statement from vote shares."""
    p_yes = {}
    for stmt in ("A", "B"):
        msgs = build_messages(rec, mode="cot", si_statement=stmt)
        yes = n_parsed = 0
        for text in sampler.sample(msgs, n_samples, temperature):
            a = parse_answer(text)
            if a in ("Yes", "No"):
                n_parsed += 1
                yes += a == "Yes"
        p_yes[stmt] = yes / n_parsed if n_parsed else 0.5
    return p_yes


def compose_si(p_yes, threshold):
    a, b = p_yes["A"] >= threshold, p_yes["B"] >= threshold
    return "Both" if (a and b) else "A" if a else "B" if b else "0"


def calibrate(probs, prior, tau):
    if not prior or tau <= 0:
        return probs
    out = {k: v * (prior.get(k, 1e-9) ** tau) for k, v in probs.items()}
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


def process_records(args, sampler, recs, ds, prior, details, pred_out=None):
    n_ok = n_seen = 0
    for i, rec in enumerate(recs):
        if ds == "sri_lankan":
            p_yes = sc_si(sampler, rec, args.n_samples, args.temperature)
            pred = compose_si(p_yes, args.si_threshold)
            extra = {"p_yes_A": p_yes["A"], "p_yes_B": p_yes["B"]}
        else:
            probs, n_parsed = sc_mcq(sampler, rec, args.n_samples,
                                     args.temperature, args.n_perms)
            probs = calibrate(probs, prior, args.prior_tau)
            pred = max(probs, key=probs.get)
            extra = {"probs": probs, "n_parsed": n_parsed}
        row = {"uid": rec["uid"], "dataset": ds, "pred": pred, **extra}
        if pred_out is None:  # eval mode
            row["gold"] = rec.get("gold")
            row["correct"] = is_correct(rec, pred)
            n_ok += row["correct"]
            n_seen += 1
        else:
            pred_out.write(json.dumps({"dataset": args.dataset_names[ds],
                                       "id": rec["uid"], "LLM_Output": pred},
                                      ensure_ascii=False) + "\n")
        details.write(json.dumps(row, ensure_ascii=False) + "\n")
        details.flush()
        if (i + 1) % 25 == 0:
            msg = f"[{ds}] {i+1}/{len(recs)}"
            if n_seen:
                msg += f" acc so far {n_ok/n_seen:.3f}"
            print(msg)
    return n_ok, n_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "predict"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--datasets", nargs="+",
                    default=["chinese", "indonesian", "sri_lankan"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n_perms", type=int, default=4,
                    help="orderings the samples are spread across (ZH/ID)")
    ap.add_argument("--prior_tau", type=float, default=0.0)
    ap.add_argument("--si_threshold", type=float, default=0.5)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--test_files", nargs="+", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--details_out", default=None)
    args = ap.parse_args()
    args.dataset_names = {"chinese": "chinese", "indonesian": "indonesian",
                          "sri_lankan": "sri_lankan"}

    sampler = Sampler(args.model, adapter=args.adapter, load_4bit=args.load_4bit)

    if args.mode == "eval":
        details_path = Path(args.details_out or
                            (ROOT / "processed" / "sc_eval_details.jsonl"))
        details = open(details_path, "w", encoding="utf-8")
        results = {}
        for ds in args.datasets:
            recs = load_jsonl(Path(args.processed_dir) / f"{ds}.jsonl")
            prior = compute_prior(recs, exclude_fold=args.fold)
            eval_recs = [r for r in recs if args.fold is None or r["fold"] == args.fold]
            n_ok, n_seen = process_records(args, sampler, eval_recs, ds, prior, details)
            results[ds] = n_ok / n_seen if n_seen else float("nan")
            print(f"[{ds}] accuracy {results[ds]:.4f}  (n={n_seen})")
        details.close()
        macro = sum(results.values()) / len(results)
        print(f"\nMACRO-AVERAGE: {macro:.4f}   {results}")
        print(f"details -> {details_path}")
    else:
        assert args.test_files and args.out, "predict needs --test_files and --out"
        details_path = Path(args.details_out or (str(args.out) + ".details.jsonl"))
        details = open(details_path, "w", encoding="utf-8")
        pred_out = open(args.out, "w", encoding="utf-8")
        dev_priors = {}
        for ds in TEST_PREPROCESSORS:
            p = Path(args.processed_dir) / f"{ds}.jsonl"
            if p.exists():
                dev_priors[ds] = compute_prior(load_jsonl(p))
        for path in args.test_files:
            name = Path(path).name.lower()
            ds = ("chinese" if "chin" in name else
                  "indonesian" if "indo" in name else
                  "sri_lankan" if "sri" in name or "lank" in name else None)
            assert ds, f"cannot infer dataset from filename {name!r}"
            recs = load_test_records(path, ds)
            print(f"[{ds}] predicting {len(recs)} items")
            process_records(args, sampler, recs, ds, dev_priors.get(ds),
                            details, pred_out=pred_out)
        pred_out.close()
        details.close()
        print(f"predictions -> {args.out}\ndetails -> {details_path}")


if __name__ == "__main__":
    main()
