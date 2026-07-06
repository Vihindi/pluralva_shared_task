"""Constrained evaluation + prediction for PlurVA-LLM Track 1.

Scoring (methodology §2.2-§2.3): for each item we score the sequence logprob of
the continuations "Answer: A".."Answer: D" (or "Answer: Yes"/"Answer: No") after
the chat generation prompt — never free generation, so no invalid outputs.
Then, optionally:
  * permutation ensemble (--n_perms 4): average content-level probabilities
    over cyclic option orderings (removes position/token selection bias without
    assuming a uniform answer prior);
  * prior calibration (--prior_tau t): multiply averaged probs by
    prior(label)^t estimated from the training folds, renormalize (PriDe-style
    prior separation, but with the true skewed dev prior instead of uniform).

Sri Lankan items are scored via binary decomposition: P(Yes) per statement,
threshold (default 0.5, tunable via --si_threshold) composes A/B/Both/0.

Modes
  eval    : accuracy on processed dev data (optionally one CV fold) per country + macro
  predict : emit submission-format predictions.jsonl from a test JSONL in the
            same schema as the dev files (Gold_Answer may be absent)

  python src/evaluate.py eval --model Qwen/Qwen3-8B --fold 0 --load_4bit
  python src/evaluate.py eval --model Qwen/Qwen3-8B --adapter runs/joint_fold0 --fold 0
  python src/evaluate.py predict --model Qwen/Qwen3-8B --adapter runs/joint_full \
      --test_files test/chinese_test.jsonl test/indonesian_test.jsonl \
      test/sri_lankan_test.jsonl --out predictions.jsonl
"""
import argparse
import collections
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import build_messages
from build_sft_data import permute_record, CYCLIC_SHIFTS
from preprocess import (preprocess_chinese, preprocess_indonesian,
                        preprocess_sri_lankan, load_jsonl)

ROOT = Path(__file__).resolve().parent.parent
LETTERS4 = ["A", "B", "C", "D"]


class Scorer:
    def __init__(self, model_name, adapter=None, load_4bit=False, batch_size=8):
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
        self.batch_size = batch_size

    def _prompt_ids(self, messages):
        kwargs = {"add_generation_prompt": True, "tokenize": True, "return_tensors": None}
        try:
            ids = self.tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            ids = self.tok.apply_chat_template(messages, **kwargs)
        # transformers ≥4.51 may return BatchEncoding even without return_tensors
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    @torch.no_grad()
    def score_candidates(self, messages, candidates):
        """Return log P(candidate tokens | prompt) for each candidate string."""
        prompt_ids = self._prompt_ids(messages)
        device = self.model.device
        out = []
        for cand in candidates:
            cand_ids = self.tok.encode(cand, add_special_tokens=False)
            ids = prompt_ids + cand_ids
            s, e = len(prompt_ids), len(ids)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attn = torch.ones_like(input_ids)
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            lp = 0.0
            for t in range(s, e):
                # one vocab row at a time — avoids O(batch × seq × vocab) fp32 tensor
                tok_lp = torch.log_softmax(logits[0, t - 1], dim=-1)
                lp += tok_lp[input_ids[0, t]].item()
            out.append(lp)
            del logits, input_ids, attn
        return out


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def score_mcq(scorer, rec, n_perms=1, prior=None, prior_tau=0.0):
    """Return dict original_letter -> prob, permutation-averaged and calibrated."""
    acc = {ltr: 0.0 for ltr in LETTERS4}
    shifts = CYCLIC_SHIFTS[:n_perms]
    for shift in shifts:
        p, old_to_new = permute_record(rec, shift)
        messages = build_messages(p, mode="direct")
        lps = scorer.score_candidates(messages, [f"Answer: {l}" for l in LETTERS4])
        probs = dict(zip(LETTERS4, softmax(lps)))
        for old, new in old_to_new.items():
            acc[old] += probs[new] / len(shifts)
    if prior and prior_tau > 0:
        acc = {k: v * (prior.get(k, 1e-9) ** prior_tau) for k, v in acc.items()}
        z = sum(acc.values())
        acc = {k: v / z for k, v in acc.items()}
    return acc


def score_si(scorer, rec, threshold=0.5):
    """Binary-decomposed Sri Lankan scoring -> (label, p_yes_A, p_yes_B)."""
    p_yes = {}
    for stmt in ("A", "B"):
        messages = build_messages(rec, mode="direct", si_statement=stmt)
        lps = scorer.score_candidates(messages, ["Answer: Yes", "Answer: No"])
        p_yes[stmt] = softmax(lps)[0]
    a, b = p_yes["A"] >= threshold, p_yes["B"] >= threshold
    label = "Both" if (a and b) else "A" if a else "B" if b else "0"
    return label, p_yes["A"], p_yes["B"]


def is_correct(rec, pred):
    if rec["dataset"] == "indonesian":
        return pred in set(rec["consensus"])
    return pred == rec["gold"]


def compute_prior(records, exclude_fold=None):
    recs = [r for r in records if exclude_fold is None or r["fold"] != exclude_fold]
    c = collections.Counter(r["gold"] for r in recs)
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def run_eval(args, scorer):
    results = {}
    details_path = Path(args.out or (ROOT / "processed" / "eval_details.jsonl"))
    details = open(details_path, "w", encoding="utf-8")
    for ds in args.datasets:
        recs = load_jsonl(Path(args.processed_dir) / f"{ds}.jsonl")
        prior = compute_prior(recs, exclude_fold=args.fold)
        eval_recs = [r for r in recs if args.fold is None or r["fold"] == args.fold]
        n_ok = 0
        for i, rec in enumerate(eval_recs):
            if ds == "sri_lankan":
                pred, pa, pb = score_si(scorer, rec, threshold=args.si_threshold)
                extra = {"p_yes_A": pa, "p_yes_B": pb}
            else:
                probs = score_mcq(scorer, rec, n_perms=args.n_perms,
                                  prior=prior, prior_tau=args.prior_tau)
                pred = max(probs, key=probs.get)
                extra = {"probs": probs}
            ok = is_correct(rec, pred)
            n_ok += ok
            details.write(json.dumps({"uid": rec["uid"], "dataset": ds, "pred": pred,
                                      "gold": rec.get("gold"), "correct": ok, **extra},
                                     ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                print(f"[{ds}] {i+1}/{len(eval_recs)} acc so far {n_ok/(i+1):.3f}")
        results[ds] = n_ok / len(eval_recs) if eval_recs else float("nan")
        print(f"[{ds}] accuracy {results[ds]:.4f}  (n={len(eval_recs)})")
    details.close()
    macro = sum(results.values()) / len(results)
    print(f"\nMACRO-AVERAGE: {macro:.4f}   {results}")
    print(f"per-item details -> {details_path}")


TEST_PREPROCESSORS = {"chinese": preprocess_chinese,
                      "indonesian": preprocess_indonesian,
                      "sri_lankan": preprocess_sri_lankan}


def load_test_records(path, ds):
    """Parse a test JSONL (dev schema, Gold_Answer possibly absent/empty)."""
    raw = load_jsonl(Path(path))
    for x in raw:  # make gold optional for the dev-style preprocessors
        x.setdefault("Gold_Answer", "A" if ds != "indonesian" else "A, A, A, A, A")
        if not str(x["Gold_Answer"]).strip():
            x["Gold_Answer"] = "A" if ds != "indonesian" else "A, A, A, A, A"
    recs, problems = TEST_PREPROCESSORS[ds](raw)
    for uid, why in problems:
        print(f"  WARNING test item {uid}: {why} (will still be predicted if possible)")
    for r in recs:
        r["fold"] = -1
    return recs


def run_predict(args, scorer):
    # priors always come from the full dev data, never from test
    dev_priors = {}
    for ds in TEST_PREPROCESSORS:
        p = Path(args.processed_dir) / f"{ds}.jsonl"
        if p.exists():
            dev_priors[ds] = compute_prior(load_jsonl(p))
    out = open(args.out, "w", encoding="utf-8")
    for path in args.test_files:
        name = Path(path).name.lower()
        ds = ("chinese" if "chin" in name else
              "indonesian" if "indo" in name else
              "sri_lankan" if "sri" in name or "lank" in name else None)
        assert ds, f"cannot infer dataset from filename {name!r}; rename the file"
        recs = load_test_records(path, ds)
        print(f"[{ds}] predicting {len(recs)} items from {path}")
        for i, rec in enumerate(recs):
            if ds == "sri_lankan":
                pred, _, _ = score_si(scorer, rec, threshold=args.si_threshold)
            else:
                probs = score_mcq(scorer, rec, n_perms=args.n_perms,
                                  prior=dev_priors.get(ds), prior_tau=args.prior_tau)
                pred = max(probs, key=probs.get)
            out.write(json.dumps({"dataset": args.dataset_names[ds],
                                  "id": rec["uid"], "LLM_Output": pred},
                                 ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(recs)}")
    out.close()
    print(f"predictions -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "predict"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--datasets", nargs="+",
                    default=["chinese", "indonesian", "sri_lankan"])
    ap.add_argument("--fold", type=int, default=None,
                    help="eval only this CV fold (use with the matching fold adapter)")
    ap.add_argument("--n_perms", type=int, default=1,
                    help="cyclic option permutations to ensemble (ZH/ID)")
    ap.add_argument("--prior_tau", type=float, default=0.0,
                    help="0 = no prior calibration; try 0.25-1.0 on CV folds")
    ap.add_argument("--si_threshold", type=float, default=0.5)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--test_files", nargs="+", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # submission "dataset" field names — adjust if organizers specify others
    args.dataset_names = {"chinese": "chinese", "indonesian": "indonesian",
                          "sri_lankan": "sri_lankan"}

    scorer = Scorer(args.model, adapter=args.adapter, load_4bit=args.load_4bit)
    if args.mode == "eval":
        run_eval(args, scorer)
    else:
        assert args.test_files and args.out, "predict needs --test_files and --out"
        run_predict(args, scorer)


if __name__ == "__main__":
    main()
