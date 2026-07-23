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
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import build_messages
from build_sft_data import permute_record, CYCLIC_SHIFTS
from preprocess import (preprocess_chinese, preprocess_indonesian,
                        preprocess_sri_lankan, load_jsonl)

ROOT = Path(__file__).resolve().parent.parent
LETTERS4 = ["A", "B", "C", "D"]


def _load_tokenizer(model_name, trust_remote_code=False):
    """Return (tokenizer, processor). The tokenizer is used for .encode/.decode
    and eos; the processor (if any) is what actually holds a working chat
    template for multimodal repos like Gemma 3/3n/4 — where processor.chat_template
    the *attribute* is None but processor.apply_chat_template() still works
    (it loads chat_template.jinja internally). We therefore keep the processor
    object and call it directly, rather than trying to copy the template string."""
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
    except Exception:
        tok = None
    proc = None
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
    except Exception:
        proc = None
    if tok is None:  # base tokenizer unavailable — fall back to the processor's
        tok = getattr(proc, "tokenizer", None) or proc
    return tok, proc


def _load_lm(model_name, kwargs):
    """Load a model whose forward returns text-vocab .logits. Tries the plain
    CausalLM mapping first, then the multimodal auto-classes (Gemma 3/3n use
    ImageTextToText; Gemma 4 uses MultimodalLM) — a text-only forward
    (input_ids, no pixel/audio inputs) still yields the logits our constrained
    scorer needs. The extra classes may not exist on older transformers, so each
    is resolved defensively."""
    import transformers
    candidates = ["AutoModelForCausalLM", "AutoModelForImageTextToText",
                  "AutoModelForMultimodalLM"]
    classes = [getattr(transformers, n) for n in candidates
               if getattr(transformers, n, None) is not None]
    last = None
    for cls in classes:
        try:
            return cls.from_pretrained(model_name, **kwargs)
        except (ValueError, KeyError, OSError) as e:
            last = e  # wrong auto-mapping for this arch; try the next class
    raise last


def _plain_prompt(messages):
    """Format messages as one plain text prompt for base LMs that have NO chat
    template (e.g. OLMoE-1B-7B-0924, jetmoe-8b). Just concatenates the turns;
    the trailing newline lets the scored "Answer: X" continuation start cleanly."""
    return "\n\n".join(m["content"] for m in messages) + "\n"


def _merge_system_into_user(messages):
    """Fold a leading system turn into the first user turn, for chat templates
    (e.g. Gemma) that don't accept a 'system' role. Returns a new list; the
    original is unchanged. No-op if there's no leading system message."""
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_txt = messages[0]["content"]
    rest = messages[1:]
    for i, m in enumerate(rest):
        if m["role"] == "user":
            merged = dict(m)
            merged["content"] = f"{sys_txt}\n\n{m['content']}"
            return rest[:i] + [merged] + rest[i + 1:]
    return rest  # no user turn (shouldn't happen for our prompts)


class Scorer:
    def __init__(self, model_name, adapter=None, load_4bit=False, batch_size=8,
                 trust_remote_code=False, merge_system=False):
        self.merge_system = merge_system
        self.tok, self.processor = _load_tokenizer(model_name, trust_remote_code)
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto",
                  "trust_remote_code": trust_remote_code}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
            kwargs.pop("torch_dtype")
        self.model = _load_lm(model_name, kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.batch_size = batch_size

    @staticmethod
    def _templater_call(obj, messages, enable_thinking, **kwargs):
        try:
            return obj.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            return obj.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _normalize_ids(ids):
        # transformers ≥4.51 may return BatchEncoding even without return_tensors
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def _prompt_ids(self, messages, enable_thinking=False):
        """Build prompt token ids. Tries the tokenizer's chat template, then the
        processor's (multimodal repos like Gemma 4 keep the working template
        there), retrying with the system turn folded in if a template rejects
        the 'system' role. Falls back to a plain concatenation only for true
        base LMs (OLMoE, jetmoe) that have no chat template anywhere."""
        if self.merge_system:
            # models whose authors discourage a system turn (e.g. DeepSeek-R1
            # distills: "all instructions should be contained within the user
            # prompt") — fold it in up front rather than only on template error
            messages = _merge_system_into_user(messages)
        kwargs = {"add_generation_prompt": True, "tokenize": True,
                  "return_tensors": None}
        for obj in (self.tok, self.processor):
            if obj is None or not hasattr(obj, "apply_chat_template"):
                continue
            for msgs in (messages, _merge_system_into_user(messages)):
                try:
                    ids = self._templater_call(obj, msgs, enable_thinking, **kwargs)
                    return self._normalize_ids(ids)
                except Exception as e:
                    if "system" in str(e).lower():
                        continue      # retry this obj with the system turn merged
                    break             # no usable template on this obj; try the next
        # no chat template anywhere -> base LM: plain-text prompt
        return list(self.tok.encode(_plain_prompt(messages),
                                    add_special_tokens=True))

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

    @torch.no_grad()
    def generate_text(self, messages, max_new_tokens=1024, enable_thinking=True,
                      temperature=0.0):
        """Free-generation with reasoning ON, for thought-channel models (e.g.
        Gemma 4) whose answer can't be read off next-token logits. Returns the
        decoded completion (special tokens kept, so channel markers survive for
        the caller's answer parser)."""
        prompt_ids = self._prompt_ids(messages, enable_thinking=enable_thinking)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long,
                                 device=self.model.device)
        attn = torch.ones_like(input_ids)
        eos = self.tok.eos_token_id
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos
        sampling = ({"do_sample": True, "temperature": temperature, "top_p": 0.95}
                    if temperature > 0 else {"do_sample": False})
        gen = self.model.generate(
            input_ids=input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, pad_token_id=pad, **sampling)
        new_ids = gen[0][len(prompt_ids):]
        return self.tok.decode(new_ids, skip_special_tokens=False)


_ANSWER_RE = re.compile(r"[Aa]nswer\s*[:：]\s*\**\s*([ABCD]|Yes|No|Both|0)\b")


def parse_generated_answer(text, allowed):
    """Pull the final answer from a generation. Prefers the LAST 'Answer: X'
    (a reasoning model states its conclusion last); falls back to the last bare
    allowed token. Returns None if nothing matches `allowed`."""
    hits = [m.group(1) for m in _ANSWER_RE.finditer(text)
            if m.group(1) in allowed]
    if hits:
        return hits[-1]
    # fallback: last standalone allowed token anywhere in the text
    esc = "|".join(re.escape(a) for a in allowed)
    bare = re.findall(rf"(?<![A-Za-z0-9])({esc})(?![A-Za-z0-9])", text)
    return bare[-1] if bare else None


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def score_mcq(scorer, rec, n_perms=1, prior=None, prior_tau=0.0, value_summaries="auto"):
    """Return dict original_letter -> prob, permutation-averaged and calibrated."""
    acc = {ltr: 0.0 for ltr in LETTERS4}
    shifts = CYCLIC_SHIFTS[:n_perms]
    for shift in shifts:
        p, old_to_new = permute_record(rec, shift)
        messages = build_messages(p, mode="direct", value_summaries=value_summaries)
        lps = scorer.score_candidates(messages, [f"Answer: {l}" for l in LETTERS4])
        probs = dict(zip(LETTERS4, softmax(lps)))
        for old, new in old_to_new.items():
            acc[old] += probs[new] / len(shifts)
    if prior and prior_tau > 0:
        acc = {k: v * (prior.get(k, 1e-9) ** prior_tau) for k, v in acc.items()}
        z = sum(acc.values())
        acc = {k: v / z for k, v in acc.items()}
    return acc


def score_si(scorer, rec, threshold=0.5, value_summaries="auto"):
    """Binary-decomposed Sri Lankan scoring -> (label, p_yes_A, p_yes_B)."""
    p_yes = {}
    for stmt in ("A", "B"):
        messages = build_messages(rec, mode="direct", si_statement=stmt,
                                  value_summaries=value_summaries)
        lps = scorer.score_candidates(messages, ["Answer: Yes", "Answer: No"])
        p_yes[stmt] = softmax(lps)[0]
    a, b = p_yes["A"] >= threshold, p_yes["B"] >= threshold
    label = "Both" if (a and b) else "A" if a else "B" if b else "0"
    return label, p_yes["A"], p_yes["B"]


def score_mcq_gen(scorer, rec, n_perms=1, value_summaries="auto",
                  max_new_tokens=1024, temperature=0.0):
    """Generation-based MCQ scoring for thought-channel models (Gemma 4): let
    the model reason, parse the final 'Answer: X', majority-vote across cyclic
    option permutations. Returns a prob dict (vote share per original letter)."""
    votes = collections.Counter()
    for shift in CYCLIC_SHIFTS[:n_perms]:
        p, old_to_new = permute_record(rec, shift)
        new_to_old = {new: old for old, new in old_to_new.items()}
        messages = build_messages(p, mode="cot", value_summaries=value_summaries)
        text = scorer.generate_text(messages, max_new_tokens=max_new_tokens,
                                    temperature=temperature)
        shown = parse_generated_answer(text, LETTERS4)  # letter in displayed order
        if shown is not None:
            votes[new_to_old.get(shown, shown)] += 1
    total = sum(votes.values())
    if not total:
        return {l: 0.0 for l in LETTERS4}  # unparseable -> all zero (predicts A)
    return {l: votes.get(l, 0) / total for l in LETTERS4}


def score_si_gen(scorer, rec, threshold=0.5, value_summaries="auto",
                 max_new_tokens=1024, temperature=0.0):
    """Generation-based Sri Lankan scoring for thought-channel models: reason
    per statement, parse the final 'Answer: Yes/No'. Same (label, pa, pb) shape
    as score_si, with pa/pb in {0.0, 1.0}."""
    p_yes = {}
    for stmt in ("A", "B"):
        messages = build_messages(rec, mode="cot", si_statement=stmt,
                                  value_summaries=value_summaries)
        text = scorer.generate_text(messages, max_new_tokens=max_new_tokens,
                                    temperature=temperature)
        ans = parse_generated_answer(text, ["Yes", "No"])
        p_yes[stmt] = 1.0 if ans == "Yes" else 0.0
    a, b = p_yes["A"] >= threshold, p_yes["B"] >= threshold
    label = "Both" if (a and b) else "A" if a else "B" if b else "0"
    return label, p_yes["A"], p_yes["B"]


def is_correct(rec, pred):
    if rec["dataset"] == "indonesian":
        return pred in set(rec["consensus"])
    return pred == rec["gold"]


def compute_prior(records, exclude_folds=None):
    """Label prior over the TRAINING records (everything not in exclude_folds)."""
    excl = set(exclude_folds) if exclude_folds else set()
    recs = [r for r in records if r["fold"] not in excl]
    c = collections.Counter(r["gold"] for r in recs)
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def run_eval(args, scorer):
    results = {}
    eval_folds = args.eval_folds  # set or None (None = whole processed file)
    details_path = Path(args.out or (ROOT / "processed" / "eval_details.jsonl"))
    details = open(details_path, "w", encoding="utf-8")
    for ds in args.datasets:
        recs = load_jsonl(Path(args.processed_dir) / f"{ds}.jsonl")
        # prior comes from the training folds (exclude the held-out eval folds)
        prior = compute_prior(recs, exclude_folds=eval_folds)
        eval_recs = [r for r in recs if eval_folds is None or r["fold"] in eval_folds]
        n_ok = 0
        for i, rec in enumerate(eval_recs):
            if ds == "sri_lankan":
                if args.generate:
                    pred, pa, pb = score_si_gen(
                        scorer, rec, threshold=args.si_threshold,
                        value_summaries=args.value_summaries,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.gen_temperature)
                else:
                    pred, pa, pb = score_si(scorer, rec, threshold=args.si_threshold,
                                            value_summaries=args.value_summaries)
                extra = {"p_yes_A": pa, "p_yes_B": pb}
            else:
                if args.generate:
                    probs = score_mcq_gen(scorer, rec, n_perms=args.n_perms,
                                          value_summaries=args.value_summaries,
                                          max_new_tokens=args.max_new_tokens,
                                          temperature=args.gen_temperature)
                else:
                    probs = score_mcq(scorer, rec, n_perms=args.n_perms,
                                      prior=prior, prior_tau=args.prior_tau,
                                      value_summaries=args.value_summaries)
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
                                  prior=dev_priors.get(ds), prior_tau=args.prior_tau,
                                  value_summaries=args.value_summaries)
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
                    help="eval only this single CV fold")
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="eval on the union of these folds (e.g. --folds 3 4 for "
                         "the 60/40 holdout experiment). Overrides --fold.")
    ap.add_argument("--n_perms", type=int, default=1,
                    help="cyclic option permutations to ensemble (ZH/ID)")
    ap.add_argument("--prior_tau", type=float, default=0.0,
                    help="0 = no prior calibration; try 0.25-1.0 on CV folds")
    ap.add_argument("--si_threshold", type=float, default=0.5)
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="allow custom modeling code from the Hub repo (needed "
                         "for e.g. jetmoe-8b); only enable for repos you trust")
    ap.add_argument("--generate", action="store_true",
                    help="reason-then-parse instead of constrained scoring: let "
                         "the model generate (thinking ON) and parse the final "
                         "'Answer: X'. Needed for thought-channel models like "
                         "Gemma 4 whose answer isn't readable from next-token logits")
    ap.add_argument("--max_new_tokens", type=int, default=1024,
                    help="generation budget per item when --generate is set "
                         "(must fit the reasoning trace + final answer)")
    ap.add_argument("--gen_temperature", type=float, default=0.0,
                    help="sampling temperature for --generate (0 = greedy). "
                         "Reasoning models like DeepSeek-R1 distills recommend "
                         "~0.6; greedy can send them into endless repetition")
    ap.add_argument("--no_system", action="store_true",
                    help="fold the system turn into the user turn — required by "
                         "models whose authors discourage a system prompt "
                         "(e.g. DeepSeek-R1 distills)")
    ap.add_argument("--test_files", nargs="+", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--value_summaries", default=None,
                    help="path to a custom summaries json (default: "
                         "id_value_summaries.json at the repo root is used "
                         "AUTOMATICALLY — this flag only overrides which file")
    ap.add_argument("--no_value_summaries", action="store_true",
                    help="disable Indonesian value-context injection entirely "
                         "(use when evaluating an adapter trained BEFORE this "
                         "default was introduced, to avoid a prompt mismatch)")
    args = ap.parse_args()
    # resolve --folds / --fold into one set of eval folds (None = whole file)
    if args.folds is not None:
        args.eval_folds = set(args.folds)
    elif args.fold is not None:
        args.eval_folds = {args.fold}
    else:
        args.eval_folds = None
    if args.eval_folds is not None:
        print(f"evaluating on fold(s): {sorted(args.eval_folds)}")
    # submission "dataset" field names — adjust if organizers specify others
    args.dataset_names = {"chinese": "chinese", "indonesian": "indonesian",
                          "sri_lankan": "sri_lankan"}
    if args.no_value_summaries:
        args.value_summaries = None
        print("Indonesian value-context injection: DISABLED (--no_value_summaries)")
    elif args.value_summaries:
        from prompts import load_id_value_summaries
        custom_path = args.value_summaries
        args.value_summaries = load_id_value_summaries(custom_path)
        print(f"loaded {len(args.value_summaries)} value summaries from "
              f"{custom_path!r} for Indonesian prompt injection")
    else:
        args.value_summaries = "auto"  # -> id_value_summaries.json at repo root
        print("Indonesian value-context injection: AUTO "
              "(id_value_summaries.json at repo root)")

    scorer = Scorer(args.model, adapter=args.adapter, load_4bit=args.load_4bit,
                    trust_remote_code=args.trust_remote_code,
                    merge_system=args.no_system)
    if args.mode == "eval":
        run_eval(args, scorer)
    else:
        assert args.test_files and args.out, "predict needs --test_files and --out"
        run_predict(args, scorer)


if __name__ == "__main__":
    main()
