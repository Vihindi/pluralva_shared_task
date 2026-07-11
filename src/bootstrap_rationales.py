"""STaR-style rationale bootstrapping (methodology §2.4; Zelikman et al. 2022).

For every dev item, sample k chain-of-thought completions from the (untuned)
base model using the value-conditioned CoT prompt. Keep the first chain whose
final "Answer: X" matches the gold label (Indonesian: lands in the consensus
set; Sri Lankan: per-statement binary judgment). If no sampled chain is correct,
fall back to *rationalization*: reveal the gold answer and ask the model to
justify it, marking source="rationalized".

Output: rationales.jsonl with {"key", "uid", "dataset", "rationale", "answer",
"source"}. key = uid for ZH/ID, uid_A / uid_B for the Sri Lankan binary items.
Feed this file to build_sft_data.py --rationales.

Requires GPU (bf16 ~16GB for an 8B model; use --load_4bit on 16GB cards).
  pip install torch transformers accelerate bitsandbytes
  python src/bootstrap_rationales.py --model Qwen/Qwen3-8B --k 8
"""
import argparse
import json
import re
import zlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import build_messages
from build_sft_data import permute_record

ROOT = Path(__file__).resolve().parent.parent
ANSWER_RE = re.compile(r"Answer:\s*([ABCD]|Yes|No)", re.IGNORECASE)


def uid_shift(uid):
    """Deterministic per-item cyclic option shift (0-3), stable across runs
    and processes (crc32, NOT Python's randomized hash()). Spreads the gold
    answer uniformly over positions A-D during rationale generation, so the
    base model's position bias can't skew which items get STaR rationales."""
    return zlib.crc32(uid.encode("utf-8")) % 4


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_answer(text):
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    a = matches[-1].capitalize()
    return a.upper() if a in ("A", "B", "C", "D") else a


def strip_final_answer(text):
    """Remove the trailing Answer: line so build_sft_data can re-append it."""
    m = list(ANSWER_RE.finditer(text))
    if m:
        text = text[: m[-1].start()]
    return text.strip()


class Generator:
    def __init__(self, model_name, load_4bit=False, max_new_tokens=400):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
            kwargs.pop("torch_dtype")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def chat(self, messages, n=1, temperature=0.8):
        kwargs = {"add_generation_prompt": True, "return_tensors": "pt"}
        try:  # Qwen3: no thinking blocks in bootstrapped rationales
            encoded = self.tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            encoded = self.tok.apply_chat_template(messages, **kwargs)
        # transformers ≥4.51 returns BatchEncoding, not a raw tensor
        input_ids = encoded["input_ids"] if hasattr(encoded, "input_ids") else encoded
        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids, do_sample=temperature > 0, temperature=temperature,
                top_p=0.95, num_return_sequences=n,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tok.eos_token_id)
        prompt_len = input_ids.shape[1]
        return [self.tok.decode(seq[prompt_len:], skip_special_tokens=True)
                for seq in out]


def rationalize_messages(messages, gold_text):
    """Reveal the gold answer and ask the model to justify it."""
    msgs = [dict(m) for m in messages]
    msgs[-1]["content"] += (
        f"\n\nThe correct answer is known to be {gold_text}. Explain briefly and "
        f"convincingly why this is the answer most consistent with the value and "
        f"local social judgment, then end with the required \"Answer: {gold_text}\" line.")
    return msgs


def targets_for(rec):
    """Yield (key, si_statement, is_correct(ans)->bool, gold_text)."""
    ds = rec["dataset"]
    if ds == "chinese":
        yield rec["uid"], None, (lambda a, g=rec["gold"]: a == g), rec["gold"]
    elif ds == "indonesian":
        cons = set(rec["consensus"])
        yield rec["uid"], None, (lambda a, c=cons: a in c), rec["consensus"][0]
    else:  # sri_lankan binary decomposition
        for stmt, ok in (("A", rec["stmt_A_ok"]), ("B", rec["stmt_B_ok"])):
            gold = "Yes" if ok else "No"
            yield f"{rec['uid']}_{stmt}", stmt, (lambda a, g=gold: a == g), gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--out", default=str(ROOT / "processed" / "rationales.jsonl"))
    ap.add_argument("--datasets", nargs="+",
                    default=["chinese", "indonesian", "sri_lankan"])
    ap.add_argument("--k", type=int, default=8, help="CoT samples per target")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--shuffle_options", action="store_true",
                    help="deterministically shuffle ZH/ID options per item "
                         "(gold/votes remapped) so gold positions are balanced "
                         "during rationale generation; the applied shift is "
                         "recorded so build_sft_data attaches each rationale "
                         "to the matching option order")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap items per dataset")
    args = ap.parse_args()

    gen = Generator(args.model, load_4bit=args.load_4bit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():  # resumable
        done = {r["key"] for r in load_jsonl(out_path)}
        print(f"resuming: {len(done)} keys already done")

    n_star, n_rationalized = 0, 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for ds in args.datasets:
            recs = load_jsonl(Path(args.processed_dir) / f"{ds}.jsonl")
            if args.limit:
                recs = recs[: args.limit]
            for i, rec in enumerate(recs):
                # sri_lankan is untouched: its binary decomposition shows one
                # statement at a time, so there is no option order to shuffle
                shift = 0
                if args.shuffle_options and ds != "sri_lankan":
                    shift = uid_shift(rec["uid"])
                    rec, _ = permute_record(rec, shift)
                for key, stmt, is_correct, gold_text in targets_for(rec):
                    if key in done:
                        continue
                    messages = build_messages(rec, mode="cot", si_statement=stmt)
                    rationale, answer, source = None, None, None
                    for text in gen.chat(messages, n=args.k, temperature=args.temperature):
                        a = parse_answer(text)
                        if a is not None and is_correct(a):
                            rationale, answer, source = strip_final_answer(text), a, "star"
                            break
                    if rationale is None:  # rationalization fallback
                        text = gen.chat(rationalize_messages(messages, gold_text),
                                        n=1, temperature=0.3)[0]
                        rationale, answer, source = (strip_final_answer(text),
                                                     gold_text, "rationalized")
                    n_star += source == "star"
                    n_rationalized += source == "rationalized"
                    fout.write(json.dumps({
                        "key": key, "uid": rec["uid"], "dataset": ds,
                        "rationale": rationale, "answer": answer, "source": source,
                        "shift": shift,
                    }, ensure_ascii=False) + "\n")
                    fout.flush()
                if (i + 1) % 20 == 0:
                    print(f"[{ds}] {i+1}/{len(recs)}  star={n_star} rationalized={n_rationalized}")
    print(f"done: star={n_star} rationalized={n_rationalized} -> {out_path}")


if __name__ == "__main__":
    main()
