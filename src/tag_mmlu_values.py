"""Assign each SinhalaMMLU item the most relevant shared-task value category.

The shared-task SI prompts are value-conditioned ("Value being tested: X"), but
MMLU items carry no value tag. This script picks, for every MMLU item, the best
match from the value vocabulary actually present in the SI dev set (46 values),
so the auxiliary training prompts are shape-identical to the real ones.

Method: constrained scoring, not free generation. For each item we score the
sequence logprob of every candidate value name as a continuation of a tagging
prompt, length-normalized (mean logprob per token) so long value names aren't
penalized. argmax over the 46 candidates -> always a legal, in-vocabulary tag,
no parsing failures.

Contamination guard: --drop_q_nos removes items flagged by check_mmlu_overlap.py
(question text that duplicates an SI dev item). Defaults to the 13 civics items
identified there: 10 exact matches + 3 genuine near-duplicates.

  python src/tag_mmlu_values.py --model meta-llama/Llama-3.1-8B-Instruct --load_4bit
  python src/tag_mmlu_values.py --subjects Civics "Health and Physical Science"

Output: processed/mmlu_value_tags.json
  {"<subject>|<q_no>": {"value": "Discipline", "score": -1.23,
                        "top3": [["Discipline", -1.23], ...]}}
"""
import argparse
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- item selection, kept auditable so the reason for each exclusion is clear -
# 1. contamination: question text duplicates an SI dev item (check_mmlu_overlap.py).
#    Civics: 10 exact + q29/q51/q63 near-duplicates. Health: all 8 exact matches.
CONTAMINATED_DROP = {
    "Civics": [9, 15, 16, 28, 29, 30, 31, 35, 41, 51, 55, 62, 63],
    "Health and Physical Science": [25, 48, 49, 83, 97, 104, 107, 116],
}

# 2. manual review: items judged unsuitable as value-judgment training data.
MANUAL_DROP = {"Civics": [
    1, 4, 6, 7, 10, 12, 17, 18, 19, 24, 37, 45, 46, 47, 48, 52, 53, 59, 61, 70,
    71, 76, 77, 78, 82, 83, 84, 88, 90, 92, 97, 98, 100, 101, 105, 107, 108,
    109, 110, 112, 113, 115, 119, 120]}

# 3. allow-list: for these subjects use ONLY these q_nos (everything else in the
#    subject is ignored). Contamination drops still apply on top.
#    Health: 20 requested; q83/97/104/107/116 are exact dev duplicates and are
#    removed by CONTAMINATED_DROP, leaving 15.
KEEP_ONLY = {"Health and Physical Science": [
    4, 7, 32, 35, 36, 41, 51, 53, 74, 83, 93, 94, 95, 97, 100, 104, 106, 107,
    115, 116]}

# union of exclusions actually applied
DEFAULT_DROP = {
    subj: sorted(set(CONTAMINATED_DROP.get(subj, [])) | set(MANUAL_DROP.get(subj, [])))
    for subj in set(CONTAMINATED_DROP) | set(MANUAL_DROP)
}


def select_items(mmlu, subjects, apply_drop=True):
    """MMLU rows for the requested subjects, after allow-list + drop-list.

    Retained today: Civics 66/123, Health and Physical Science 15/130.
    """
    wanted = {s.strip().lower() for s in subjects}
    out = []
    for m in mmlu:
        subj = m.get("subject", "").strip()
        if subj.lower() not in wanted:
            continue
        qno = m.get("q_no")
        if subj in KEEP_ONLY and qno not in set(KEEP_ONLY[subj]):
            continue
        if apply_drop and qno in set(DEFAULT_DROP.get(subj, [])):
            continue
        out.append(m)
    return out

TAG_SYSTEM = (
    "You are an expert on Sri Lankan societal values. You will read a school "
    "question in Sinhala together with its answer options and the correct "
    "answer. Your job is to identify which single societal value the question "
    "is fundamentally testing."
)

TAG_USER = """Question (Sinhala):
{question}

Options (Sinhala):
{options}

Correct answer: {answer}

Which single Sri Lankan societal value does this question test? Reply with exactly one value name from the allowed list.

Value:"""


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def value_vocabulary(si_dev_path):
    """The value names actually used by the shared-task SI data."""
    dev = load_jsonl(si_dev_path)
    return sorted({d["value_english"].strip() for d in dev if d.get("value_english")})


def build_tag_messages(item):
    opts = "\n".join(f"{i+1}. {c}" for i, c in enumerate(item.get("choices", [])))
    ans_idx = item.get("answer")
    choices = item.get("choices", [])
    answer_text = (choices[ans_idx - 1]
                   if isinstance(ans_idx, int) and 1 <= ans_idx <= len(choices)
                   else str(ans_idx))
    user = TAG_USER.format(question=item.get("question", ""), options=opts,
                           answer=answer_text)
    return [{"role": "system", "content": TAG_SYSTEM},
            {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--mmlu", default=str(ROOT / "sinhala_mmlu.json"))
    ap.add_argument("--si_dev", default=str(ROOT / "processed" / "sri_lankan.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "processed" / "mmlu_value_tags.json"))
    ap.add_argument("--subjects", nargs="+", default=["Civics"],
                    help="MMLU subjects to tag (default: Civics only)")
    ap.add_argument("--drop_q_nos", nargs="+", type=int, default=None,
                    help="override the built-in contamination drop-list "
                         "(only applies to the first --subjects entry)")
    ap.add_argument("--no_drop", action="store_true",
                    help="tag everything, including known duplicates (not "
                         "recommended — breaks CV integrity if used in training)")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap items")
    args = ap.parse_args()

    values = value_vocabulary(args.si_dev)
    print(f"value vocabulary: {len(values)} values from the SI dev set")

    mmlu = load_jsonl(args.mmlu)
    if args.drop_q_nos is not None:  # manual override for the first subject
        DEFAULT_DROP[args.subjects[0]] = args.drop_q_nos
    subset = select_items(mmlu, args.subjects, apply_drop=not args.no_drop)
    by_subj = collections.Counter(m.get("subject", "").strip() for m in subset)
    print(f"{len(subset)} items selected from {args.subjects}: {dict(by_subj)}")
    if args.no_drop:
        print("  WARNING --no_drop: contaminated items included (breaks CV integrity)")
    if args.limit:
        subset = subset[: args.limit]

    # reuse evaluate.py's Scorer: handles chat templates, 4bit, adapters
    from evaluate import Scorer
    scorer = Scorer(args.model, load_4bit=args.load_4bit)

    tags = {}
    dist = collections.Counter()
    for i, item in enumerate(subset):
        messages = build_tag_messages(item)
        # length-normalized scoring so long value names aren't penalized
        scored = []
        for v in values:
            lp = scorer.score_candidates(messages, [f" {v}"])[0]
            n_tok = max(1, len(scorer.tok.encode(f" {v}", add_special_tokens=False)))
            scored.append((v, lp / n_tok))
        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_score = scored[0]
        key = f"{item.get('subject', '').strip()}|{item.get('q_no')}"
        tags[key] = {"value": best, "score": round(best_score, 4),
                     "top3": [[v, round(s, 4)] for v, s in scored[:3]],
                     "q_no": item.get("q_no"),
                     "subject": item.get("subject", "").strip()}
        dist[best] += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(subset)} tagged")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)

    print(f"\ntagged {len(tags)} items -> {args.out}")
    print(f"distinct values used: {len(dist)} / {len(values)}")
    print("top value assignments:")
    for v, c in dist.most_common(15):
        print(f"   {v:24s} {c}")
    unused = [v for v in values if v not in dist]
    if unused:
        print(f"unused values ({len(unused)}): {', '.join(unused[:12])}"
              f"{' ...' if len(unused) > 12 else ''}")


if __name__ == "__main__":
    main()
