"""Build Stage-1 SFT training files (chat-format JSONL) from processed data.

Per methodology §2.4:
  * Option-permutation augmentation (Chinese, Indonesian): each item is emitted
    under the 4 cyclic option orderings with the gold letter / votes remapped.
    This teaches order-invariance (mitigates MCQ selection bias) and multiplies
    the tiny training set 4x.
  * Indonesian soft labels via vote-expansion: one training example per annotator
    vote (5 per item). Under per-example cross-entropy this is exactly equivalent
    in expectation to training the letter distribution against the empirical
    5-vote distribution with soft cross-entropy — no custom loss needed.
  * Sri Lankan binary decomposition: each item becomes two independent Yes/No
    judgments (statement A, statement B), removing the A-position bias and the
    {A:52%} class skew from the learning problem.
  * Optional rationales (from bootstrap_rationales.py): when a rationale exists
    for an example, the assistant target becomes "<rationale>\nAnswer: X"
    (STaR-style); otherwise the target is the bare "Answer: X" line.

Outputs (out_dir, default sft_data/):
  {zh|id|si}_train_fold{k}.jsonl  — training split excluding CV fold k (k=0..4)
  {zh|id|si}_train_full.jsonl     — all dev data (for the final submission model)
Each line: {"messages":[{system},{user},{assistant}], "meta":{...}}
Joint multi-country training = pass several files to train_lora.py.
"""
import argparse
import json
import random
from pathlib import Path

from prompts import build_messages

SEED = 42
N_FOLDS = 5
LETTERS4 = ["A", "B", "C", "D"]
ROOT = Path(__file__).resolve().parent.parent

# cyclic shifts: position i shows the option that was at (i+s) mod 4
CYCLIC_SHIFTS = [0, 1, 2, 3]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def permute_record(rec, shift):
    """Return a copy of rec with options cyclically shifted and labels remapped.

    New position i (letter LETTERS4[i]) holds the option originally at letter
    LETTERS4[(i+shift) % 4]. old_letter -> new_letter map inverts that.
    """
    if shift == 0:
        return dict(rec), {ltr: ltr for ltr in LETTERS4}
    new = dict(rec)
    new_options = {}
    old_to_new = {}
    for i, new_ltr in enumerate(LETTERS4):
        old_ltr = LETTERS4[(i + shift) % 4]
        new_options[new_ltr] = rec["options"][old_ltr]
        old_to_new[old_ltr] = new_ltr
    new["options"] = new_options
    if "gold" in rec and rec["gold"] in old_to_new:
        new["gold"] = old_to_new[rec["gold"]]
    if "votes" in rec:
        new["votes"] = [old_to_new[v] for v in rec["votes"]]
        new["consensus"] = sorted(old_to_new[c] for c in rec["consensus"])
    return new, old_to_new


def make_example(rec, target_letter, rationales, rationale_key, si_statement=None):
    messages = build_messages(rec, mode="direct", si_statement=si_statement)
    rat = rationales.get(rationale_key)
    if rat:
        assistant = f"{rat['text']}\nAnswer: {target_letter}"
        msgs = build_messages(rec, mode="cot", si_statement=si_statement)
        messages = msgs
    else:
        assistant = f"Answer: {target_letter}"
    return {
        "messages": messages + [{"role": "assistant", "content": assistant}],
        "meta": {"uid": rec["uid"], "dataset": rec["dataset"],
                 "fold": rec["fold"], "with_rationale": bool(rat)},
    }


def rationale_key_for(rationales, uid, shift):
    """A rationale's letter references only make sense under the option order
    it was generated with — attach it to the permutation copy whose shift
    matches the one recorded by bootstrap_rationales.py (0 for old files)."""
    rat = rationales.get(uid)
    if rat and rat["shift"] == shift:
        return uid
    return None


def shifts_for(rationales, uid, n_perms):
    """Which cyclic option orderings to emit for this item.

    n_perms > 1 : the first n_perms cyclic shifts (permutation augmentation).
    n_perms == 1: a single copy — but at the shift the item's rationale was
                  generated under (so it still attaches and stays balanced);
                  falls back to 0 when no rationale / old files."""
    if n_perms > 1:
        return CYCLIC_SHIFTS[:n_perms]
    rat = rationales.get(uid)
    return [rat["shift"] if rat else 0]


def build_chinese(recs, rationales, n_perms):
    out = []
    for rec in recs:
        for shift in shifts_for(rationales, rec["uid"], n_perms):
            p, _ = permute_record(rec, shift)
            key = rationale_key_for(rationales, rec["uid"], shift)
            out.append(make_example(p, p["gold"], rationales, key))
    return out


def build_indonesian(recs, rationales, n_perms):
    out = []
    for rec in recs:
        for shift in shifts_for(rationales, rec["uid"], n_perms):
            p, _ = permute_record(rec, shift)
            key = rationale_key_for(rationales, rec["uid"], shift)
            # vote-expansion: one example per annotator vote == soft-label CE
            for vote in p["votes"]:
                # a rationale argues for one answer; never pair it with a
                # different vote's target letter
                vote_key = key if (key and rationales[key]["answer"] == vote) else None
                out.append(make_example(p, vote, rationales, vote_key))
    return out


def build_sri_lankan(recs, rationales):
    out = []
    for rec in recs:
        for stmt, ok in (("A", rec["stmt_A_ok"]), ("B", rec["stmt_B_ok"])):
            target = "Yes" if ok else "No"
            key = f"{rec['uid']}_{stmt}"
            out.append(make_example(rec, target, rationales, key, si_statement=stmt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--out_dir", default=str(ROOT / "sft_data"))
    ap.add_argument("--rationales", default=None,
                    help="optional rationales.jsonl from bootstrap_rationales.py")
    ap.add_argument("--n_perms", type=int, default=4,
                    help="cyclic option permutations for ZH/ID (1 = no augmentation)")
    args = ap.parse_args()
    processed, out_dir = Path(args.processed_dir), Path(args.out_dir)

    rationales = {}
    if args.rationales:
        for r in load_jsonl(Path(args.rationales)):
            rationales[r["key"]] = {"text": r["rationale"],
                                    "answer": r.get("answer"),
                                    "shift": r.get("shift", 0)}
        print(f"loaded {len(rationales)} rationales")

    builders = {
        "zh": ("chinese.jsonl", lambda recs: build_chinese(recs, rationales, args.n_perms)),
        "id": ("indonesian.jsonl", lambda recs: build_indonesian(recs, rationales, args.n_perms)),
        "si": ("sri_lankan.jsonl", lambda recs: build_sri_lankan(recs, rationales)),
    }
    rng = random.Random(SEED)
    summary = {}
    for tag, (fname, build) in builders.items():
        recs = load_jsonl(processed / fname)
        examples = build(recs)
        rng.shuffle(examples)
        save_jsonl(examples, out_dir / f"{tag}_train_full.jsonl")
        for k in range(N_FOLDS):
            fold_ex = [e for e in examples if e["meta"]["fold"] != k]
            save_jsonl(fold_ex, out_dir / f"{tag}_train_fold{k}.jsonl")
        summary[tag] = {
            "source_items": len(recs), "train_full": len(examples),
            "with_rationale": sum(e["meta"]["with_rationale"] for e in examples),
            "per_fold_train": {k: sum(1 for e in examples if e["meta"]["fold"] != k)
                               for k in range(N_FOLDS)},
        }
        print(f"[{tag}] {len(recs)} items -> {len(examples)} SFT examples "
              f"({summary[tag]['with_rationale']} with rationales)")

    with open(out_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
