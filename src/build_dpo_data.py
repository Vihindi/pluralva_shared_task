"""Build Stage-2 DPO preference pairs from the processed dev data (§2.5).

Pair construction, per dataset:
  Chinese    : the item's gold option vs each of the 3 distractors
               (1 chosen, 3 rejected -> 3 pairs), under --n_perms cyclic
               option orderings with letters remapped.
  Indonesian : graded pairs from the 5-vote distribution — for every ordered
               label pair (i, j) with votes_i > votes_j emit (i chosen, j
               rejected). This encodes *how much* annotators preferred an
               option instead of a binarized consensus, and emits nothing for
               tied labels (both consensus answers are correct, so neither may
               be pushed below the other).
  Sri Lankan : per binary judgment, correct Yes/No vs the opposite (2 pairs
               per item). Mostly useful to sharpen the decision boundary the
               threshold calibration later operates on.

Output (dpo_data/): {zh,id,si}_dpo_fold{k}.jsonl (train pairs excluding CV
fold k) and {zh,id,si}_dpo_full.jsonl — trl DPOTrainer conversational format:
  {"prompt": [messages...], "chosen": [assistant msg], "rejected": [assistant msg]}
"""
import argparse
import json
import random
from pathlib import Path

from prompts import build_messages
from build_sft_data import permute_record, CYCLIC_SHIFTS, load_jsonl, save_jsonl

SEED = 42
N_FOLDS = 5
LETTERS4 = ["A", "B", "C", "D"]
ROOT = Path(__file__).resolve().parent.parent


def pair(rec, chosen_text, rejected_text, si_statement=None):
    return {
        "prompt": build_messages(rec, mode="direct", si_statement=si_statement),
        "chosen": [{"role": "assistant", "content": f"Answer: {chosen_text}"}],
        "rejected": [{"role": "assistant", "content": f"Answer: {rejected_text}"}],
        "meta": {"uid": rec["uid"], "dataset": rec["dataset"], "fold": rec["fold"]},
    }


def build_chinese(recs, n_perms):
    out = []
    for rec in recs:
        for shift in CYCLIC_SHIFTS[:n_perms]:
            p, _ = permute_record(rec, shift)
            for distractor in LETTERS4:
                if distractor != p["gold"]:
                    out.append(pair(p, p["gold"], distractor))
    return out


def build_indonesian(recs, n_perms):
    out = []
    for rec in recs:
        for shift in CYCLIC_SHIFTS[:n_perms]:
            p, _ = permute_record(rec, shift)
            counts = {ltr: p["votes"].count(ltr) for ltr in LETTERS4}
            for i in LETTERS4:
                for j in LETTERS4:
                    if counts[i] > counts[j]:
                        out.append(pair(p, i, j))
    return out


def build_sri_lankan(recs):
    out = []
    for rec in recs:
        for stmt, ok in (("A", rec["stmt_A_ok"]), ("B", rec["stmt_B_ok"])):
            chosen, rejected = ("Yes", "No") if ok else ("No", "Yes")
            out.append(pair(rec, chosen, rejected, si_statement=stmt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--out_dir", default=str(ROOT / "dpo_data"))
    ap.add_argument("--n_perms", type=int, default=2,
                    help="cyclic option orderings for ZH/ID (pairs scale linearly)")
    args = ap.parse_args()
    processed, out_dir = Path(args.processed_dir), Path(args.out_dir)

    builders = {
        "zh": ("chinese.jsonl", lambda r: build_chinese(r, args.n_perms)),
        "id": ("indonesian.jsonl", lambda r: build_indonesian(r, args.n_perms)),
        "si": ("sri_lankan.jsonl", build_sri_lankan),
    }
    rng = random.Random(SEED)
    for tag, (fname, build) in builders.items():
        recs = load_jsonl(processed / fname)
        pairs = build(recs)
        rng.shuffle(pairs)
        save_jsonl(pairs, out_dir / f"{tag}_dpo_full.jsonl")
        for k in range(N_FOLDS):
            save_jsonl([p for p in pairs if p["meta"]["fold"] != k],
                       out_dir / f"{tag}_dpo_fold{k}.jsonl")
        print(f"[{tag}] {len(recs)} items -> {len(pairs)} preference pairs")


if __name__ == "__main__":
    main()
