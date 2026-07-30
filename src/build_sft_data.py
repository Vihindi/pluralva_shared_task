"""Build Stage-1 SFT training files (chat-format JSONL) from processed data.

Per methodology §2.4:
  * Option-permutation augmentation (Chinese, Indonesian): each item can be
    emitted under several cyclic option orderings with the gold letter / votes
    remapped, teaching order-invariance (mitigates MCQ selection bias). DEFAULT
    --n_perms is 1 (no augmentation, full rationale coverage); pass --n_perms 4
    for the 4x augmentation. Value-context injection is OFF by default; pass
    --value_summaries auto to re-enable it.
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
Additional output:
  zh_train_remaining_permutations.jsonl - the 20 non-cyclic members of each
                                          Chinese question's complete 4! set
Each line: {"messages":[{system},{user},{assistant}], "meta":{...}}
Joint multi-country training = pass several files to train_lora.py.
"""
import argparse
import itertools
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
CYCLIC_ORDERS = {
    tuple(LETTERS4[(i + shift) % 4] for i in range(4))
    for shift in CYCLIC_SHIFTS
}
REMAINING_ORDERS = [
    order for order in itertools.permutations(LETTERS4)
    if order not in CYCLIC_ORDERS
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def permute_record_order(rec, order):
    """Return a copy whose new A/B/C/D positions contain ``order`` old labels."""
    if sorted(order) != LETTERS4:
        raise ValueError(f"invalid four-option permutation: {order}")
    new = dict(rec)
    new_options = {}
    old_to_new = {}
    for new_ltr, old_ltr in zip(LETTERS4, order):
        new_options[new_ltr] = rec["options"][old_ltr]
        old_to_new[old_ltr] = new_ltr
    new["options"] = new_options
    if "gold" in rec and rec["gold"] in old_to_new:
        new["gold"] = old_to_new[rec["gold"]]
    if "votes" in rec:
        new["votes"] = [old_to_new[v] for v in rec["votes"]]
        new["consensus"] = sorted(old_to_new[c] for c in rec["consensus"])
    return new, old_to_new


def permute_record(rec, shift):
    """Return a copy of rec with options cyclically shifted and labels remapped.

    New position i (letter LETTERS4[i]) holds the option originally at letter
    LETTERS4[(i+shift) % 4]. old_letter -> new_letter map inverts that.
    """
    order = tuple(LETTERS4[(i + shift) % 4] for i in range(4))
    return permute_record_order(rec, order)


def make_example(rec, target_letter, rationales, rationale_key, si_statement=None,
                 value_summaries="auto", meta_extra=None,
                 si_negation_prompt=False):
    messages = build_messages(rec, mode="direct", si_statement=si_statement,
                              value_summaries=value_summaries,
                              si_negation_prompt=si_negation_prompt)
    rat = rationales.get(rationale_key)
    if rat:
        assistant = f"{rat['text']}\nAnswer: {target_letter}"
        msgs = build_messages(rec, mode="cot", si_statement=si_statement,
                              value_summaries=value_summaries,
                              si_negation_prompt=si_negation_prompt)
        messages = msgs
    else:
        assistant = f"Answer: {target_letter}"
    meta = {"uid": rec["uid"], "dataset": rec["dataset"],
            "fold": rec["fold"], "with_rationale": bool(rat)}
    if meta_extra:
        meta.update(meta_extra)
    return {
        "messages": messages + [{"role": "assistant", "content": assistant}],
        "meta": meta,
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


def build_chinese(recs, rationales, n_perms, value_summaries="auto"):
    out = []
    for rec in recs:
        for shift in shifts_for(rationales, rec["uid"], n_perms):
            p, _ = permute_record(rec, shift)
            key = rationale_key_for(rationales, rec["uid"], shift)
            out.append(make_example(p, p["gold"], rationales, key,
                                    value_summaries=value_summaries,
                                    meta_extra={
                                        "augmentation_source": "primary",
                                        "permutation_order": "".join(
                                            LETTERS4[
                                                (i + shift) % 4]
                                            for i in range(4)),
                                    }))
    return out


def build_chinese_remaining_permutations(
        recs, value_summaries="auto"):
    """The 20 non-cyclic members of the complete 4! option-order set."""
    out = []
    for rec in recs:
        for order in REMAINING_ORDERS:
            permuted, _ = permute_record_order(rec, order)
            out.append(make_example(
                permuted,
                permuted["gold"],
                rationales={},
                rationale_key=None,
                value_summaries=value_summaries,
                meta_extra={
                    "augmentation_source": "remaining_permutation",
                    "permutation_order": "".join(order),
                },
            ))
    return out


def load_reviewed_labels(path):
    """Load indonesian_72_reviewed_labels.jsonl -> {ID: {reviewed_label,
    use_in_main_sft, ...}}. These are the 72 tied dev items with a human-picked
    single label."""
    d = {}
    for r in load_jsonl(Path(path)):
        d[r["ID"]] = r
    return d


def id_majority_gold(rec, reviewed):
    """Single hard gold for an Indonesian item under 'majority' mode.

    - clear majority (consensus set size 1): that label.
    - tied (size 2): the human reviewed_label; None if the reviewer flagged it
      use_in_main_sft=false (genuinely ambiguous -> exclude the item).
    Falls back to the representative consensus label if a tie somehow isn't in
    the reviewed file (shouldn't happen: all 72 ties are covered)."""
    cons = rec["consensus"]
    if len(cons) == 1:
        return cons[0]
    rev = reviewed.get(rec["uid"]) if reviewed else None
    if rev is None:
        return cons[0]
    if not rev.get("use_in_main_sft", True):
        return None
    return rev["reviewed_label"]


def build_indonesian(recs, rationales, n_perms, value_summaries="auto",
                     id_mode="probability", reviewed=None):
    out = []
    for rec in recs:
        if id_mode == "majority":
            # one hard-label example per item; ties resolved by human review
            gold = id_majority_gold(rec, reviewed)
            if gold is None:
                continue  # ambiguous item excluded from training
            rec_g = dict(rec, gold=gold)  # so permute_record remaps the label
            for shift in shifts_for(rationales, rec["uid"], n_perms):
                p, _ = permute_record(rec_g, shift)
                key = rationale_key_for(rationales, rec["uid"], shift)
                rkey = key if (key and rationales[key]["answer"] == p["gold"]) else None
                out.append(make_example(p, p["gold"], rationales, rkey,
                                        value_summaries=value_summaries))
        else:
            for shift in shifts_for(rationales, rec["uid"], n_perms):
                p, _ = permute_record(rec, shift)
                key = rationale_key_for(rationales, rec["uid"], shift)
                # vote-expansion: one example per annotator vote == soft-label CE
                for vote in p["votes"]:
                    # a rationale argues for one answer; never pair it with a
                    # different vote's target letter
                    vote_key = key if (key and rationales[key]["answer"] == vote) else None
                    out.append(make_example(p, vote, rationales, vote_key,
                                            value_summaries=value_summaries))
    return out


def build_sri_lankan(recs, rationales, value_summaries="auto",
                     si_mode="binary", si_negation_prompt=False):
    out = []
    for rec in recs:
        if si_mode == "4way":
            # one example per item; target is the 4-way gold in {A,B,Both,0},
            # si_statement=None -> both statements shown in one 4-way prompt
            out.append(make_example(rec, rec["gold"], rationales, rec["uid"],
                                    si_statement=None,
                                    value_summaries=value_summaries))
        else:
            for stmt, ok in (("A", rec["stmt_A_ok"]), ("B", rec["stmt_B_ok"])):
                target = "Yes" if ok else "No"
                key = f"{rec['uid']}_{stmt}"
                out.append(make_example(rec, target, rationales, key, si_statement=stmt,
                                        value_summaries=value_summaries,
                                        si_negation_prompt=si_negation_prompt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--out_dir", default=str(ROOT / "sft_data"))
    ap.add_argument("--rationales", nargs="+", default=None,
                    help="one or more rationales.jsonl files from "
                         "bootstrap_rationales.py (e.g. the real dev rationales "
                         "plus a separate aux-data rationales file); merged into "
                         "one lookup, keys must not collide across files")
    ap.add_argument("--si_aux_files", nargs="+", default=[],
                    help="additional sri_lankan-schema jsonl files to merge into "
                         "the Sri Lankan build (e.g. processed_mmlu_aux/"
                         "sri_lankan.jsonl). These are NEVER read from "
                         "--processed_dir, so the real 203-item dev set stays "
                         "untouched; auxiliary items must already carry "
                         "fold=-1 so they train in every fold but are never "
                         "used for CV evaluation.")
    ap.add_argument("--n_perms", type=int, default=1,
                    help="cyclic option permutations for ZH/ID. DEFAULT 1 "
                         "(no augmentation); pass 4 to re-enable the 4x "
                         "position-debiasing augmentation.")
    ap.add_argument("--value_summaries", default=None,
                    help="'auto' -> inject the per-country files "
                         "zh/id/si_value_summaries.json at the repo root; or a "
                         "path to one custom summaries json applied to ALL "
                         "datasets. DEFAULT: OFF (no value-context block).")
    ap.add_argument("--no_value_summaries", action="store_true",
                    help="explicitly disable value-context injection (this is now "
                         "the default; kept for clarity/back-compat)")
    ap.add_argument("--si_mode", choices=["binary", "4way"], default="binary",
                    help="Sri Lankan target format: 'binary' (default) makes two "
                         "Yes/No examples per item; '4way' makes one A/B/Both/0 "
                         "example per item (both statements shown). Must match the "
                         "--si_mode used for bootstrap_rationales.py and evaluate.py.")
    ap.add_argument(
        "--enable_si_negation_prompt",
        action="store_true",
        help="binary Sinhala only: route verified negative/exclusion questions "
             "to the specialized instruction. Default: disabled; every row "
             "uses the normal binary prompt.",
    )
    ap.add_argument("--id_mode", choices=["probability", "majority"],
                    default="probability",
                    help="Indonesian target construction: 'probability' (default) "
                         "= vote-expansion (one example per annotator vote, keeps "
                         "the 5-vote soft label / consensus ties). 'majority' = one "
                         "hard-label example per item; the 72 tied items are "
                         "resolved by --id_reviewed_labels and the 3 flagged "
                         "ambiguous ones are dropped.")
    ap.add_argument("--id_reviewed_labels", default=None,
                    help="path to indonesian_72_reviewed_labels.jsonl; required "
                         "when --id_mode majority.")
    ap.add_argument(
        "--only_zh_remaining_permutations",
        action="store_true",
        help="write only zh_train_remaining_permutations.jsonl: the 20 "
             "non-cyclic members of each Chinese question's complete 4! "
             "option-order set; do not rebuild the normal SFT files",
    )
    args = ap.parse_args()
    processed, out_dir = Path(args.processed_dir), Path(args.out_dir)

    if args.no_value_summaries or not args.value_summaries:
        args.value_summaries = None
        print("value-context injection: DISABLED (default)")
    elif args.value_summaries == "auto":
        args.value_summaries = "auto"
        print("value-context injection: AUTO (zh/id/si_value_summaries.json "
              "at repo root, per country)")
    else:
        from prompts import load_value_summaries
        custom_path = args.value_summaries
        args.value_summaries = load_value_summaries(custom_path)
        print(f"loaded {len(args.value_summaries)} value summaries from "
              f"{custom_path!r} (applied to all datasets whose keys match)")

    if args.only_zh_remaining_permutations:
        chinese = load_jsonl(processed / "chinese.jsonl")
        examples = build_chinese_remaining_permutations(
            chinese, args.value_summaries)
        random.Random(SEED).shuffle(examples)
        output_path = out_dir / "zh_train_remaining_permutations.jsonl"
        save_jsonl(examples, output_path)
        print(
            f"[zh-extra] {len(chinese)} items x {len(REMAINING_ORDERS)} "
            f"remaining permutations -> {len(examples)} examples")
        print(f"saved -> {output_path}")
        return

    rationales = {}
    if args.rationales:
        for rat_path in args.rationales:
            n_before = len(rationales)
            for r in load_jsonl(Path(rat_path)):
                if r["key"] in rationales:
                    print(f"  WARNING: duplicate rationale key {r['key']!r} in "
                          f"{rat_path!r} — later file wins")
                rationales[r["key"]] = {"text": r["rationale"],
                                        "answer": r.get("answer"),
                                        "shift": r.get("shift", 0)}
            print(f"  loaded {len(rationales) - n_before} rationales from {rat_path!r}")
        print(f"loaded {len(rationales)} rationales total")

    reviewed = None
    if args.id_mode == "majority":
        if not args.id_reviewed_labels:
            raise SystemExit("--id_mode majority requires --id_reviewed_labels "
                             "(indonesian_72_reviewed_labels.jsonl)")
        reviewed = load_reviewed_labels(args.id_reviewed_labels)
        n_excl = sum(1 for r in reviewed.values() if not r.get("use_in_main_sft", True))
        print(f"Indonesian mode: MAJORITY (loaded {len(reviewed)} reviewed ties, "
              f"{n_excl} flagged ambiguous -> excluded)")
    else:
        print("Indonesian mode: PROBABILITY (vote-expansion / consensus)")

    vs = args.value_summaries
    builders = {
        "zh": ("chinese.jsonl", lambda recs: build_chinese(recs, rationales, args.n_perms, vs)),
        "id": ("indonesian.jsonl", lambda recs: build_indonesian(recs, rationales, args.n_perms, vs, args.id_mode, reviewed)),
        "si": ("sri_lankan.jsonl", lambda recs: build_sri_lankan(
            recs, rationales, vs, args.si_mode,
            args.enable_si_negation_prompt)),
    }
    rng = random.Random(SEED)
    summary = {}
    for tag, (fname, build) in builders.items():
        recs = load_jsonl(processed / fname)
        n_main = len(recs)
        if tag == "si" and args.si_aux_files:
            main_uids = {r["uid"] for r in recs}
            for aux_path in args.si_aux_files:
                aux_recs = load_jsonl(Path(aux_path))
                collide = [r["uid"] for r in aux_recs if r["uid"] in main_uids]
                if collide:
                    raise SystemExit(f"aux file {aux_path!r} has uids that "
                                     f"collide with the real dev set: {collide[:5]}")
                non_neg_fold = [r["uid"] for r in aux_recs if r.get("fold", -1) != -1]
                if non_neg_fold:
                    print(f"  WARNING: {len(non_neg_fold)} items in {aux_path!r} "
                          f"don't have fold=-1 — they will be excluded from some "
                          f"fold-specific training files: {non_neg_fold[:5]}")
                recs = recs + aux_recs
                print(f"  merged {len(aux_recs)} aux items from {aux_path!r}")
        examples = build(recs)
        rng.shuffle(examples)
        save_jsonl(examples, out_dir / f"{tag}_train_full.jsonl")
        if tag == "zh" and args.n_perms == 4:
            extra_examples = build_chinese_remaining_permutations(recs, vs)
            rng.shuffle(extra_examples)
            save_jsonl(
                extra_examples,
                out_dir / "zh_train_remaining_permutations.jsonl",
            )
            print(
                f"[zh-extra] {len(recs)} items x "
                f"{len(REMAINING_ORDERS)} remaining permutations -> "
                f"{len(extra_examples)} examples")
        for k in range(N_FOLDS):
            fold_ex = [e for e in examples if e["meta"]["fold"] != k]
            save_jsonl(fold_ex, out_dir / f"{tag}_train_fold{k}.jsonl")
        summary[tag] = {
            "source_items": len(recs), "main_items": n_main,
            "aux_items": len(recs) - n_main, "train_full": len(examples),
            "with_rationale": sum(e["meta"]["with_rationale"] for e in examples),
            "per_fold_train": {k: sum(1 for e in examples if e["meta"]["fold"] != k)
                               for k in range(N_FOLDS)},
        }
        if tag == "zh" and args.n_perms == 4:
            summary[tag]["remaining_permutation_rows"] = len(extra_examples)
        aux_note = (f" ({n_main} dev + {len(recs) - n_main} aux)"
                    if len(recs) != n_main else "")
        print(f"[{tag}] {len(recs)} items{aux_note} -> {len(examples)} SFT examples "
              f"({summary[tag]['with_rationale']} with rationales)")

    with open(out_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
