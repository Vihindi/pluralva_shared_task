"""Convert the completed MMLU worksheet into the sri_lankan.jsonl schema, so it
can be fed to bootstrap_rationales.py to generate reasoning for these items.

Reads processed/mmlu_worksheet.jsonl (JSONL or pretty-printed JSON array —
whichever format it's currently in), keeps only rows where statement_A,
statement_B, and gold have actually been filled in (the 21 rows appended by
append_test_overlap_to_worksheet.py are skipped until you convert them), and
writes one record per row in EXACTLY the schema preprocess.py produces for the
real Sri Lankan dev data:

  {"uid", "dataset": "sri_lankan", "value_native", "value_english", "scenario",
   "question", "options": {"A", "B"}, "gold", "stmt_A_ok", "stmt_B_ok", "fold"}

fold is always -1 (these are auxiliary training items, never a CV eval item —
same convention discussed for the earlier civics augmentation).

CRITICAL: this writes to a SEPARATE directory (default processed_mmlu_aux/),
never to processed/sri_lankan.jsonl — that file is the real 203-item dev set
that every fold/CV number in this pipeline depends on, and must never be
touched by auxiliary data.

  python src/build_mmlu_aux_data.py
  python src/bootstrap_rationales.py --processed_dir processed_mmlu_aux \
      --datasets sri_lankan --out processed/rationales_mmlu_aux.jsonl \
      --model <base> --load_4bit
"""
import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VALID_GOLD = {"A", "B", "Both", "0"}
GOLD_NORMALIZE = {"a": "A", "b": "B", "both": "Both", "0": "0",
                  "none": "0", "neither": "0"}

# English -> Sinhala glosses, reused from the Gemini tagger so value_native
# isn't left empty (not used by SI prompt-building, but kept for schema parity
# with the real dev data).
try:
    from tag_mmlu_values_gemini import VALUE_SINHALA
except Exception:
    VALUE_SINHALA = {}


def load_worksheet(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_gold(raw):
    g = (raw or "").strip()
    if g in VALID_GOLD:
        return g
    return GOLD_NORMALIZE.get(g.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worksheet", default=str(ROOT / "processed" / "mmlu_worksheet.jsonl"))
    ap.add_argument("--out_dir", default=str(ROOT / "processed_mmlu_aux"),
                    help="separate directory — NEVER processed/ (protects the "
                         "real 203-item sri_lankan.jsonl dev set)")
    args = ap.parse_args()

    rows = load_worksheet(args.worksheet)
    print(f"worksheet has {len(rows)} rows")

    converted, skipped = [], collections.Counter()
    skipped_uids = collections.defaultdict(list)
    for r in rows:
        stmt_a = (r.get("statement_A") or "").strip()
        stmt_b = (r.get("statement_B") or "").strip()
        gold = normalize_gold(r.get("gold"))

        if not stmt_a or not stmt_b:
            skipped["missing statement_A/B"] += 1
            skipped_uids["missing statement_A/B"].append(r["uid"])
            continue
        if gold is None:
            skipped[f"invalid/missing gold ({r.get('gold')!r})"] += 1
            skipped_uids[f"invalid/missing gold"].append(r["uid"])
            continue

        value_en = (r.get("value_english") or "").strip()
        if not value_en:
            skipped["no value tag"] += 1
            skipped_uids["no value tag"].append(r["uid"])
            continue

        converted.append({
            "uid": r["uid"],
            "dataset": "sri_lankan",
            "value_native": VALUE_SINHALA.get(value_en, value_en),
            "value_english": value_en,
            "scenario": "",
            "question": r["question"],
            "options": {"A": stmt_a, "B": stmt_b},
            "gold": gold,
            "stmt_A_ok": gold in ("A", "Both"),
            "stmt_B_ok": gold in ("B", "Both"),
            "fold": -1,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sri_lankan.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nconverted: {len(converted)}")
    if skipped:
        print("skipped:")
        for reason, n in skipped.most_common():
            print(f"   {reason}: {n}  {skipped_uids[reason][:6]}"
                  f"{' ...' if n > 6 else ''}")
    gold_dist = collections.Counter(r["gold"] for r in converted)
    print(f"gold distribution: {dict(gold_dist)}")
    print(f"\n-> {out_path}")
    print(f"\nnext step:")
    print(f"  python src/bootstrap_rationales.py --processed_dir {out_dir} \\")
    print(f"      --datasets sri_lankan --out processed/rationales_mmlu_aux.jsonl \\")
    print(f"      --model <base> --load_4bit")


if __name__ == "__main__":
    main()
