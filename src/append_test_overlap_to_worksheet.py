"""Append MMLU items that duplicate a hidden-test question but are NOT already
in the worksheet, to the bottom of processed/mmlu_worksheet.jsonl.

This is intentionally append-only and never touches existing rows: the
worksheet may already contain manually-filled statement_A/statement_B/gold
values (real work), so this script only reads the current file, finds what's
missing, and adds new blank rows at the end for you to convert.

"Missing" = any MMLU item, in ANY subject, whose question text exactly matches
a question in PlurVA-LLM_Test_Set/sri_lankan_test_without_gold.jsonl, that
isn't already present in the worksheet (matched by subject + q_no).

Every row (old and newly appended) gets a `test_match_id` field: the hidden
test ID(s) that item's question duplicates, or "" if none. This lets you see,
for every item already in your worksheet, whether it also happens to leak into
the hidden test — not just the newly appended ones.

Newly appended rows have no value tag (tagging was only run on the original
Civics/Health selection), so value_english is left blank and no worked example
is appended to their prompt — same fallback behavior as build_mmlu_worksheet.py
uses for any untagged item.

  python src/append_test_overlap_to_worksheet.py
"""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from build_mmlu_worksheet import (SUBJECT_ABBR, GEMINI_PROMPT_TEMPLATE,
                                  one_example_per_value, format_example_block)

PUNCT = re.compile(r"[.,?!;:\"'()\[\]{}\-–—/\\]")
WS = re.compile(r"[\s​‍]+")


def norm(t):
    t = unicodedata.normalize("NFC", t or "")
    return PUNCT.sub("", WS.sub(" ", t)).strip().lower()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_worksheet(path):
    """Worksheet may be JSONL (one object per line) or a pretty-printed JSON
    array (e.g. after being reformatted by an editor) — handle both."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    return load_jsonl(path)


def find_test_matches(mmlu, test):
    """question-text -> list of test IDs, for exact matches only."""
    test_q = {}
    for t in test:
        test_q.setdefault(norm(t["Question"]), []).append(t["ID"])
    matches = {}  # (subject, q_no) -> [test_id, ...]
    for m in mmlu:
        k = norm(m.get("question", ""))
        if k in test_q:
            matches[(m.get("subject", "").strip(), m.get("q_no"))] = test_q[k]
    return matches


def build_row(m, tags, examples_by_value, test_match_ids):
    subj = m.get("subject", "").strip()
    qno = m.get("q_no")
    key = f"{subj}|{qno}"
    tag = tags.get(key)
    value = tag["value"] if tag else ""

    choices = m.get("choices", [])
    ai = m.get("answer")
    answer_text = (choices[ai - 1] if isinstance(ai, int) and 1 <= ai <= len(choices)
                   else "")
    abbr = SUBJECT_ABBR.get(subj, subj[:3].upper())
    uid = f"MMLU_{abbr}_{qno}"

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        question=m.get("question", ""), answer=answer_text,
        value=value or "(no tag — assign one manually)")
    example = examples_by_value.get(value)
    if example is not None:
        prompt += format_example_block(example)

    return {
        "uid": uid,
        "subject": subj,
        "q_no": qno,
        "question": m.get("question", ""),
        "choice_1": choices[0] if len(choices) > 0 else "",
        "choice_2": choices[1] if len(choices) > 1 else "",
        "choice_3": choices[2] if len(choices) > 2 else "",
        "choice_4": choices[3] if len(choices) > 3 else "",
        "correct_choice_idx": ai,
        "correct_choice_text": answer_text,
        "value_english": value,
        "gemini_prompt": prompt,
        "statement_A": "",
        "statement_B": "",
        "gold": "",
        "test_match_id": ", ".join(test_match_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu", default=str(ROOT / "sinhala_mmlu.json"))
    ap.add_argument("--test", default=str(
        ROOT / "PlurVA-LLM_Test_Set" / "sri_lankan_test_without_gold.jsonl"))
    ap.add_argument("--tags", default=str(ROOT / "processed" / "mmlu_value_tags.json"))
    ap.add_argument("--si_dev", default=str(ROOT / "processed" / "sri_lankan.jsonl"))
    ap.add_argument("--worksheet", default=str(ROOT / "processed" / "mmlu_worksheet.jsonl"))
    ap.add_argument("--out_csv", default=str(ROOT / "processed" / "mmlu_worksheet.csv"))
    args = ap.parse_args()

    mmlu = load_jsonl(args.mmlu)
    test = load_jsonl(args.test)
    tags = json.load(open(args.tags, encoding="utf-8"))
    examples_by_value = one_example_per_value(args.si_dev)

    existing = load_worksheet(args.worksheet)
    existing_keys = {(r["subject"], r["q_no"]) for r in existing}
    print(f"worksheet currently has {len(existing)} rows")

    matches = find_test_matches(mmlu, test)
    print(f"MMLU items exactly matching a hidden-test question (all subjects): "
          f"{len(matches)}")

    # backfill test_match_id on existing rows (informational; nothing else touched)
    n_existing_flagged = 0
    for r in existing:
        tids = matches.get((r["subject"], r["q_no"]))
        if "test_match_id" not in r:
            r["test_match_id"] = ", ".join(tids) if tids else ""
        if tids:
            n_existing_flagged += 1
    print(f"  of which already in the worksheet: {n_existing_flagged}")

    mmlu_by_key = {(m.get("subject", "").strip(), m.get("q_no")): m for m in mmlu}
    new_rows = []
    for key, tids in matches.items():
        if key in existing_keys:
            continue
        m = mmlu_by_key[key]
        new_rows.append(build_row(m, tags, examples_by_value, tids))

    print(f"  NOT in the worksheet -> appending {len(new_rows)} new rows")
    by_subj = {}
    for r in new_rows:
        by_subj[r["subject"]] = by_subj.get(r["subject"], 0) + 1
    print("  new rows by subject:", by_subj)

    combined = existing + new_rows

    with open(args.worksheet, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    fieldnames = list(combined[0].keys())
    with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in combined:
            w.writerow(r)

    print(f"\ntotal rows now: {len(combined)}")
    print(f"  -> {args.worksheet}")
    print(f"  -> {args.out_csv}")


if __name__ == "__main__":
    main()
