"""Build an editable worksheet for manually converting selected SinhalaMMLU
items (4-choice) into the SI binary format (2 statements), one row per item.

For each of the 81 selected items (66 Civics + 15 Health, after contamination/
manual exclusions — see tag_mmlu_values.py) this includes everything needed to
prompt Gemini chat by hand and everything needed to paste the result back:

  - q_no, subject, question, all 4 original choices, the correct one marked
  - the value tag already assigned by tag_mmlu_values(_gemini).py
  - a suggested uid (MMLU_<SUBJ>_<q_no>) matching the SI dev naming convention
  - BLANK fields for you to fill after conversion: statement_A, statement_B, gold

Two output formats, always kept in sync:
  processed/mmlu_worksheet.jsonl  - one JSON object per line (for scripts)
  processed/mmlu_worksheet.csv    - same rows, spreadsheet-friendly (for you)

After filling in statement_A / statement_B / gold, feed the JSONL back to
build_mmlu_aux_data.py (not yet written) to produce the final SFT-ready file.

  python src/build_mmlu_worksheet.py
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SUBJECT_ABBR = {"Civics": "CIV", "Health and Physical Science": "HPS"}

GEMINI_PROMPT_TEMPLATE = (
    "Rewrite this factual multiple-choice question into two competing value "
    "statements for a Sri Lankan societal-values judgment task. Question "
    "(Sinhala): {question} | Correct answer: {answer} | Value being tested: "
    "{value}. Write Statement A and Statement B in Sinhala: one statement "
    "must be the value-aligned, socially correct position; the other must be "
    "a plausible but value-violating position. Do not just repeat the "
    "original answer options verbatim — elaborate each into a fuller stance, "
    "matching the style of: 'X should always be done with pride and respect' "
    "vs 'following the rules when doing X is not that essential.'"
)

# Appended verbatim after GEMINI_PROMPT_TEMPLATE, filled with a real SI dev
# item that shares the row's own value category — same block format used when
# the 46 per-value examples were shown in chat.
EXAMPLE_BLOCK_TEMPLATE = (
    "\n\nConsider following example:\n"
    "Value: {value}\n"
    "UID: {uid}\n"
    "Question: {question}\n"
    "A: {a}\n"
    "B: {b}\n"
    "Gold: {gold}"
)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def one_example_per_value(si_dev_path):
    """value_english -> first matching real SI dev record (question/options/gold).

    Same "first occurrence per value" rule used to build the 46-value
    reference list shown earlier in chat, so the examples match exactly.
    """
    dev = load_jsonl(si_dev_path)
    seen = {}
    for d in dev:
        v = d["value_english"]
        if v not in seen:
            seen[v] = d
    return seen


def format_example_block(record):
    return EXAMPLE_BLOCK_TEMPLATE.format(
        value=record["value_english"], uid=record["uid"],
        question=record["question"], a=record["options"]["A"],
        b=record["options"]["B"], gold=record["gold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu", default=str(ROOT / "sinhala_mmlu.json"))
    ap.add_argument("--tags", default=str(ROOT / "processed" / "mmlu_value_tags.json"))
    ap.add_argument("--subjects", nargs="+",
                    default=["Civics", "Health and Physical Science"])
    ap.add_argument("--out_jsonl", default=str(ROOT / "processed" / "mmlu_worksheet.jsonl"))
    ap.add_argument("--out_csv", default=str(ROOT / "processed" / "mmlu_worksheet.csv"))
    ap.add_argument("--si_dev", default=str(ROOT / "processed" / "sri_lankan.jsonl"),
                    help="source of the one-example-per-value used in the prompt")
    args = ap.parse_args()

    from tag_mmlu_values import select_items
    mmlu = load_jsonl(args.mmlu)
    tags = json.load(open(args.tags, encoding="utf-8"))
    subset = select_items(mmlu, args.subjects)
    examples_by_value = one_example_per_value(args.si_dev)

    rows = []
    missing_tags = 0
    missing_examples = []
    for m in subset:
        subj = m.get("subject", "").strip()
        qno = m.get("q_no")
        key = f"{subj}|{qno}"
        tag = tags.get(key)
        if tag is None:
            missing_tags += 1
            value = ""
        else:
            value = tag["value"]

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
        elif value:
            missing_examples.append((uid, value))

        row = {
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
            # --- fill these in after converting with Gemini chat ---
            "statement_A": "",
            "statement_B": "",
            "gold": "",   # A | B | Both | 0
        }
        rows.append(row)
    if missing_examples:
        print(f"WARNING: {len(missing_examples)} rows have a value tag with no "
              f"matching SI dev example (prompt has no worked example): "
              f"{missing_examples}")

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"{len(rows)} rows written ({missing_tags} missing a value tag)")
    print(f"  -> {args.out_jsonl}")
    print(f"  -> {args.out_csv}")
    by_subj = {}
    for r in rows:
        by_subj[r["subject"]] = by_subj.get(r["subject"], 0) + 1
    print("by subject:", by_subj)


if __name__ == "__main__":
    main()
