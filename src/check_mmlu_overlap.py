"""Duplication check: SinhalaMMLU (civics + all subjects) vs the shared-task SI dev set.

The organizers built the Sri Lankan dataset partly by *reformulating* SinhalaMMLU
items, so overlap may not be verbatim. We therefore check several signals:

  1. exact match on normalized question text
  2. high fuzzy similarity on question text (difflib ratio + char-3gram Jaccard)
  3. option/statement reuse — an MMLU choice appearing as an SI statement
     (this is the likeliest leak: the question gets rewritten but the answer
     options are carried over)

Any MMLU item that trips these against a dev item is a candidate duplicate and
must be dropped before using it as auxiliary training data, otherwise CV scores
are inflated by items the model trained on.

  python src/check_mmlu_overlap.py                    # civics only (default)
  python src/check_mmlu_overlap.py --all_subjects     # every MMLU subject
"""
import argparse
import collections
import difflib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# thresholds — deliberately loose so we see near-misses too, and can eyeball them
EXACT = 1.0
HIGH = 0.85      # near-certain duplicate
MEDIUM = 0.70    # worth a human look


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(text):
    """Lowercase-ish normalization for Sinhala: strip punctuation/whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\s​‍]+", " ", text)      # incl. zero-width joiners
    text = re.sub(r"[.,?!;:\"'()\[\]{}\-–—/\\]", "", text)
    return text.strip().lower()


def char_ngrams(text, n=3):
    text = normalize(text)
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(a, b):
    """Blend of sequence ratio and char-3gram Jaccard; robust to reordering."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    jac = jaccard(char_ngrams(a), char_ngrams(b))
    return max(seq, jac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu", default=str(ROOT / "sinhala_mmlu.json"))
    ap.add_argument("--si_dev", default=str(ROOT / "processed" / "sri_lankan.jsonl"))
    ap.add_argument("--subject", default="Civics")
    ap.add_argument("--all_subjects", action="store_true",
                    help="check every MMLU subject, not just --subject")
    ap.add_argument("--out", default=str(ROOT / "processed" / "mmlu_overlap_report.json"))
    args = ap.parse_args()

    mmlu = load_jsonl(args.mmlu)
    dev = load_jsonl(args.si_dev)
    print(f"SinhalaMMLU: {len(mmlu)} items   SI dev: {len(dev)} items\n")

    if args.all_subjects:
        subset = mmlu
        label = "ALL SUBJECTS"
    else:
        subset = [r for r in mmlu if r.get("subject", "").strip().lower()
                  == args.subject.strip().lower()]
        label = args.subject
    print(f"checking {len(subset)} MMLU items ({label}) against {len(dev)} dev items")

    # pre-compute dev text blobs
    dev_q = [(d["uid"], d["question"]) for d in dev]
    dev_stmts = []
    for d in dev:
        for k in ("A", "B"):
            if d["options"].get(k):
                dev_stmts.append((d["uid"], k, d["options"][k]))
    dev_q_norm = {normalize(q): uid for uid, q in dev_q}
    dev_stmt_norm = {normalize(s): (uid, k) for uid, k, s in dev_stmts}

    findings = []
    for m in subset:
        mq = m.get("question", "")
        rec = {"q_no": m.get("q_no"), "subject": m.get("subject"),
               "question": mq, "hits": []}

        # 1. exact question match
        if normalize(mq) in dev_q_norm:
            rec["hits"].append({"type": "exact_question", "score": 1.0,
                                "dev_uid": dev_q_norm[normalize(mq)]})

        # 2. fuzzy question match (best dev item)
        best_uid, best_score = None, 0.0
        for uid, q in dev_q:
            s = similarity(mq, q)
            if s > best_score:
                best_uid, best_score = uid, s
        if best_score >= MEDIUM:
            rec["hits"].append({"type": "fuzzy_question", "score": round(best_score, 3),
                                "dev_uid": best_uid})
        rec["best_question_sim"] = round(best_score, 3)
        rec["best_question_uid"] = best_uid

        # 3. option/statement reuse (any MMLU choice vs any dev statement)
        best_opt = {"score": 0.0}
        for ci, choice in enumerate(m.get("choices", [])):
            if normalize(choice) in dev_stmt_norm:
                uid, k = dev_stmt_norm[normalize(choice)]
                rec["hits"].append({"type": "exact_option", "score": 1.0,
                                    "dev_uid": uid, "dev_stmt": k,
                                    "mmlu_choice_idx": ci + 1})
            for uid, k, s in dev_stmts:
                sc = similarity(choice, s)
                if sc > best_opt["score"]:
                    best_opt = {"score": round(sc, 3), "dev_uid": uid,
                                "dev_stmt": k, "mmlu_choice_idx": ci + 1}
        if best_opt["score"] >= MEDIUM:
            rec["hits"].append({"type": "fuzzy_option", **best_opt})
        rec["best_option_sim"] = best_opt["score"]

        if rec["hits"]:
            findings.append(rec)

    # ---- summary ----
    by_type = collections.Counter(h["type"] for r in findings for h in r["hits"])
    flagged_high = [r for r in findings
                    if max((h["score"] for h in r["hits"]), default=0) >= HIGH]
    flagged_med = [r for r in findings
                   if MEDIUM <= max((h["score"] for h in r["hits"]), default=0) < HIGH]

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)
    print(f"MMLU items with ANY signal (>= {MEDIUM}): {len(findings)} / {len(subset)}")
    print(f"  near-certain duplicates (>= {HIGH}): {len(flagged_high)}")
    print(f"  worth eyeballing ({MEDIUM}-{HIGH}):   {len(flagged_med)}")
    print(f"  signal breakdown: {dict(by_type)}")

    # distribution of best similarity across ALL checked items (not just hits)
    all_q_sims = []
    for m in subset:
        best = max((similarity(m.get("question", ""), q) for _, q in dev_q),
                   default=0.0)
        all_q_sims.append(best)
    buckets = collections.Counter()
    for s in all_q_sims:
        if s >= 0.9: buckets[">=0.90"] += 1
        elif s >= 0.8: buckets["0.80-0.90"] += 1
        elif s >= 0.7: buckets["0.70-0.80"] += 1
        elif s >= 0.5: buckets["0.50-0.70"] += 1
        else: buckets["<0.50"] += 1
    print(f"\n  best question-similarity distribution across all {len(subset)} items:")
    for k in [">=0.90", "0.80-0.90", "0.70-0.80", "0.50-0.70", "<0.50"]:
        if buckets.get(k):
            print(f"    {k}: {buckets[k]}")

    if flagged_high:
        print(f"\n  --- near-certain duplicates ---")
        for r in flagged_high[:10]:
            top = max(r["hits"], key=lambda h: h["score"])
            print(f"    q_no={r['q_no']} [{top['type']} {top['score']}] "
                  f"-> {top.get('dev_uid')}")
    if flagged_med:
        print(f"\n  --- medium-similarity (sample) ---")
        for r in flagged_med[:8]:
            top = max(r["hits"], key=lambda h: h["score"])
            print(f"    q_no={r['q_no']} [{top['type']} {top['score']}] "
                  f"-> {top.get('dev_uid')}")

    out = {
        "checked_subject": label, "n_mmlu_checked": len(subset), "n_dev": len(dev),
        "thresholds": {"high": HIGH, "medium": MEDIUM},
        "n_any_signal": len(findings), "n_high": len(flagged_high),
        "n_medium": len(flagged_med), "signal_breakdown": dict(by_type),
        "drop_q_nos": sorted(r["q_no"] for r in flagged_high),
        "review_q_nos": sorted(r["q_no"] for r in flagged_med),
        "findings": findings,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
