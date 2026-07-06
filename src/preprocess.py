"""Preprocess PlurVA-LLM dev datasets into a unified schema with CV folds.

Input : data/chinese_dev.jsonl, data/indonesian_dev.jsonl, data/sri_lankan_dev.jsonl
Output: processed/{chinese,indonesian,sri_lankan}.jsonl + processed/stats.json

Per-dataset handling
  Chinese    : single gold label in {A,B,C,D}; value taxonomy path kept (native + English).
  Indonesian : Gold_Answer is 5 raw annotator votes ("A, A, A, A, C"). We keep
               the full vote vector, the empirical vote distribution (soft label),
               the consensus set (all labels tied at max frequency — any of them
               scores correct per the task rules), and the agreement level.
  Sri Lankan : gold in {A,B,Both,0} ("None" normalized to "0"). Decomposed into
               two independent binary judgments: stmt_A_ok / stmt_B_ok, so the
               skewed 4-way task can be trained/evaluated as balanced binary tasks.

Folds: deterministic stratified 5-fold (seed 42). Strata = gold label (Chinese,
Sri Lankan) or consensus-set signature (Indonesian), so every fold preserves the
label skew that prior calibration later relies on.
"""
import argparse
import collections
import json
import random
from pathlib import Path

SEED = 42
N_FOLDS = 5
LETTERS4 = ["A", "B", "C", "D"]
SI_LABELS = ["A", "B", "Both", "0"]

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def assign_folds(records, stratum_key, seed=SEED, n_folds=N_FOLDS):
    """Deterministic stratified k-fold: shuffle within each stratum, deal round-robin."""
    rng = random.Random(seed)
    by_stratum = collections.defaultdict(list)
    for i, r in enumerate(records):
        by_stratum[stratum_key(r)].append(i)
    offset = 0  # stagger fold start across strata so small strata spread evenly
    for stratum in sorted(by_stratum):
        idxs = by_stratum[stratum]
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            records[i]["fold"] = (j + offset) % n_folds
        offset += len(idxs)
    return records


def preprocess_chinese(raw):
    out, problems = [], []
    for x in raw:
        gold = x["Gold_Answer"].strip().upper()
        options = {k: x[f"Option_{k}"].strip() for k in LETTERS4}
        if gold not in LETTERS4:
            problems.append((x["ID"], f"bad gold {gold!r}"))
            continue
        if any(not v for v in options.values()):
            problems.append((x["ID"], "empty option"))
            continue
        out.append({
            "uid": x["ID"],
            "dataset": "chinese",
            "value_native": x["Value"].strip(),
            "value_english": x["Value_English"].strip(),
            "scenario": "",
            "question": x["Question"].strip(),
            "options": options,
            "gold": gold,
        })
    return out, problems


def preprocess_indonesian(raw):
    out, problems = [], []
    for x in raw:
        votes = [v.strip().upper() for v in x["Gold_Answer"].split(",")]
        options = {k: x[f"Option_{k}"].strip() for k in LETTERS4}
        if len(votes) != 5 or any(v not in LETTERS4 for v in votes):
            problems.append((x["ID"], f"bad votes {votes!r}"))
            continue
        if any(not v for v in options.values()):
            problems.append((x["ID"], "empty option"))
            continue
        counts = collections.Counter(votes)
        top = max(counts.values())
        consensus = sorted(k for k, c in counts.items() if c == top)
        out.append({
            "uid": x["ID"],
            "dataset": "indonesian",
            "value_native": x["Value"].strip(),
            "value_english": x["Value_English"].strip(),
            "scenario": x.get("Scenario", "").strip(),
            "question": x["Question"].strip(),
            "options": options,
            "votes": votes,
            "vote_dist": {k: counts.get(k, 0) / 5 for k in LETTERS4},
            "consensus": consensus,          # prediction is correct iff in this set
            "agreement": top,                # 2..5 — plurality strength
            "gold": consensus[0],            # representative hard label (reporting only)
        })
    return out, problems


def preprocess_sri_lankan(raw):
    out, problems = [], []
    for x in raw:
        gold = x["Gold_Answer"].strip()
        if gold.lower() == "none":
            gold = "0"
        if gold.capitalize() == "Both":
            gold = "Both"
        else:
            gold = gold.upper() if gold.upper() in ("A", "B") else gold
        options = {"A": x["Option_A"].strip(), "B": x["Option_B"].strip()}
        if gold not in SI_LABELS:
            problems.append((x["ID"], f"bad gold {gold!r}"))
            continue
        if not options["A"] or not options["B"]:
            problems.append((x["ID"], "empty statement"))
            continue
        out.append({
            "uid": x["ID"],
            "dataset": "sri_lankan",
            "value_native": x["Value"].strip(),
            "value_english": x["Value_English"].strip(),
            "scenario": "",
            "question": x["Question"].strip(),
            "options": options,
            "gold": gold,
            # binary decomposition: is each statement an acceptable answer?
            "stmt_A_ok": gold in ("A", "Both"),
            "stmt_B_ok": gold in ("B", "Both"),
        })
    return out, problems


def label_prior(records, key):
    c = collections.Counter(key(r) for r in records)
    n = sum(c.values())
    return {k: v / n for k, v in sorted(c.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "processed"))
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)

    stats, all_problems = {}, {}
    jobs = [
        ("chinese", "chinese_dev.jsonl", preprocess_chinese, lambda r: r["gold"]),
        ("indonesian", "indonesian_dev.jsonl", preprocess_indonesian,
         lambda r: "".join(r["consensus"])),
        ("sri_lankan", "sri_lankan_dev.jsonl", preprocess_sri_lankan, lambda r: r["gold"]),
    ]
    for name, fname, fn, stratum in jobs:
        raw = load_jsonl(data_dir / fname)
        recs, problems = fn(raw)
        dupes = [u for u, c in collections.Counter(r["uid"] for r in recs).items() if c > 1]
        recs = assign_folds(recs, stratum)
        save_jsonl(recs, out_dir / f"{name}.jsonl")
        all_problems[name] = problems

        fold_sizes = collections.Counter(r["fold"] for r in recs)
        s = {
            "n_raw": len(raw), "n_kept": len(recs),
            "n_problems": len(problems), "duplicate_uids": dupes,
            "fold_sizes": dict(sorted(fold_sizes.items())),
            "label_prior": label_prior(recs, lambda r: r["gold"]),
        }
        if name == "indonesian":
            s["consensus_size"] = dict(collections.Counter(len(r["consensus"]) for r in recs))
            s["agreement_dist"] = dict(sorted(collections.Counter(r["agreement"] for r in recs).items()))
        if name == "sri_lankan":
            s["binary_prior"] = {
                "stmt_A_ok": sum(r["stmt_A_ok"] for r in recs) / len(recs),
                "stmt_B_ok": sum(r["stmt_B_ok"] for r in recs) / len(recs),
            }
        stats[name] = s
        print(f"[{name}] kept {len(recs)}/{len(raw)}  problems={len(problems)}  "
              f"folds={dict(sorted(fold_sizes.items()))}")
        for uid, why in problems:
            print(f"    PROBLEM {uid}: {why}")

    # per-fold label balance check (stratification sanity)
    for name, _, _, _ in jobs:
        recs = load_jsonl(out_dir / f"{name}.jsonl")
        for f in range(N_FOLDS):
            fold = [r for r in recs if r["fold"] == f]
            stats[name].setdefault("per_fold_prior", {})[str(f)] = label_prior(
                fold, lambda r: r["gold"])

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
