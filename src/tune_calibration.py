"""Offline calibration sweep (§2.3, §2.5) — no GPU needed.

Input: per-item details files written by evaluate.py / self_consistency.py /
ensemble.py on CV folds, run with --prior_tau 0 (raw probabilities). Pool the
5 fold files (each item then appears exactly once, scored by an adapter that
never saw it) and this script finds, on those pooled out-of-fold predictions:

  * chinese / indonesian : best prior-calibration temperature tau
        pred = argmax probs(label) * prior(label)^tau
    (prior estimated excluding the item's own fold — CV-clean)
  * sri_lankan           : best per-statement thresholds (th_A, th_B) for
    composing A/B/Both/0 from p_yes_A/p_yes_B. Separate thresholds are swept
    because the dev Yes-rates are asymmetric (stmt A 73%, stmt B 38%).

  python src/tune_calibration.py --details processed/eval_fold*.jsonl
Apply the reported values via --prior_tau / --si_threshold (or th_A/th_B with
ensemble.py) when predicting on the hidden test.
"""
import argparse
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAU_GRID = [round(x * 0.1, 1) for x in range(0, 16)]          # 0.0 .. 1.5
TH_GRID = [round(0.20 + 0.02 * i, 2) for i in range(31)]      # 0.20 .. 0.80


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fold_priors(processed):
    """dataset -> fold -> prior over gold labels excluding that fold."""
    out = {}
    for ds in ("chinese", "indonesian"):
        recs = load_jsonl(processed / f"{ds}.jsonl")
        folds = sorted({r["fold"] for r in recs})
        out[ds] = {}
        for f in folds:
            c = collections.Counter(r["gold"] for r in recs if r["fold"] != f)
            n = sum(c.values())
            out[ds][f] = {k: v / n for k, v in c.items()}
    return out


def gold_lookup(processed):
    """uid -> (fold, correct-set) for scoring; ID uses the consensus set."""
    out = {}
    for ds in ("chinese", "indonesian", "sri_lankan"):
        for r in load_jsonl(processed / f"{ds}.jsonl"):
            accept = set(r["consensus"]) if ds == "indonesian" else {r["gold"]}
            out[r["uid"]] = (ds, r["fold"], accept, r)
    return out


def sweep_tau(rows, priors, lookup):
    best = (None, -1.0)
    curve = {}
    for tau in TAU_GRID:
        n_ok = 0
        for row in rows:
            ds, fold, accept, _ = lookup[row["uid"]]
            prior = priors[ds][fold]
            probs = {k: v * (prior.get(k, 1e-9) ** tau) if tau > 0 else v
                     for k, v in row["probs"].items()}
            n_ok += max(probs, key=probs.get) in accept
        acc = n_ok / len(rows)
        curve[tau] = round(acc, 4)
        if acc > best[1]:
            best = (tau, acc)
    return best, curve


def sweep_si(rows, lookup):
    best = (None, None, -1.0)
    for th_a in TH_GRID:
        for th_b in TH_GRID:
            n_ok = 0
            for row in rows:
                _, _, accept, _ = lookup[row["uid"]]
                a = row["p_yes_A"] >= th_a
                b = row["p_yes_B"] >= th_b
                pred = "Both" if (a and b) else "A" if a else "B" if b else "0"
                n_ok += pred in accept
            acc = n_ok / len(rows)
            if acc > best[2]:
                best = (th_a, th_b, acc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", nargs="+", required=True,
                    help="details JSONL files/globs from CV-fold runs (tau=0)")
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    args = ap.parse_args()
    processed = Path(args.processed_dir)

    paths = []
    for pat in args.details:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    rows = []
    for p in paths:
        rows.extend(load_jsonl(p))
    print(f"{len(rows)} detail rows from {len(paths)} file(s)")

    lookup = gold_lookup(processed)
    priors = fold_priors(processed)
    by_ds = collections.defaultdict(dict)  # dedupe: last row per uid wins
    for row in rows:
        if row["uid"] in lookup:
            by_ds[lookup[row["uid"]][0]][row["uid"]] = row

    report = {}
    for ds in ("chinese", "indonesian"):
        items = [r for r in by_ds[ds].values() if "probs" in r]
        if not items:
            continue
        (tau, acc), curve = sweep_tau(items, priors, lookup)
        report[ds] = {"best_tau": tau, "acc_at_best": acc, "n": len(items),
                      "acc_at_tau0": curve[0.0], "curve": curve}
        print(f"[{ds}] n={len(items)}  tau=0: {curve[0.0]:.4f}  ->  "
              f"best tau={tau}: {acc:.4f}")

    si_items = [r for r in by_ds["sri_lankan"].values() if "p_yes_A" in r]
    if si_items:
        th_a, th_b, acc = sweep_si(si_items, lookup)
        # accuracy at the default 0.5/0.5 for reference
        n_ok = 0
        for row in si_items:
            _, _, accept, _ = lookup[row["uid"]]
            a, b = row["p_yes_A"] >= 0.5, row["p_yes_B"] >= 0.5
            pred = "Both" if (a and b) else "A" if a else "B" if b else "0"
            n_ok += pred in accept
        report["sri_lankan"] = {"best_th_A": th_a, "best_th_B": th_b,
                                "acc_at_best": acc, "n": len(si_items),
                                "acc_at_0.5": n_ok / len(si_items)}
        print(f"[sri_lankan] n={len(si_items)}  th=0.5/0.5: "
              f"{n_ok/len(si_items):.4f}  ->  best th_A={th_a} th_B={th_b}: {acc:.4f}")

    out = processed / "calibration.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
