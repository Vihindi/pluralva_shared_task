"""Stage-3 probability-level ensembling (§2.6) — no GPU needed.

Averages the per-item probability distributions saved in details files by
evaluate.py and/or self_consistency.py — e.g. the 5 CV-fold adapters each
predicting the full hidden test (LoRA-ensemble), or constrained scoring +
self-consistency for the same adapter. Items are matched by uid; a uid missing
from some files is averaged over the files that have it. Calibration
(--prior_tau, --th_a/--th_b from tune_calibration.py) is applied AFTER
averaging, on dev-estimated priors only.

  # honest fold-ensemble on the hidden test:
  python src/ensemble.py predict --details pred_fold0.details.jsonl ... \
      pred_fold4.details.jsonl --prior_tau 0.5 --th_a 0.48 --th_b 0.52 \
      --out predictions.jsonl
  # sanity: ensemble of CV eval runs (disjoint folds -> pooled OOF accuracy):
  python src/ensemble.py eval --details processed/eval_fold*.jsonl
"""
import argparse
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTERS4 = ["A", "B", "C", "D"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def average_rows(all_rows, weights):
    """uid -> averaged row (probs or p_yes), weighted per source file."""
    acc = {}
    for w, rows in zip(weights, all_rows):
        for row in rows:
            slot = acc.setdefault(row["uid"], {"dataset": row["dataset"],
                                               "probs": collections.defaultdict(float),
                                               "p_yes_A": 0.0, "p_yes_B": 0.0,
                                               "w_mcq": 0.0, "w_si": 0.0})
            if "probs" in row:
                for k, v in row["probs"].items():
                    slot["probs"][k] += w * v
                slot["w_mcq"] += w
            if "p_yes_A" in row:
                slot["p_yes_A"] += w * row["p_yes_A"]
                slot["p_yes_B"] += w * row["p_yes_B"]
                slot["w_si"] += w
    out = {}
    for uid, s in acc.items():
        row = {"uid": uid, "dataset": s["dataset"]}
        if s["w_mcq"] > 0:
            row["probs"] = {k: v / s["w_mcq"] for k, v in s["probs"].items()}
        if s["w_si"] > 0:
            row["p_yes_A"] = s["p_yes_A"] / s["w_si"]
            row["p_yes_B"] = s["p_yes_B"] / s["w_si"]
        out[uid] = row
    return out


def decide(row, priors, args):
    ds = row["dataset"]
    if ds == "sri_lankan":
        a = row["p_yes_A"] >= args.th_a
        b = row["p_yes_B"] >= args.th_b
        return "Both" if (a and b) else "A" if a else "B" if b else "0"
    probs = dict(row["probs"])
    prior = priors.get(ds)
    if prior and args.prior_tau > 0:
        probs = {k: v * (prior.get(k, 1e-9) ** args.prior_tau)
                 for k, v in probs.items()}
    return max(probs, key=probs.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "predict"])
    ap.add_argument("--details", nargs="+", required=True,
                    help="details JSONL files/globs to ensemble")
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--prior_tau", type=float, default=0.0)
    ap.add_argument("--th_a", type=float, default=0.5,
                    help="SI statement-A threshold (from tune_calibration.py)")
    ap.add_argument("--th_b", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="predictions.jsonl (predict mode)")
    ap.add_argument("--details_out", default=None,
                    help="optionally save the averaged distributions")
    args = ap.parse_args()
    # submission "dataset" field names — keep in sync with evaluate.py
    dataset_names = {"chinese": "chinese", "indonesian": "indonesian",
                     "sri_lankan": "sri_lankan"}

    paths = []
    for pat in args.details:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    all_rows = [load_jsonl(p) for p in paths]
    weights = args.weights or [1.0] * len(paths)
    assert len(weights) == len(paths), "--weights must match number of files"
    print(f"ensembling {len(paths)} file(s): {[Path(p).name for p in paths]}")

    merged = average_rows(all_rows, weights)

    processed = Path(args.processed_dir)
    priors, gold = {}, {}
    for ds in dataset_names:
        p = processed / f"{ds}.jsonl"
        if not p.exists():
            continue
        recs = load_jsonl(p)
        c = collections.Counter(r["gold"] for r in recs)
        n = sum(c.values())
        priors[ds] = {k: v / n for k, v in c.items()}
        for r in recs:
            gold[r["uid"]] = set(r["consensus"]) if ds == "indonesian" else {r["gold"]}

    if args.details_out:
        with open(args.details_out, "w", encoding="utf-8") as f:
            for row in merged.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.mode == "eval":
        per_ds = collections.defaultdict(lambda: [0, 0])
        for uid, row in merged.items():
            if uid not in gold:
                continue
            pred = decide(row, priors, args)
            per_ds[row["dataset"]][0] += pred in gold[uid]
            per_ds[row["dataset"]][1] += 1
        results = {}
        for ds, (ok, n) in sorted(per_ds.items()):
            results[ds] = ok / n
            print(f"[{ds}] accuracy {ok/n:.4f}  (n={n})")
        print(f"\nMACRO-AVERAGE: {sum(results.values())/len(results):.4f}")
    else:
        assert args.out, "predict mode needs --out"
        with open(args.out, "w", encoding="utf-8") as f:
            for uid, row in merged.items():
                pred = decide(row, priors, args)
                f.write(json.dumps({"dataset": dataset_names[row["dataset"]],
                                    "id": uid, "LLM_Output": pred},
                                   ensure_ascii=False) + "\n")
        print(f"predictions -> {args.out}  ({len(merged)} items)")


if __name__ == "__main__":
    main()
