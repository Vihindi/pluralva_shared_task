"""Tune country-specific calibration on held-out predictions.

Input files are JSONL details produced by ``src/evaluate.py eval``:

* Chinese/Indonesian rows contain ``uid``, ``dataset``, ``gold``, and
  ``probs`` over A/B/C/D.
* Sinhala rows contain ``uid``, ``dataset``, ``gold``, ``p_yes_A``, and
  ``p_yes_B``.

Example:

    python src/tune_calibration.py \
        --details processed/eval_fold0.jsonl \
                  processed/eval_fold1.jsonl \
                  processed/eval_fold2.jsonl \
                  processed/eval_fold3.jsonl \
                  processed/eval_fold4.jsonl \
        --out processed/calibration.json

Chinese uses held-out-fold gold-label priors. Indonesian uses the mean
annotator vote distribution from the corresponding training folds and scores
a prediction as correct when it belongs to the consensus set. Sinhala tunes
separate Yes cutoffs for statements A and B.
"""

import argparse
import collections
import json
from pathlib import Path


LETTERS4 = ("A", "B", "C", "D")
MCQ_DATASETS = ("chinese", "indonesian")
ALL_DATASETS = (*MCQ_DATASETS, "sri_lankan")
VALID_SI_LABELS = {"A", "B", "Both", "0"}


def load_sinhala_details(paths, required=True):
    """Load and validate unique held-out Sinhala prediction rows."""
    rows = []
    seen = {}
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"details file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc

                if row.get("dataset") != "sri_lankan":
                    continue

                missing = {
                    key
                    for key in ("uid", "gold", "p_yes_A", "p_yes_B")
                    if key not in row
                }
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: missing fields "
                        f"{sorted(missing)}"
                    )
                if row["gold"] not in VALID_SI_LABELS:
                    raise ValueError(
                        f"{path}:{line_number}: invalid Sinhala gold label "
                        f"{row['gold']!r}"
                    )

                for key in ("p_yes_A", "p_yes_B"):
                    try:
                        row[key] = float(row[key])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{path}:{line_number}: {key} is not numeric"
                        ) from exc
                    if not 0.0 <= row[key] <= 1.0:
                        raise ValueError(
                            f"{path}:{line_number}: {key}={row[key]} is "
                            "outside [0, 1]"
                        )

                uid = str(row["uid"])
                if uid in seen:
                    previous = seen[uid]
                    raise ValueError(
                        f"duplicate Sinhala uid {uid!r} in {path}; already "
                        f"seen in {previous}. Pass one held-out prediction per "
                        "source item, not predictions from multiple adapters "
                        "for the same item."
                    )
                seen[uid] = path
                rows.append(row)

    if required and not rows:
        raise ValueError("no Sinhala rows found in the supplied details files")
    return rows


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
    return rows


def compute_mcq_prior(records, dataset, excluded_fold=None):
    """Compute a training-fold label prior for one MCQ country.

    Chinese uses its single gold label. Indonesian averages the complete
    five-annotator vote distribution, preserving ties and disagreement.
    """
    train = [
        row for row in records
        if excluded_fold is None or row.get("fold") != excluded_fold
    ]
    if not train:
        raise ValueError(
            f"no {dataset} training records remain after excluding "
            f"fold {excluded_fold}"
        )

    totals = {label: 0.0 for label in LETTERS4}
    if dataset == "chinese":
        for row in train:
            gold = row.get("gold")
            if gold not in totals:
                raise ValueError(
                    f"{row.get('uid')}: invalid Chinese gold label {gold!r}"
                )
            totals[gold] += 1.0
    elif dataset == "indonesian":
        for row in train:
            vote_dist = row.get("vote_dist")
            if not isinstance(vote_dist, dict):
                raise ValueError(
                    f"{row.get('uid')}: Indonesian record lacks vote_dist"
                )
            for label in LETTERS4:
                totals[label] += float(vote_dist.get(label, 0.0))
    else:
        raise ValueError(f"unsupported MCQ dataset: {dataset}")

    total = sum(totals.values())
    if total <= 0:
        raise ValueError(f"{dataset} prior has zero mass")
    return {label: totals[label] / total for label in LETTERS4}


def load_mcq_details(paths, dataset, processed_dir):
    """Load unique OOF MCQ rows and attach fold-clean priors/scoring labels."""
    if dataset not in MCQ_DATASETS:
        raise ValueError(f"unsupported MCQ dataset: {dataset}")

    processed_path = Path(processed_dir) / f"{dataset}.jsonl"
    if not processed_path.exists():
        raise FileNotFoundError(f"processed data not found: {processed_path}")
    processed = load_jsonl(processed_path)
    processed_by_uid = {str(row["uid"]): row for row in processed}
    if len(processed_by_uid) != len(processed):
        raise ValueError(f"{processed_path}: duplicate uid")

    fold_priors = {
        fold: compute_mcq_prior(processed, dataset, excluded_fold=fold)
        for fold in sorted({row.get("fold") for row in processed})
    }

    rows = []
    seen = {}
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"details file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    detail = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if detail.get("dataset") != dataset:
                    continue

                uid = str(detail.get("uid", ""))
                if not uid:
                    raise ValueError(f"{path}:{line_number}: missing uid")
                if uid in seen:
                    raise ValueError(
                        f"duplicate {dataset} uid {uid!r} in {path}; already "
                        f"seen in {seen[uid]}. Pass one held-out prediction "
                        "per source item."
                    )
                if uid not in processed_by_uid:
                    raise ValueError(
                        f"{path}:{line_number}: {uid!r} is absent from "
                        f"{processed_path}"
                    )

                probs = detail.get("probs")
                if not isinstance(probs, dict):
                    raise ValueError(
                        f"{path}:{line_number}: missing A/B/C/D probs"
                    )
                if set(probs) != set(LETTERS4):
                    raise ValueError(
                        f"{path}:{line_number}: probs must contain exactly "
                        f"{list(LETTERS4)}, got {sorted(probs)}"
                    )
                clean_probs = {}
                for label in LETTERS4:
                    try:
                        probability = float(probs[label])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{path}:{line_number}: probability for {label} "
                            "is not numeric"
                        ) from exc
                    if not 0.0 <= probability <= 1.0:
                        raise ValueError(
                            f"{path}:{line_number}: probability for {label} "
                            f"is outside [0, 1]: {probability}"
                        )
                    clean_probs[label] = probability
                if sum(clean_probs.values()) <= 0:
                    raise ValueError(
                        f"{path}:{line_number}: probabilities have zero mass"
                    )

                source = processed_by_uid[uid]
                fold = source.get("fold")
                if fold not in fold_priors:
                    raise ValueError(f"{uid}: invalid or missing fold {fold!r}")
                row = {
                    "uid": uid,
                    "dataset": dataset,
                    "probs": clean_probs,
                    "gold": source.get("gold"),
                    "fold": fold,
                    "prior": fold_priors[fold],
                }
                if dataset == "indonesian":
                    consensus = source.get("consensus")
                    if (
                        not isinstance(consensus, list)
                        or not consensus
                        or any(label not in LETTERS4 for label in consensus)
                    ):
                        raise ValueError(
                            f"{uid}: invalid Indonesian consensus {consensus!r}"
                        )
                    row["consensus"] = list(consensus)
                seen[uid] = path
                rows.append(row)
    return rows


def predict_mcq(row, tau):
    scores = {
        label: row["probs"][label] * (row["prior"][label] ** tau)
        for label in LETTERS4
    }
    return max(LETTERS4, key=lambda label: scores[label])


def evaluate_tau(rows, dataset, tau):
    predictions = [predict_mcq(row, tau) for row in rows]
    if dataset == "chinese":
        correct = sum(
            prediction == row["gold"]
            for prediction, row in zip(predictions, rows)
        )
    elif dataset == "indonesian":
        correct = sum(
            prediction in set(row["consensus"])
            for prediction, row in zip(predictions, rows)
        )
    else:
        raise ValueError(f"unsupported MCQ dataset: {dataset}")
    return {
        "correct": correct,
        "accuracy": correct / len(rows),
        "prediction_counts": dict(
            sorted(collections.Counter(predictions).items())
        ),
    }


def tune_prior_tau(rows, dataset, values):
    results = [
        {"prior_tau": tau, **evaluate_tau(rows, dataset, tau)}
        for tau in values
    ]
    # Prefer the weakest calibration when accuracy ties.
    results.sort(key=lambda result: (-result["accuracy"], result["prior_tau"]))
    return results


def compose_label(p_yes_a, p_yes_b, th_a, th_b):
    """Convert the two Yes probabilities into A/B/Both/0."""
    a = p_yes_a >= th_a
    b = p_yes_b >= th_b
    return "Both" if a and b else "A" if a else "B" if b else "0"


def evaluate_thresholds(rows, th_a, th_b):
    predictions = [
        compose_label(row["p_yes_A"], row["p_yes_B"], th_a, th_b)
        for row in rows
    ]
    correct = sum(
        prediction == row["gold"]
        for prediction, row in zip(predictions, rows)
    )
    return {
        "correct": correct,
        "accuracy": correct / len(rows),
        "prediction_counts": dict(
            sorted(collections.Counter(predictions).items())
        ),
    }


def threshold_grid(minimum, maximum, step):
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("--grid_min must be inside [0, 1]")
    if not 0.0 <= maximum <= 1.0:
        raise ValueError("--grid_max must be inside [0, 1]")
    if maximum < minimum:
        raise ValueError("--grid_max must be greater than or equal to --grid_min")
    if step <= 0:
        raise ValueError("--grid_step must be positive")

    # Integer construction avoids values such as 0.6000000000000001.
    scale = 1_000_000
    lo = round(minimum * scale)
    hi = round(maximum * scale)
    stride = round(step * scale)
    if stride <= 0:
        raise ValueError("--grid_step is too small")
    values = [value / scale for value in range(lo, hi + 1, stride)]
    if values[-1] < maximum - 1e-12:
        values.append(round(maximum, 6))
    return values


def tau_grid(minimum, maximum, step):
    if minimum < 0.0:
        raise ValueError("--tau_min must be non-negative")
    if maximum < minimum:
        raise ValueError("--tau_max must be greater than or equal to --tau_min")
    if step <= 0:
        raise ValueError("--tau_step must be positive")

    scale = 1_000_000
    lo = round(minimum * scale)
    hi = round(maximum * scale)
    stride = round(step * scale)
    if stride <= 0:
        raise ValueError("--tau_step is too small")
    values = [value / scale for value in range(lo, hi + 1, stride)]
    if values[-1] < maximum - 1e-12:
        values.append(round(maximum, 6))
    return values


def tune_sinhala_thresholds(rows, values):
    """Return every grid result, ranked with a conservative tie-break.

    Accuracy is the primary criterion.  If several pairs have identical
    accuracy, prefer the pair closest to the uncalibrated 0.5/0.5 setting.
    """
    results = []
    for th_a in values:
        for th_b in values:
            metrics = evaluate_thresholds(rows, th_a, th_b)
            results.append({"th_a": th_a, "th_b": th_b, **metrics})

    results.sort(
        key=lambda result: (
            -result["accuracy"],
            abs(result["th_a"] - 0.5) + abs(result["th_b"] - 0.5),
            abs(result["th_a"] - 0.5),
            result["th_a"],
            result["th_b"],
        )
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tune Chinese/Indonesian prior strength and separate binary "
            "Sinhala Yes thresholds from held-out predictions."
        )
    )
    parser.add_argument(
        "--details",
        nargs="+",
        required=True,
        help="held-out evaluate.py details JSONL files (normally five CV folds)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=list(ALL_DATASETS),
        help="countries to tune (default: every country present in --details)",
    )
    parser.add_argument(
        "--processed_dir",
        default="processed",
        help="processed data containing folds, gold, votes, and consensus",
    )
    parser.add_argument("--grid_min", type=float, default=0.20)
    parser.add_argument("--grid_max", type=float, default=0.80)
    parser.add_argument("--grid_step", type=float, default=0.01)
    parser.add_argument("--tau_min", type=float, default=0.0)
    parser.add_argument("--tau_max", type=float, default=1.5)
    parser.add_argument("--tau_step", type=float, default=0.05)
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="number of highest-ranked threshold pairs to print and save",
    )
    parser.add_argument(
        "--out",
        default="processed/calibration.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")

    report = {
        "method": "out_of_fold_country_calibration_grid_search",
        "details": [str(Path(path)) for path in args.details],
        "datasets": {},
    }
    submission_args = []

    tau_values = tau_grid(args.tau_min, args.tau_max, args.tau_step)
    for dataset, argument_name in (
        ("chinese", "zh_prior_tau"),
        ("indonesian", "id_prior_tau"),
    ):
        if dataset not in args.datasets:
            continue
        rows = load_mcq_details(
            args.details, dataset, processed_dir=args.processed_dir
        )
        if not rows:
            continue
        ranked = tune_prior_tau(rows, dataset, tau_values)
        best = ranked[0]
        baseline = evaluate_tau(rows, dataset, 0.0)
        country_report = {
            "n_items": len(rows),
            "prior_method": (
                "training-fold gold-label frequency"
                if dataset == "chinese"
                else "training-fold mean annotator vote distribution"
            ),
            "correctness": (
                "exact gold match"
                if dataset == "chinese"
                else "prediction belongs to consensus set"
            ),
            "grid": {
                "min": args.tau_min,
                "max": args.tau_max,
                "step": args.tau_step,
                "values": len(tau_values),
            },
            "prior_tau": best["prior_tau"],
            "accuracy": best["accuracy"],
            "correct": best["correct"],
            "prediction_counts": best["prediction_counts"],
            "baseline_tau_0": baseline,
            "absolute_improvement": best["accuracy"] - baseline["accuracy"],
            "top_results": ranked[: args.top_k],
        }
        report["datasets"][dataset] = country_report
        report[argument_name] = best["prior_tau"]
        submission_args.append(
            f"--{argument_name} {best['prior_tau']:.6g}"
        )

        print(f"{dataset.title()} held-out items: {len(rows)}")
        print(
            f"Baseline tau=0: {baseline['accuracy']:.4f} "
            f"({baseline['correct']}/{len(rows)})"
        )
        print(
            f"Best prior_tau={best['prior_tau']:.4f}: "
            f"{best['accuracy']:.4f} ({best['correct']}/{len(rows)}), "
            f"change={best['accuracy'] - baseline['accuracy']:+.4f}"
        )
        print(
            f"Prediction counts at best tau: "
            f"{best['prediction_counts']}\n"
        )

    if "sri_lankan" in args.datasets:
        rows = load_sinhala_details(args.details, required=False)
        if rows:
            values = threshold_grid(
                args.grid_min, args.grid_max, args.grid_step
            )
            ranked = tune_sinhala_thresholds(rows, values)
            best = ranked[0]
            default = evaluate_thresholds(rows, 0.5, 0.5)
            gold_counts = dict(
                sorted(collections.Counter(
                    row["gold"] for row in rows
                ).items())
            )
            si_report = {
                "n_items": len(rows),
                "gold_counts": gold_counts,
                "grid": {
                    "min": args.grid_min,
                    "max": args.grid_max,
                    "step": args.grid_step,
                    "values": len(values),
                    "pairs_tested": len(values) ** 2,
                },
                "th_a": best["th_a"],
                "th_b": best["th_b"],
                "accuracy": best["accuracy"],
                "correct": best["correct"],
                "prediction_counts": best["prediction_counts"],
                "default_0_5": default,
                "absolute_improvement": (
                    best["accuracy"] - default["accuracy"]
                ),
                "top_results": ranked[: args.top_k],
            }
            report["datasets"]["sri_lankan"] = si_report
            # Preserve the original top-level fields for existing consumers.
            report["th_a"] = best["th_a"]
            report["th_b"] = best["th_b"]
            report["accuracy"] = best["accuracy"]
            report["correct"] = best["correct"]
            report["prediction_counts"] = best["prediction_counts"]
            report["default_0_5"] = default
            report["absolute_improvement"] = (
                best["accuracy"] - default["accuracy"]
            )
            submission_args.extend([
                f"--th_a {best['th_a']:.6g}",
                f"--th_b {best['th_b']:.6g}",
            ])

            print(f"Sinhala held-out items: {len(rows)}")
            print(
                f"Default 0.50/0.50: {default['accuracy']:.4f} "
                f"({default['correct']}/{len(rows)})"
            )
            print(
                f"Best thresholds: th_a={best['th_a']:.4f}, "
                f"th_b={best['th_b']:.4f}"
            )
            print(
                f"Best accuracy: {best['accuracy']:.4f} "
                f"({best['correct']}/{len(rows)}), "
                f"change={best['accuracy'] - default['accuracy']:+.4f}"
            )
            print(f"Gold counts: {gold_counts}")
            print(
                f"Prediction counts at best thresholds: "
                f"{best['prediction_counts']}\n"
            )

    if not report["datasets"]:
        raise ValueError(
            "none of the selected datasets were found in --details"
        )
    report["make_submission_args"] = " ".join(submission_args)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Use for submission: {report['make_submission_args']}")
    print(f"Calibration report -> {output_path}")


if __name__ == "__main__":
    main()
