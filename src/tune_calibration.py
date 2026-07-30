"""Tune separate Sinhala binary-decision thresholds on held-out predictions.

Input files are JSONL details produced by ``src/evaluate.py eval``.  Each
Sinhala row must contain:

    uid, dataset="sri_lankan", gold, p_yes_A, p_yes_B

Example:

    python src/tune_calibration.py \
        --details processed/si_eval_fold0.jsonl \
                  processed/si_eval_fold1.jsonl \
                  processed/si_eval_fold2.jsonl \
                  processed/si_eval_fold3.jsonl \
                  processed/si_eval_fold4.jsonl \
        --out processed/si_calibration.json

The output ``th_a`` and ``th_b`` values can be passed directly to
``src/make_submission.py``.
"""

import argparse
import collections
import json
from pathlib import Path


VALID_LABELS = {"A", "B", "Both", "0"}


def load_sinhala_details(paths):
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
                if row["gold"] not in VALID_LABELS:
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

    if not rows:
        raise ValueError("no Sinhala rows found in the supplied details files")
    return rows


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
            "Tune separate statement-A and statement-B Yes thresholds for "
            "binary Sinhala predictions."
        )
    )
    parser.add_argument(
        "--details",
        nargs="+",
        required=True,
        help="held-out evaluate.py details JSONL files (normally five CV folds)",
    )
    parser.add_argument("--grid_min", type=float, default=0.20)
    parser.add_argument("--grid_max", type=float, default=0.80)
    parser.add_argument("--grid_step", type=float, default=0.01)
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="number of highest-ranked threshold pairs to print and save",
    )
    parser.add_argument(
        "--out",
        default="processed/si_calibration.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")

    rows = load_sinhala_details(args.details)
    values = threshold_grid(args.grid_min, args.grid_max, args.grid_step)
    ranked = tune_sinhala_thresholds(rows, values)
    best = ranked[0]
    default = evaluate_thresholds(rows, 0.5, 0.5)
    gold_counts = dict(
        sorted(collections.Counter(row["gold"] for row in rows).items())
    )

    report = {
        "method": "sinhala_binary_threshold_grid_search",
        "details": [str(Path(path)) for path in args.details],
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
        "absolute_improvement": best["accuracy"] - default["accuracy"],
        "top_results": ranked[: args.top_k],
        "make_submission_args": (
            f"--th_a {best['th_a']:.6g} --th_b {best['th_b']:.6g}"
        ),
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

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
    print(f"Prediction counts at best thresholds: {best['prediction_counts']}")
    print(
        "Use for submission: "
        f"--th_a {best['th_a']:.6g} --th_b {best['th_b']:.6g}"
    )
    print(f"Calibration report -> {output_path}")


if __name__ == "__main__":
    main()
