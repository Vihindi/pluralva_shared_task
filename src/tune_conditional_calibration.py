"""Sweep conditional Sinhala thresholds and report every combination.

The decision rule is:

1. Predict statement B using ``th_b``.
2. If B is predicted No, predict A using ``th_a_if_b_no``.
3. If B is predicted Yes, predict A using ``th_a_if_b_yes``.

Input rows must contain ``gold``, ``p_yes_A``, and ``p_yes_B``. The supplied
details may contain other datasets; only ``dataset == "sri_lankan"`` is used.
"""

import argparse
import csv
import json
from pathlib import Path

from tune_calibration import load_sinhala_details, threshold_grid


def compose_conditional_label(
    p_yes_a,
    p_yes_b,
    th_b,
    th_a_if_b_no,
    th_a_if_b_yes,
):
    """Compose A/B/Both/0 using a B-first conditional A threshold."""
    b_yes = p_yes_b >= th_b
    th_a = th_a_if_b_yes if b_yes else th_a_if_b_no
    a_yes = p_yes_a >= th_a
    if a_yes and b_yes:
        return "Both"
    if a_yes:
        return "A"
    if b_yes:
        return "B"
    return "0"


def evaluate_conditional(
    rows,
    th_b,
    th_a_if_b_no,
    th_a_if_b_yes,
):
    """Score one conditional threshold combination."""
    counts = {"0": 0, "A": 0, "B": 0, "Both": 0}
    correct = 0
    for row in rows:
        prediction = compose_conditional_label(
            float(row["p_yes_A"]),
            float(row["p_yes_B"]),
            th_b,
            th_a_if_b_no,
            th_a_if_b_yes,
        )
        counts[prediction] += 1
        correct += prediction == row["gold"]
    return {
        "correct": correct,
        "accuracy": correct / len(rows),
        "prediction_counts": counts,
    }


def _precompute_scores(rows, b_values, a_no_values, a_yes_values):
    """Precompute each half of the conditional rule for fast grid sweeping."""
    prepared = [
        (
            float(row["p_yes_A"]),
            float(row["p_yes_B"]),
            row["gold"],
        )
        for row in rows
    ]
    tables = []
    for th_b in b_values:
        b_no_rows = [row for row in prepared if row[1] < th_b]
        b_yes_rows = [row for row in prepared if row[1] >= th_b]

        no_table = []
        for th_a in a_no_values:
            a_count = sum(p_yes_a >= th_a for p_yes_a, _, _ in b_no_rows)
            correct = sum(
                ("A" if p_yes_a >= th_a else "0") == gold
                for p_yes_a, _, gold in b_no_rows
            )
            no_table.append((correct, a_count, len(b_no_rows) - a_count))

        yes_table = []
        for th_a in a_yes_values:
            both_count = sum(
                p_yes_a >= th_a for p_yes_a, _, _ in b_yes_rows
            )
            correct = sum(
                ("Both" if p_yes_a >= th_a else "B") == gold
                for p_yes_a, _, gold in b_yes_rows
            )
            yes_table.append(
                (correct, len(b_yes_rows) - both_count, both_count)
            )
        tables.append((no_table, yes_table))
    return tables


def sweep_conditional_thresholds(
    rows,
    b_values,
    a_no_values,
    a_yes_values,
    out_csv,
    top_k=20,
    print_all=False,
    progress_every=10000,
):
    """Evaluate the Cartesian product and stream every result to a CSV file."""
    if not rows:
        raise ValueError("rows must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total = len(b_values) * len(a_no_values) * len(a_yes_values)
    tables = _precompute_scores(
        rows, b_values, a_no_values, a_yes_values
    )
    best = []
    experiment = 0
    fields = [
        "experiment",
        "th_b",
        "th_a_if_b_no",
        "th_a_if_b_yes",
        "correct",
        "accuracy",
        "n_0",
        "n_A",
        "n_B",
        "n_Both",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for b_index, th_b in enumerate(b_values):
            no_table, yes_table = tables[b_index]
            for no_index, th_a_no in enumerate(a_no_values):
                no_correct, n_a, n_zero = no_table[no_index]
                for yes_index, th_a_yes in enumerate(a_yes_values):
                    yes_correct, n_b, n_both = yes_table[yes_index]
                    experiment += 1
                    correct = no_correct + yes_correct
                    result = {
                        "experiment": experiment,
                        "th_b": th_b,
                        "th_a_if_b_no": th_a_no,
                        "th_a_if_b_yes": th_a_yes,
                        "correct": correct,
                        "accuracy": correct / len(rows),
                        "n_0": n_zero,
                        "n_A": n_a,
                        "n_B": n_b,
                        "n_Both": n_both,
                    }
                    writer.writerow(result)
                    rank_key = (
                        -correct,
                        abs(th_b - 0.5)
                        + abs(th_a_no - 0.5)
                        + abs(th_a_yes - 0.5),
                        th_b,
                        th_a_no,
                        th_a_yes,
                    )
                    best.append((rank_key, result))
                    best.sort(key=lambda item: item[0])
                    del best[top_k:]

                    if print_all:
                        print(
                            f"EXP {experiment}/{total}: "
                            f"th_b={th_b:.6f}, "
                            f"th_a_if_b_no={th_a_no:.6f}, "
                            f"th_a_if_b_yes={th_a_yes:.6f} -> "
                            f"{correct}/{len(rows)} "
                            f"({100 * correct / len(rows):.4f}%)"
                        )
                    elif experiment % progress_every == 0:
                        print(
                            f"Progress {experiment:,}/{total:,}; "
                            f"best={-best[0][0][0]}/{len(rows)}"
                        )

    return [result for _, result in best], total


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Try every B-first conditional Sinhala threshold combination, "
            "save all scores to CSV, and report the best combinations."
        )
    )
    parser.add_argument(
        "--details",
        nargs="+",
        required=True,
        help="details JSONL files containing Sinhala probabilities and gold",
    )
    parser.add_argument("--th_b_min", type=float, default=0.85)
    parser.add_argument("--th_b_max", type=float, default=0.99)
    parser.add_argument("--th_b_step", type=float, default=0.01)
    parser.add_argument("--th_a_b_no_min", type=float, default=0.01)
    parser.add_argument("--th_a_b_no_max", type=float, default=0.50)
    parser.add_argument("--th_a_b_no_step", type=float, default=0.01)
    parser.add_argument("--th_a_b_yes_min", type=float, default=0.20)
    parser.add_argument("--th_a_b_yes_max", type=float, default=0.80)
    parser.add_argument("--th_a_b_yes_step", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--print_all",
        action="store_true",
        help="print every experiment in addition to saving it to CSV",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10000,
        help="progress interval when --print_all is not used",
    )
    parser.add_argument(
        "--out_csv",
        default="processed/si_conditional_calibration_experiments.csv",
    )
    parser.add_argument(
        "--out_json",
        default="processed/si_conditional_calibration_best.json",
    )
    args = parser.parse_args()

    rows = load_sinhala_details(args.details)
    b_values = threshold_grid(
        args.th_b_min, args.th_b_max, args.th_b_step
    )
    a_no_values = threshold_grid(
        args.th_a_b_no_min,
        args.th_a_b_no_max,
        args.th_a_b_no_step,
    )
    a_yes_values = threshold_grid(
        args.th_a_b_yes_min,
        args.th_a_b_yes_max,
        args.th_a_b_yes_step,
    )
    top_results, total = sweep_conditional_thresholds(
        rows,
        b_values,
        a_no_values,
        a_yes_values,
        args.out_csv,
        top_k=args.top_k,
        print_all=args.print_all,
        progress_every=args.progress_every,
    )
    report = {
        "method": "sinhala_b_first_conditional_threshold_grid_search",
        "details": [str(Path(path)) for path in args.details],
        "n_items": len(rows),
        "experiments": total,
        "ranges": {
            "th_b": b_values,
            "th_a_if_b_no": a_no_values,
            "th_a_if_b_yes": a_yes_values,
        },
        "best": top_results[0],
        "top_results": top_results,
        "all_results_csv": str(Path(args.out_csv)),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"\nCompleted {total:,} experiments on {len(rows)} items.")
    for index, result in enumerate(top_results, 1):
        print(
            f"TOP {index:02d}: th_b={result['th_b']:.6f}, "
            f"th_a_if_b_no={result['th_a_if_b_no']:.6f}, "
            f"th_a_if_b_yes={result['th_a_if_b_yes']:.6f} -> "
            f"{result['correct']}/{len(rows)} "
            f"({100 * result['accuracy']:.4f}%)"
        )
    print(f"All experiments -> {Path(args.out_csv)}")
    print(f"Best-results report -> {out_json}")


if __name__ == "__main__":
    main()
