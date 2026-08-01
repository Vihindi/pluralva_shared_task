"""Measure final-answer accuracy on one or more built SFT JSONL files.

This is a post-training evaluation: it loads a saved LoRA adapter, predicts the
final label for every SFT row, and compares it with the final ``Answer: X`` in
the row's assistant target. The default constrained mode scores every legal
answer continuation and selects the most probable one. ``--generate`` instead
uses greedy free generation and counts missing/invalid answers as incorrect.

Predictions are appended after every row, so rerunning the same command resumes
without recomputing completed rows.
"""

import argparse
import collections
import json
import re
from pathlib import Path

from evaluate import Scorer, parse_generated_answer, softmax


ANSWER_RE = re.compile(
    r"(?:^|\n)Answer:\s*(?P<label>[^\r\n]*?\S)\s*\Z",
    flags=re.IGNORECASE,
)
MCQ_LABELS = ("A", "B", "C", "D")
SI_BINARY_LABELS = ("Yes", "No")
SI_FOUR_WAY_LABELS = ("A", "B", "Both", "0")


def parse_target(text):
    match = ANSWER_RE.search(text)
    if match is None:
        raise ValueError(
            "assistant target must end with a line such as 'Answer: Yes'"
        )
    return match.group("label")


def allowed_labels(dataset, target):
    if dataset in {"chinese", "indonesian"}:
        allowed = MCQ_LABELS
    elif dataset == "sri_lankan" and target in SI_BINARY_LABELS:
        allowed = SI_BINARY_LABELS
    elif dataset == "sri_lankan" and target in SI_FOUR_WAY_LABELS:
        allowed = SI_FOUR_WAY_LABELS
    else:
        raise ValueError(
            f"unsupported dataset/target combination: {dataset!r}/{target!r}"
        )
    if target not in allowed:
        raise ValueError(
            f"target {target!r} is invalid for dataset {dataset!r}"
        )
    return allowed


def load_sft_rows(paths, selected_datasets=None, max_rows=None):
    rows = []
    selected = set(selected_datasets) if selected_datasets else None
    for file_index, path_text in enumerate(paths):
        path = Path(path_text)
        with open(path, encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                dataset = row.get("dataset")
                if selected is not None and dataset not in selected:
                    continue
                messages = row.get("messages")
                if not messages or messages[-1].get("role") != "assistant":
                    raise ValueError(
                        f"{path}:{line_number}: missing final assistant target"
                    )
                target = parse_target(messages[-1]["content"])
                allowed = allowed_labels(dataset, target)
                rows.append({
                    "row_key": f"{file_index}:{line_number}",
                    "source_file": str(path),
                    "source_line": line_number,
                    "uid": row.get("uid"),
                    "dataset": dataset,
                    "messages": messages[:-1],
                    "target": target,
                    "allowed": allowed,
                })
                if max_rows is not None and len(rows) >= max_rows:
                    return rows
    return rows


def load_completed(path, expected_rows):
    if not path.exists():
        return {}
    expected = {row["row_key"]: row for row in expected_rows}
    completed = {}
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            result = json.loads(line)
            key = result.get("row_key")
            if key in completed:
                raise ValueError(f"{path}:{line_number}: duplicate row_key {key}")
            if key not in expected:
                raise ValueError(
                    f"{path}:{line_number}: row_key {key!r} is not present in "
                    "the current --train_files; use a new --out path"
                )
            source = expected[key]
            for field in ("uid", "dataset", "target"):
                if result.get(field) != source[field]:
                    raise ValueError(
                        f"{path}:{line_number}: saved {field} does not match "
                        "the current SFT row; use a new --out path"
                    )
            completed[key] = result
    return completed


def predict_row(scorer, row, generate, max_new_tokens):
    if generate:
        generated = scorer.generate_text(
            row["messages"],
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
            temperature=0.0,
        )
        prediction = parse_generated_answer(generated, row["allowed"])
        return prediction, {"generated_text": generated}

    candidates = [f"Answer: {label}" for label in row["allowed"]]
    logprobs = scorer.score_candidates(row["messages"], candidates)
    probabilities = softmax(logprobs)
    best_index = max(range(len(logprobs)), key=logprobs.__getitem__)
    prediction = row["allowed"][best_index]
    details = {
        "probabilities": {
            label: probability
            for label, probability in zip(row["allowed"], probabilities)
        },
        "logprobs": {
            label: logprob
            for label, logprob in zip(row["allowed"], logprobs)
        },
    }
    return prediction, details


def calculate_summary(results):
    totals = collections.Counter()
    correct = collections.Counter()
    invalid = collections.Counter()
    confusion = collections.defaultdict(collections.Counter)
    for row in results:
        dataset = row["dataset"]
        totals[dataset] += 1
        totals["overall"] += 1
        if row["prediction"] is None:
            invalid[dataset] += 1
            invalid["overall"] += 1
        if row["correct"]:
            correct[dataset] += 1
            correct["overall"] += 1
        confusion[dataset][f"{row['target']}->{row['prediction']}"] += 1

    ordered = [
        dataset for dataset in ("chinese", "indonesian", "sri_lankan")
        if totals[dataset]
    ] + ["overall"]
    return {
        "accuracy": {
            dataset: {
                "correct": correct[dataset],
                "total": totals[dataset],
                "accuracy": correct[dataset] / totals[dataset],
                "invalid_predictions": invalid[dataset],
            }
            for dataset in ordered
        },
        "confusion": {
            dataset: dict(sorted(confusion[dataset].items()))
            for dataset in ordered if dataset != "overall"
        },
    }


def print_summary(summary):
    print("\nFinal-answer training accuracy")
    print("--------------------------------")
    for dataset, values in summary["accuracy"].items():
        print(
            f"{dataset:12s}: {values['correct']:5d}/{values['total']:<5d} "
            f"= {100 * values['accuracy']:.2f}%"
            + (
                f" | invalid={values['invalid_predictions']}"
                if values["invalid_predictions"] else ""
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train_files", nargs="+", required=True)
    parser.add_argument(
        "--out", default="training_accuracy_predictions.jsonl",
        help="resumable per-row prediction output",
    )
    parser.add_argument(
        "--summary_out", default=None,
        help="summary JSON path (default: <out>.summary.json)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["chinese", "indonesian", "sri_lankan"],
        default=None,
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="use greedy generation instead of constrained label scoring",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_system", action="store_true")
    args = parser.parse_args()

    if args.load_4bit and args.load_8bit:
        raise ValueError("choose only one of --load_4bit or --load_8bit")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max_rows must be positive")

    rows = load_sft_rows(args.train_files, args.datasets, args.max_rows)
    if not rows:
        raise SystemExit("no matching SFT rows found")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path, rows)
    print(
        f"loaded {len(rows)} SFT rows; {len(completed)} already evaluated; "
        f"mode={'generation' if args.generate else 'constrained scoring'}"
    )

    scorer = Scorer(
        args.model,
        adapter=args.adapter,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        trust_remote_code=args.trust_remote_code,
        merge_system=args.no_system,
    )

    pending = [row for row in rows if row["row_key"] not in completed]
    with open(output_path, "a", encoding="utf-8") as output:
        for index, row in enumerate(pending, 1):
            prediction, details = predict_row(
                scorer, row, args.generate, args.max_new_tokens
            )
            result = {
                "row_key": row["row_key"],
                "source_file": row["source_file"],
                "source_line": row["source_line"],
                "uid": row["uid"],
                "dataset": row["dataset"],
                "target": row["target"],
                "prediction": prediction,
                "correct": prediction == row["target"],
                "mode": "generation" if args.generate else "constrained",
                **details,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            completed[row["row_key"]] = result
            if index % 25 == 0 or index == len(pending):
                partial = calculate_summary(completed.values())
                overall = partial["accuracy"]["overall"]
                print(
                    f"evaluated {len(completed)}/{len(rows)} | "
                    f"running accuracy={100 * overall['accuracy']:.2f}%"
                )

    ordered_results = [completed[row["row_key"]] for row in rows]
    summary = calculate_summary(ordered_results)
    summary.update({
        "model": args.model,
        "adapter": args.adapter,
        "train_files": args.train_files,
        "mode": "generation" if args.generate else "constrained",
        "rows": len(rows),
    })
    summary_path = Path(args.summary_out or f"{args.out}.summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print_summary(summary)
    print(f"row predictions -> {output_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
