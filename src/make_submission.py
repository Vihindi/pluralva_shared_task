"""Predict the hidden test set and package the official submission.

Produces submission/predictions.jsonl + submission/predictions.zip in the
required format: {"dataset": ..., "id": ..., "LLM_Output": ...} per line,
LLM_Output in {A,B,C,D} for chinese/indonesian and {A,B,Both,0} for sri_lankan.

Four ways to use it:

1) Predict with one model/adapter (GPU; resumable — reruns skip finished items):
   python src/make_submission.py predict \
       --model meta-llama/Llama-3.1-8B-Instruct --adapter runs/joint_full \
       --n_perms 4 --th_a 0.5 --th_b 0.5 --load_4bit

   Raw probabilities are checkpointed to submission/test_details.jsonl after
   every item, so a Colab disconnect costs nothing: rerun the same command and
   it resumes. Sinhala calibration (--th_a/--th_b, from tune_calibration.py)
   is applied at composition time from the saved raw probabilities.

2) Predict with three routed country adapters (one shared base-model load):
   python src/make_submission.py predict \
       --model Qwen/Qwen3.5-4B --country_specific \
       --zh_adapter runs/country_lora/qwen35_4b_full/chinese \
       --id_adapter runs/country_lora/qwen35_4b_full/indonesian \
       --si_adapter runs/country_lora/qwen35_4b_full/sri_lankan \
       --n_perms 4 --load_4bit

   Supplying only one named adapter predicts only that country. This can also
   be made explicit with --countries:
   python src/make_submission.py predict --model MODEL --country_specific \
       --si_adapter runs/sri_lankan --countries sri_lankan --load_4bit

3) Compose from existing details (no GPU) — e.g. the 5 fold-adapter runs made
   with evaluate.py predict (their <out>.details.jsonl files), fold-ensembled:
   python src/make_submission.py compose \
       --details pred_fold0.jsonl.details.jsonl ... pred_fold4.jsonl.details.jsonl \
       --th_b 0.92 --th_a_if_b_no 0.16 --th_a_if_b_yes 0.50

4) Package an already-made predictions.jsonl (validate + zip only):
   python src/make_submission.py package --predictions predictions.jsonl

Every path ends with a strict validation: exact coverage of the selected test IDs
(nothing missing, nothing extra, no duplicates), legal labels, exact key names.
The zip contains predictions.jsonl at its root, as Codabench expects.
"""
import argparse
import collections
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTERS4 = ["A", "B", "C", "D"]
VALID = {"chinese": set(LETTERS4), "indonesian": set(LETTERS4),
         "sri_lankan": {"A", "B", "Both", "0"}}
# submission "dataset" field values — change here if organizers specify others
DATASET_NAMES = {"chinese": "chinese", "indonesian": "indonesian",
                 "sri_lankan": "sri_lankan"}
TEST_FILES = {"chinese": "chinese_test_without_gold.jsonl",
              "indonesian": "indonesian_test_without_gold.jsonl",
              "sri_lankan": "sri_lankan_test_without_gold.jsonl"}
COUNTRIES = tuple(TEST_FILES)
ADAPTER_ATTRS = {
    "chinese": "zh_adapter",
    "indonesian": "id_adapter",
    "sri_lankan": "si_adapter",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_ids(test_dir, datasets=None):
    """dataset -> ordered list of test IDs (the coverage ground truth)."""
    out = {}
    for ds in datasets or COUNTRIES:
        fname = TEST_FILES[ds]
        p = Path(test_dir) / fname
        if not p.exists():
            raise SystemExit(f"missing test file: {p}")
        out[ds] = [x["ID"] for x in load_jsonl(p)]
    return out


def dev_priors(processed_dir):
    priors = {}
    for ds in ("chinese", "indonesian"):
        p = Path(processed_dir) / f"{ds}.jsonl"
        if p.exists():
            c = collections.Counter(r["gold"] for r in load_jsonl(p))
            n = sum(c.values())
            priors[ds] = {k: v / n for k, v in c.items()}
    return priors


def decide(row, priors, args):
    """Averaged/raw details row -> final label, applying calibration once."""
    # binary SI stores p_yes_A/p_yes_B; 4-way SI and all MCQ store a probs dict
    if row["dataset"] == "sri_lankan" and "p_yes_A" in row:
        b = row["p_yes_B"] >= args.th_b
        th_a_if_b_no = getattr(args, "th_a_if_b_no", None)
        th_a_if_b_yes = getattr(args, "th_a_if_b_yes", None)
        if th_a_if_b_no is not None and th_a_if_b_yes is not None:
            th_a = th_a_if_b_yes if b else th_a_if_b_no
        else:
            th_a = args.th_a
        a = row["p_yes_A"] >= th_a
        return "Both" if (a and b) else "A" if a else "B" if b else "0"
    probs = dict(row["probs"])
    prior = priors.get(row["dataset"])
    if prior and args.prior_tau > 0:
        probs = {k: v * (prior.get(k, 1e-9) ** args.prior_tau)
                 for k, v in probs.items()}
    return max(probs, key=probs.get)


def average_details(paths):
    """uid -> averaged row across detail files (fold-ensemble)."""
    acc = {}
    for p in paths:
        for row in load_jsonl(p):
            s = acc.setdefault(row["uid"], {"dataset": row["dataset"],
                                            "probs": collections.defaultdict(float),
                                            "p_yes_A": 0.0, "p_yes_B": 0.0,
                                            "n_mcq": 0, "n_si": 0})
            if "probs" in row:
                for k, v in row["probs"].items():
                    s["probs"][k] += v
                s["n_mcq"] += 1
            if "p_yes_A" in row:
                s["p_yes_A"] += row["p_yes_A"]
                s["p_yes_B"] += row["p_yes_B"]
                s["n_si"] += 1
    merged = {}
    for uid, s in acc.items():
        row = {"uid": uid, "dataset": s["dataset"]}
        if s["n_mcq"]:
            row["probs"] = {k: v / s["n_mcq"] for k, v in s["probs"].items()}
        if s["n_si"]:
            row["p_yes_A"] = s["p_yes_A"] / s["n_si"]
            row["p_yes_B"] = s["p_yes_B"] / s["n_si"]
        merged[uid] = row
    return merged


def run_predict(args):
    """GPU path: score every test item, checkpointing raw probs per item."""
    from evaluate import (Scorer, score_mcq, score_si, score_si_4way,
                          load_test_records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "test_details.jsonl"
    done = set()
    if details_path.exists():
        done = {r["uid"] for r in load_jsonl(details_path)}
        print(f"resuming: {len(done)} items already scored")

    country_adapters = None
    if args.country_specific:
        country_adapters = {
            ds: getattr(args, ADAPTER_ATTRS[ds])
            for ds in args.selected_datasets
        }
    scorer = Scorer(
        args.model,
        adapter=args.adapter,
        adapters=country_adapters,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )
    with open(details_path, "a", encoding="utf-8") as details:
        for ds in args.selected_datasets:
            fname = TEST_FILES[ds]
            if args.country_specific:
                scorer.set_adapter(ds)
                print(f"[{ds}] active adapter: {country_adapters[ds]}")
            recs = load_test_records(Path(args.test_dir) / fname, ds)
            todo = [r for r in recs if r["uid"] not in done]
            print(f"[{ds}] {len(todo)}/{len(recs)} items to score")
            for i, rec in enumerate(todo):
                if ds == "sri_lankan" and args.si_mode == "4way":
                    # raw probs over {A,B,Both,0}; calibration at composition
                    probs = score_si_4way(scorer, rec,
                                          value_summaries=args.value_summaries)
                    row = {"uid": rec["uid"], "dataset": ds, "probs": probs}
                elif ds == "sri_lankan":
                    _, pa, pb = score_si(scorer, rec,
                                         value_summaries=args.value_summaries,
                                         si_negation_prompt=getattr(
                                             args,
                                             "enable_si_negation_prompt",
                                             False))
                    row = {"uid": rec["uid"], "dataset": ds,
                           "p_yes_A": pa, "p_yes_B": pb}
                else:
                    # raw probs only (tau=0); calibration happens at composition
                    probs = score_mcq(scorer, rec, n_perms=args.n_perms,
                                      value_summaries=args.value_summaries)
                    row = {"uid": rec["uid"], "dataset": ds, "probs": probs}
                details.write(json.dumps(row, ensure_ascii=False) + "\n")
                details.flush()
                if (i + 1) % 50 == 0:
                    print(f"  [{ds}] {i+1}/{len(todo)}")
    return average_details([details_path])


def validate(pred_rows, ids_by_ds):
    errors = []
    seen = collections.Counter()
    expected = {uid: ds for ds, uids in ids_by_ds.items() for uid in uids}
    name_to_ds = {v: k for k, v in DATASET_NAMES.items()}
    for r in pred_rows:
        if set(r.keys()) != {"dataset", "id", "LLM_Output"}:
            errors.append(f"bad keys {sorted(r.keys())} for id={r.get('id')}")
            continue
        ds = name_to_ds.get(r["dataset"])
        if ds is None:
            errors.append(f"unknown dataset {r['dataset']!r} for id={r['id']}")
            continue
        if r["id"] not in expected:
            errors.append(f"id {r['id']} not in the test set")
        elif expected[r["id"]] != ds:
            errors.append(f"id {r['id']} labeled dataset {r['dataset']!r}")
        if r["LLM_Output"] not in VALID[ds]:
            errors.append(f"illegal LLM_Output {r['LLM_Output']!r} for {r['id']}")
        seen[r["id"]] += 1
    dupes = [u for u, c in seen.items() if c > 1]
    missing = [u for u in expected if u not in seen]
    if dupes:
        errors.append(f"{len(dupes)} duplicate ids (e.g. {dupes[:3]})")
    if missing:
        errors.append(f"{len(missing)} missing ids (e.g. {missing[:5]})")
    return errors


def write_and_zip(pred_rows, out_dir, ids_by_ds):
    errors = validate(pred_rows, ids_by_ds)
    if errors:
        for e in errors[:20]:
            print("FORMAT ERROR:", e)
        raise SystemExit(f"{len(errors)} validation errors — not packaging.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "predictions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in pred_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    zip_path = out_dir / "predictions.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl_path, arcname="predictions.jsonl")  # file at zip root
    n_by_ds = collections.Counter(r["dataset"] for r in pred_rows)
    label_dist = collections.defaultdict(collections.Counter)
    for r in pred_rows:
        label_dist[r["dataset"]][r["LLM_Output"]] += 1
    print(f"\nVALID: {len(pred_rows)} predictions  {dict(n_by_ds)}")
    for ds, c in label_dist.items():
        print(f"  {ds} label dist: {dict(c.most_common())}")
    print(f"submission -> {jsonl_path}\nzipped     -> {zip_path}")


def compose_rows(merged, ids_by_ds, priors, args):
    rows = []
    for ds, uids in ids_by_ds.items():
        for uid in uids:  # keep official test-file order
            if uid not in merged:
                raise SystemExit(f"no prediction data for {uid} — "
                                 f"prediction run incomplete? rerun predict.")
            rows.append({"dataset": DATASET_NAMES[ds], "id": uid,
                         "LLM_Output": decide(merged[uid], priors, args)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["predict", "compose", "package"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--country_specific", action="store_true",
                    help="predict with routed country adapters; when "
                         "--countries is omitted, the supplied --zh_adapter/"
                         "--id_adapter/--si_adapter flags determine which "
                         "countries are processed")
    ap.add_argument("--zh_adapter", default=None,
                    help="Chinese adapter path for --country_specific")
    ap.add_argument("--id_adapter", default=None,
                    help="Indonesian adapter path for --country_specific")
    ap.add_argument("--si_adapter", default=None,
                    help="Sri Lankan binary or four-way adapter path for "
                         "--country_specific; select its target format with "
                         "--si_mode")
    ap.add_argument(
        "--countries",
        nargs="+",
        choices=COUNTRIES,
        default=None,
        help="process only these countries. In country-specific mode this is "
             "optional: supplying only --si_adapter, for example, selects only "
             "sri_lankan automatically",
    )
    ap.add_argument("--test_dir", default=str(ROOT / "PlurVA-LLM_Test_Set"))
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--out_dir", default=str(ROOT / "submission"))
    ap.add_argument("--n_perms", type=int, default=4)
    ap.add_argument("--prior_tau", type=float, default=0.0,
                    help="optional shared Chinese/Indonesian prior strength "
                         "(0 = off; not tuned by tune_calibration.py)")
    ap.add_argument("--th_a", type=float, default=0.5,
                    help="binary SI only: global statement-A Yes cutoff; "
                         "ignored when both conditional A cutoffs are supplied")
    ap.add_argument("--th_b", type=float, default=0.5,
                    help="binary SI only: statement-B Yes cutoff")
    ap.add_argument(
        "--th_a_if_b_no",
        type=float,
        default=None,
        help="binary SI only: conditional A cutoff when B is predicted No",
    )
    ap.add_argument(
        "--th_a_if_b_yes",
        type=float,
        default=None,
        help="binary SI only: conditional A cutoff when B is predicted Yes",
    )
    ap.add_argument("--si_mode", choices=["binary", "4way"], default="binary",
                    help="Sri Lankan scoring mode; must match the adapter's "
                         "training --si_mode. '4way' ignores --th_a/--th_b.")
    ap.add_argument(
        "--enable_si_negation_prompt",
        action="store_true",
        help="binary Sinhala only: use the specialized negative-question "
             "instruction. Enable only when the SFT data was built with the "
             "same flag.",
    )
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--details", nargs="+", default=[],
                    help="compose mode: details files to (fold-)ensemble")
    ap.add_argument("--predictions", default=None,
                    help="package mode: existing predictions.jsonl")
    ap.add_argument("--value_summaries", default=None,
                    help="path to a single custom summaries json applied to ALL "
                         "datasets (default: the per-country files "
                         "zh/id/si_value_summaries.json at the repo root are used "
                         "AUTOMATICALLY — this flag overrides them with one file)")
    ap.add_argument("--no_value_summaries", action="store_true",
                    help="disable value-context injection entirely for all three "
                         "countries (use when predicting with an adapter trained "
                         "BEFORE this default was introduced, to avoid a prompt "
                         "mismatch)")
    args = ap.parse_args()
    if args.load_4bit and args.load_8bit:
        raise ValueError("Cannot use both --load_4bit and --load_8bit")
    conditional_a = (
        args.th_a_if_b_no is not None,
        args.th_a_if_b_yes is not None,
    )
    if conditional_a[0] != conditional_a[1]:
        raise ValueError(
            "--th_a_if_b_no and --th_a_if_b_yes must be supplied together"
        )
    for name in ("th_a", "th_b", "th_a_if_b_no", "th_a_if_b_yes"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be between 0 and 1")
    adapter_flags = {
        "chinese": ("--zh_adapter", args.zh_adapter),
        "indonesian": ("--id_adapter", args.id_adapter),
        "sri_lankan": ("--si_adapter", args.si_adapter),
    }
    if args.country_specific:
        if args.mode != "predict":
            raise ValueError("--country_specific is only valid in predict mode")
        if args.adapter:
            raise ValueError(
                "--adapter and --country_specific are mutually exclusive")
        supplied = [ds for ds, (_, path) in adapter_flags.items() if path]
        if args.countries:
            args.selected_datasets = list(dict.fromkeys(args.countries))
            unused = [
                flag for ds, (flag, path) in adapter_flags.items()
                if path and ds not in args.selected_datasets
            ]
            if unused:
                raise ValueError(
                    "adapter supplied for a country not selected by "
                    f"--countries: {', '.join(unused)}")
        else:
            args.selected_datasets = supplied
        if not args.selected_datasets:
            raise ValueError(
                "--country_specific requires at least one of --zh_adapter, "
                "--id_adapter, or --si_adapter")
        missing = [
            adapter_flags[ds][0] for ds in args.selected_datasets
            if not adapter_flags[ds][1]
        ]
        if missing:
            raise ValueError(
                "selected countries require " + ", ".join(missing))
        nonexistent = [
            f"{adapter_flags[ds][0]}={adapter_flags[ds][1]}"
            for ds in args.selected_datasets
            if not Path(adapter_flags[ds][1]).exists()
        ]
        if nonexistent:
            raise FileNotFoundError(
                "country adapter path(s) not found: " + ", ".join(nonexistent))
    elif any(path for _, path in adapter_flags.values()):
        raise ValueError(
            "--zh_adapter/--id_adapter/--si_adapter require --country_specific")
    else:
        args.selected_datasets = (
            list(dict.fromkeys(args.countries))
            if args.countries else list(COUNTRIES)
        )
    if args.no_value_summaries:
        args.value_summaries = None
        print("value-context injection: DISABLED (--no_value_summaries)")
    elif args.value_summaries:
        from prompts import load_value_summaries
        custom_path = args.value_summaries
        args.value_summaries = load_value_summaries(custom_path)
        print(f"loaded {len(args.value_summaries)} value summaries from "
              f"{custom_path!r} (applied to all datasets whose keys match)")
    else:
        args.value_summaries = "auto"  # -> per-country files at repo root
        print("value-context injection: AUTO (zh/id/si_value_summaries.json "
              "at repo root, per country)")

    ids_by_ds = test_ids(args.test_dir, args.selected_datasets)
    total = sum(len(v) for v in ids_by_ds.values())
    print(f"test set: {total} items "
          f"({', '.join(f'{k}={len(v)}' for k, v in ids_by_ds.items())})")

    if args.mode == "package":
        assert args.predictions, "package mode needs --predictions"
        write_and_zip(load_jsonl(args.predictions), args.out_dir, ids_by_ds)
        return

    priors = dev_priors(args.processed_dir)
    if args.mode == "predict":
        merged = run_predict(args)
    else:
        assert args.details, "compose mode needs --details"
        merged = average_details(args.details)
    rows = compose_rows(merged, ids_by_ds, priors, args)
    write_and_zip(rows, args.out_dir, ids_by_ds)


if __name__ == "__main__":
    main()
