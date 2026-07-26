"""Zero-shot test-set prediction via the Gemini API (experiment / ablation).

Reuses the same chat prompts as the rest of the pipeline (prompts.build_messages)
and packages via make_submission.write_and_zip. Not intended as an official
Track-1 submission system — closed APIs may be disallowed for that track.

Defaults: Sri Lankan binary Yes/No decomposition, no value-summary injection,
one MCQ call per item (no option-permutation ensemble).

  export GEMINI_API_KEY=...   # or Colab userdata
  python src/predict_gemini_zeroshot.py --limit 5 --datasets sri_lankan  # smoke
  python src/predict_gemini_zeroshot.py --out_dir submission_gemini
  python src/make_submission.py package \
      --predictions submission_gemini/predictions.jsonl \
      --out_dir submission_gemini

Resumable: finished uids in <out_dir>/test_details.jsonl are skipped on rerun.
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from make_submission import (DATASET_NAMES, TEST_FILES, load_jsonl, test_ids,
                             write_and_zip)
from preprocess import (preprocess_chinese, preprocess_indonesian,
                        preprocess_sri_lankan)
from prompts import build_messages

ROOT = Path(__file__).resolve().parent.parent
API_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "{model}:generateContent")
LETTERS4 = ["A", "B", "C", "D"]
SI4_LABELS = ["A", "B", "Both", "0"]
TEST_PREPROCESSORS = {"chinese": preprocess_chinese,
                      "indonesian": preprocess_indonesian,
                      "sri_lankan": preprocess_sri_lankan}
_ANSWER_RE = re.compile(r"[Aa]nswer\s*[:：]\s*\**\s*([ABCD]|Yes|No|Both|0)\b")


def parse_generated_answer(text, allowed):
    """Same contract as evaluate.parse_generated_answer (no torch import)."""
    hits = [m.group(1) for m in _ANSWER_RE.finditer(text)
            if m.group(1) in allowed]
    if hits:
        return hits[-1]
    esc = "|".join(re.escape(a) for a in allowed)
    bare = re.findall(rf"(?<![A-Za-z0-9])({esc})(?![A-Za-z0-9])", text)
    return bare[-1] if bare else None


def load_test_records(path, ds):
    """Parse a test JSONL (Gold_Answer may be absent); same as evaluate.py."""
    raw = load_jsonl(Path(path))
    for x in raw:
        x.setdefault("Gold_Answer", "A" if ds != "indonesian" else "A, A, A, A, A")
        if not str(x["Gold_Answer"]).strip():
            x["Gold_Answer"] = "A" if ds != "indonesian" else "A, A, A, A, A"
    recs, problems = TEST_PREPROCESSORS[ds](raw)
    for uid, why in problems:
        print(f"  WARNING test item {uid}: {why} (will still be predicted if possible)")
    for r in recs:
        r["fold"] = -1
    return recs


def messages_to_gemini(messages):
    """Split chat messages into Gemini systemInstruction + user contents."""
    system_parts, user_parts = [], []
    for m in messages:
        role, text = m["role"], m["content"]
        if role == "system":
            system_parts.append(text)
        else:
            user_parts.append(text)
    body = {
        "contents": [{"role": "user",
                      "parts": [{"text": "\n\n".join(user_parts)}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
    }
    if system_parts:
        body["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]}
    return body


def call_gemini(api_key, model, messages, timeout=60, retries=5):
    """One generation call; returns raw assistant text."""
    body = messages_to_gemini(messages)
    data = json.dumps(body).encode("utf-8")
    url = API_URL.format(model=model)
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "X-goog-api-key": api_key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            parts = payload["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            break
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
    raise RuntimeError(f"Gemini call failed: {last_err}")


def ask(api_key, model, messages, allowed, sleep):
    text = call_gemini(api_key, model, messages)
    if sleep > 0:
        time.sleep(sleep)
    ans = parse_generated_answer(text, allowed)
    return ans, text


def predict_mcq(api_key, model, rec, value_summaries, sleep):
    messages = build_messages(rec, mode="direct",
                              value_summaries=value_summaries)
    ans, text = ask(api_key, model, messages, LETTERS4, sleep)
    pred = ans if ans in LETTERS4 else "A"
    return pred, {"raw": text, "parsed": ans, "pred": pred}


def predict_si_binary(api_key, model, rec, value_summaries, sleep):
    votes = {}
    details = {}
    for stmt in ("A", "B"):
        messages = build_messages(rec, mode="direct", si_statement=stmt,
                                  value_summaries=value_summaries)
        ans, text = ask(api_key, model, messages, ["Yes", "No"], sleep)
        yes = (ans == "Yes")
        votes[stmt] = yes
        details[stmt] = {"raw": text, "parsed": ans, "yes": yes}
    a, b = votes["A"], votes["B"]
    pred = "Both" if (a and b) else "A" if a else "B" if b else "0"
    return pred, {"p_yes_A": float(a), "p_yes_B": float(b),
                  "pred": pred, "stmts": details}


def predict_si_4way(api_key, model, rec, value_summaries, sleep):
    messages = build_messages(rec, mode="direct", si_statement=None,
                              value_summaries=value_summaries)
    ans, text = ask(api_key, model, messages, SI4_LABELS, sleep)
    pred = ans if ans in SI4_LABELS else "A"
    return pred, {"raw": text, "parsed": ans, "pred": pred}


def compose_predictions(details_rows, ids_by_ds):
    by_uid = {r["uid"]: r for r in details_rows}
    rows = []
    for ds, uids in ids_by_ds.items():
        for uid in uids:
            if uid not in by_uid:
                raise SystemExit(
                    f"missing prediction for {uid} — rerun until complete, "
                    f"or pass --datasets to only package a subset (package "
                    f"mode still requires full coverage).")
            rows.append({"dataset": DATASET_NAMES[ds], "id": uid,
                         "LLM_Output": by_uid[uid]["pred"]})
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Gemini zero-shot test prediction (experiment only)")
    ap.add_argument("--api_key", default=os.environ.get("GEMINI_API_KEY"),
                    help="defaults to $GEMINI_API_KEY (do not hardcode)")
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="Gemini model id (default: gemini-2.5-flash)")
    ap.add_argument("--test_dir", default=str(ROOT / "PlurVA-LLM_Test_Set"))
    ap.add_argument("--out_dir", default=str(ROOT / "submission_gemini"))
    ap.add_argument("--datasets", nargs="+",
                    default=["chinese", "indonesian", "sri_lankan"],
                    choices=["chinese", "indonesian", "sri_lankan"])
    ap.add_argument("--si_mode", choices=["binary", "4way"], default="binary")
    ap.add_argument("--value_summaries", default=None,
                    help="'auto' or path to a summaries json; default OFF")
    ap.add_argument("--no_value_summaries", action="store_true",
                    help="explicitly disable value-context (this is the default)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: cap items per dataset (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="pause between API calls (rate limits)")
    ap.add_argument("--package", action="store_true",
                    help="after predicting, also validate+zip via write_and_zip "
                         "(requires full test coverage — omit --limit / "
                         "partial --datasets)")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("no API key: set GEMINI_API_KEY or pass --api_key")

    if args.no_value_summaries or not args.value_summaries:
        value_summaries = None
        print("value-context injection: DISABLED")
    elif args.value_summaries == "auto":
        value_summaries = "auto"
        print("value-context injection: AUTO")
    else:
        from prompts import load_value_summaries
        value_summaries = load_value_summaries(args.value_summaries)
        print(f"value-context injection: custom ({len(value_summaries)} keys)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "test_details.jsonl"

    done = {}
    if details_path.exists():
        for r in load_jsonl(details_path):
            done[r["uid"]] = r
        print(f"resuming: {len(done)} items already scored")

    print(f"model={args.model}  si_mode={args.si_mode}  "
          f"datasets={args.datasets}")

    with open(details_path, "a", encoding="utf-8") as fout:
        for ds in args.datasets:
            fname = TEST_FILES[ds]
            recs = load_test_records(Path(args.test_dir) / fname, ds)
            if args.limit:
                recs = recs[: args.limit]
            todo = [r for r in recs if r["uid"] not in done]
            print(f"[{ds}] {len(todo)}/{len(recs)} items to score "
                  f"({len(recs) - len(todo)} already done)")
            for i, rec in enumerate(todo):
                try:
                    if ds == "sri_lankan" and args.si_mode == "4way":
                        pred, detail = predict_si_4way(
                            args.api_key, args.model, rec, value_summaries,
                            args.sleep)
                    elif ds == "sri_lankan":
                        pred, detail = predict_si_binary(
                            args.api_key, args.model, rec, value_summaries,
                            args.sleep)
                    else:
                        pred, detail = predict_mcq(
                            args.api_key, args.model, rec, value_summaries,
                            args.sleep)
                except Exception as e:
                    print(f"  ERROR {rec['uid']}: {e}")
                    raise
                row = {"uid": rec["uid"], "dataset": ds, "pred": pred,
                       "detail": detail}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                done[rec["uid"]] = row
                if (i + 1) % 25 == 0 or (i + 1) == len(todo):
                    print(f"  [{ds}] {i + 1}/{len(todo)}")

    # predictions.jsonl covering whatever we have scored (may be partial)
    pred_path = out_dir / "predictions.jsonl"
    scored = [done[u] for u in done]
    # keep stable order: walk datasets/test order when available
    ids_by_ds = test_ids(args.test_dir)
    ordered = []
    seen = set()
    for ds, uids in ids_by_ds.items():
        for uid in uids:
            if uid in done:
                ordered.append(done[uid])
                seen.add(uid)
    for uid, row in done.items():
        if uid not in seen:
            ordered.append(row)

    with open(pred_path, "w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(
                {"dataset": DATASET_NAMES[r["dataset"]], "id": r["uid"],
                 "LLM_Output": r["pred"]},
                ensure_ascii=False) + "\n")
    print(f"wrote {len(ordered)} predictions -> {pred_path}")

    if args.package:
        rows = compose_predictions(scored, ids_by_ds)
        write_and_zip(rows, out_dir, ids_by_ds)
    else:
        print("\nNext (full coverage required for a valid zip):")
        print(f"  python src/make_submission.py package "
              f"--predictions {pred_path} --out_dir {out_dir}")


if __name__ == "__main__":
    main()
