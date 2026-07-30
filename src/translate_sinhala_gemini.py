"""Translate Sri Lankan Sinhala JSONL records to English with Gemini.

The input rows are preserved verbatim. Each output row adds:

    "translation_english": {
        "question": "...",
        "scenario": "...",
        "options": {"A": "...", "B": "..."}
    },
    "translation_meta": {
        "source_language": "si",
        "target_language": "en",
        "model": "gemini-3.1-flash-lite"
    }

Both repository schemas are supported:

* Normalized: uid/question/scenario/options
* Raw dev/test: ID/Question/Scenario/Option_A...Option_D

The output is checkpointed after every record and is safely resumable. API keys
are read from an environment variable or a Google Colab userdata secret and
are never written to disk or printed.

Examples (Colab):

    # Secret saved in Colab as GEMINI_API_KEY
    !python src/translate_sinhala_gemini.py \
        --input data/sri_lankan_dev.jsonl \
        --output translated/sri_lankan_dev_english.jsonl

    # Smoke test without modifying the full output
    !python src/translate_sinhala_gemini.py \
        --input negation_sinhala_data.jsonl \
        --output translated/negation_sinhala_english.jsonl \
        --limit 5

NOTE: If these translations will be used for shared-task training, confirm that
offline annotation/translation by a closed commercial API is permitted and
disclose it in the system description.
"""

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
DEFAULT_MODEL = "gemini-3.1-flash-lite"
OPTION_LABELS = ("A", "B", "C", "D")

SYSTEM_INSTRUCTION = """You are a professional Sinhala-to-English translator.
Translate the supplied Sri Lankan sinhala question faithfully into clear,
natural English.

Requirements:
- Preserve the exact meaning, especially every negation, exclusion, comparison,
  obligation, and culturally specific distinction.
- Translate each option independently without deciding which option is correct.
- Do not answer, explain, summarize, simplify, or add facts.
- Preserve names, numbers, abbreviations, and technical terms accurately.
- Return only the requested structured JSON."""


def load_jsonl(path):
    rows = []
    seen = set()
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            uid = record_uid(row)
            if uid in seen:
                raise ValueError(f"{path}:{line_no}: duplicate ID {uid!r}")
            seen.add(uid)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no JSONL records")
    return rows


def record_uid(row):
    uid = row.get("uid", row.get("ID", row.get("id")))
    if not isinstance(uid, str) or not uid:
        raise ValueError("record is missing uid/ID/id")
    return uid


def extract_translatable(row):
    """Return one schema-independent translation payload."""
    normalized = "question" in row or "options" in row
    question_key = "question" if normalized else "Question"
    scenario_key = "scenario" if normalized else "Scenario"
    question = row.get(question_key, "")
    scenario = row.get(scenario_key, "")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{record_uid(row)}: missing question text")
    if not isinstance(scenario, str):
        raise ValueError(f"{record_uid(row)}: scenario must be a string")

    if normalized:
        raw_options = row.get("options")
        if not isinstance(raw_options, dict):
            raise ValueError(f"{record_uid(row)}: options must be an object")
        options = {
            str(label): text for label, text in raw_options.items()
            if isinstance(text, str) and text.strip()
        }
    else:
        options = {
            label: row.get(f"Option_{label}", "")
            for label in OPTION_LABELS
            if isinstance(row.get(f"Option_{label}", ""), str)
            and row.get(f"Option_{label}", "").strip()
        }
    if not options:
        raise ValueError(f"{record_uid(row)}: no non-empty answer options")
    return {
        "id": record_uid(row),
        "question": question,
        "scenario": scenario,
        "options": options,
    }


def response_schema(option_labels):
    return {
        "type": "OBJECT",
        "properties": {
            "question_english": {"type": "STRING"},
            "scenario_english": {"type": "STRING"},
            "options_english": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "label": {
                            "type": "STRING",
                            "enum": list(option_labels),
                        },
                        "text": {"type": "STRING"},
                    },
                    "required": ["label", "text"],
                },
            },
        },
        "required": [
            "question_english",
            "scenario_english",
            "options_english",
        ],
    }


def build_prompt(payload):
    return (
        "Translate this JSON object from Sinhala to English. Keep each option "
        "under its original label.\n\n" +
        json.dumps(payload, ensure_ascii=False)
    )


def parse_translation(result, source):
    question = result.get("question_english")
    scenario = result.get("scenario_english")
    option_rows = result.get("options_english")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Gemini returned an empty question translation")
    if not isinstance(scenario, str):
        raise ValueError("Gemini returned an invalid scenario translation")
    if not isinstance(option_rows, list):
        raise ValueError("Gemini returned invalid options_english")

    options = {}
    for item in option_rows:
        if not isinstance(item, dict):
            raise ValueError("Gemini returned a non-object option")
        label, text = item.get("label"), item.get("text")
        if label in options:
            raise ValueError(f"Gemini returned duplicate option {label!r}")
        if label not in source["options"]:
            raise ValueError(f"Gemini returned unexpected option {label!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Gemini returned empty option {label!r}")
        options[label] = text
    expected = set(source["options"])
    if set(options) != expected:
        missing = sorted(expected - set(options))
        raise ValueError(f"Gemini omitted option(s): {missing}")
    if not source["scenario"].strip() and scenario.strip():
        raise ValueError(
            "Gemini invented a scenario even though the source scenario is empty")
    return {
        "question": question,
        "scenario": scenario,
        "options": {
            label: options[label] for label in source["options"]
        },
    }


def call_gemini(api_key, model, source, timeout, retries):
    body = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}],
        },
        "contents": [{
            "role": "user",
            "parts": [{"text": build_prompt(source)}],
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseSchema": response_schema(source["options"].keys()),
        },
    }
    request_data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    url = API_URL.format(model=model)
    last_error = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            data=request_data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            candidates = payload.get("candidates") or []
            if not candidates:
                raise ValueError(
                    f"Gemini returned no candidates: "
                    f"{payload.get('promptFeedback', {})}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(
                part.get("text", "") for part in parts
                if isinstance(part, dict)
            )
            if not text:
                raise ValueError("Gemini returned no response text")
            result = json.loads(text)
            return parse_translation(result, source)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            last_error = f"HTTP {e.code}: {detail}"
            transient = e.code in (408, 429, 500, 502, 503, 504)
            if not transient or attempt == retries - 1:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                ValueError) as e:
            last_error = str(e)
            if attempt == retries - 1:
                break
        delay = min(60.0, (2 ** attempt) * 2.0 + random.random())
        print(f"    retry {attempt + 2}/{retries} after {delay:.1f}s: "
              f"{last_error}")
        time.sleep(delay)
    raise RuntimeError(f"Gemini translation failed: {last_error}")


def resolve_api_key(secret_name):
    key = os.environ.get(secret_name)
    if key:
        return key
    try:
        from google.colab import userdata
        key = userdata.get(secret_name)
    except Exception:
        key = None
    if key:
        return key
    raise SystemExit(
        f"Gemini API key not found. Save a Colab secret named {secret_name!r} "
        f"or set the {secret_name} environment variable.")


def translated_row(row, translation, model):
    result = dict(row)
    result["translation_english"] = translation
    result["translation_meta"] = {
        "source_language": "si",
        "target_language": "en",
        "model": model,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "sri_lankan_dev.jsonl"),
        help="Sinhala JSONL input file",
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "translated" / "sri_lankan_dev_english.jsonl"),
        help="resumable translated JSONL output",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--secret_name",
        default="GEMINI_API_KEY",
        help="environment variable or Colab userdata secret containing the key",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="translate at most this many input records; 0 means all",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="seconds to wait after each successful API request",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    if args.limit < 0 or args.sleep < 0 or args.timeout <= 0 or args.retries <= 0:
        raise ValueError(
            "--limit/--sleep cannot be negative and timeout/retries must be "
            "positive")
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--input and --output must be different files")

    api_key = resolve_api_key(args.secret_name)
    source_rows = load_jsonl(input_path)
    if args.limit:
        source_rows = source_rows[:args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output_path.exists() and output_path.stat().st_size:
        existing = load_jsonl(output_path)
        input_uids = {record_uid(row) for row in source_rows}
        foreign = [
            record_uid(row) for row in existing
            if record_uid(row) not in input_uids
        ]
        if foreign:
            raise ValueError(
                f"{output_path} contains IDs not present in this input: "
                f"{foreign[:5]}")
        wrong_models = {
            row.get("translation_meta", {}).get("model")
            for row in existing
            if row.get("translation_meta", {}).get("model") != args.model
        }
        if wrong_models:
            raise ValueError(
                f"{output_path} contains translations from model(s) "
                f"{sorted(str(model) for model in wrong_models)}; use a "
                "different --output path when changing --model")
        completed = {record_uid(row) for row in existing}
        print(f"resuming: {len(completed)} records already translated")

    todo = [row for row in source_rows if record_uid(row) not in completed]
    print(
        f"model={args.model}; input={len(source_rows)}; "
        f"remaining={len(todo)}; output={output_path}")
    failures = []
    with open(output_path, "a", encoding="utf-8", newline="\n") as output:
        for index, row in enumerate(todo, 1):
            uid = record_uid(row)
            try:
                source = extract_translatable(row)
                translation = call_gemini(
                    api_key,
                    args.model,
                    source,
                    timeout=args.timeout,
                    retries=args.retries,
                )
            except (ValueError, RuntimeError) as e:
                failures.append((uid, str(e)))
                print(f"[{index}/{len(todo)}] FAILED {uid}: {e}")
                continue
            output.write(json.dumps(
                translated_row(row, translation, args.model),
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n")
            output.flush()
            print(f"[{index}/{len(todo)}] translated {uid}")
            time.sleep(args.sleep)

    total_completed = len(completed) + len(todo) - len(failures)
    print(f"completed {total_completed}/{len(source_rows)} -> {output_path}")
    if failures:
        print(f"{len(failures)} record(s) failed; rerun the same command:")
        for uid, error in failures[:10]:
            print(f"  {uid}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
