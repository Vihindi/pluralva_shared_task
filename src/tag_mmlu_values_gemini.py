"""Tag SinhalaMMLU items with a shared-task value category using the Gemini API.

Alternative to tag_mmlu_values.py (which uses a local <=8B model). Gemini is
given every value in the SI dev vocabulary with BOTH its English and Sinhala
term, plus the Sinhala question, its options and the correct answer, and must
return one value from that list.

Reliability: the request uses Gemini structured output with an `enum` over the
46 legal values, so the returned category is guaranteed in-vocabulary — the API
equivalent of the constrained scoring used elsewhere in this pipeline. No free
text to parse, no invalid tags.

  export GEMINI_API_KEY=...           # never hardcode: this is a git repo
  python src/tag_mmlu_values_gemini.py --limit 10        # smoke test first
  python src/tag_mmlu_values_gemini.py                   # full civics run

Output: processed/mmlu_value_tags.json — same schema as tag_mmlu_values.py, so
downstream steps are identical whichever tagger you use.

NOTE ON TRACK 1: closed commercial APIs are not permitted for the submitted
system. Using Gemini for offline data annotation is a grey area (it is arguably
distillation from a larger hidden model). Confirm with the organizers before
relying on tags produced here, and disclose it in the system description.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
           "{model}:generateContent")

# Item selection (allow-lists + contamination/manual drops) lives in
# tag_mmlu_values.py so the two taggers can never select different items.
from tag_mmlu_values import (CONTAMINATED_DROP, MANUAL_DROP, KEEP_ONLY,  # noqa: F401
                             DEFAULT_DROP, select_items)

# English -> Sinhala for the 46 values in the SI dev set.
# VERIFY THESE: they were written by the assistant, not a native speaker, and
# a wrong gloss will bias the tagging. Contentment/Happiness and Pride/Social
# Standing are the pairs most likely to need correction.
VALUE_SINHALA = {
    "Acceptance": "පිළිගැනීම",
    "Accountability": "වගවීම",
    "Ambition": "අභිලාෂය",
    "Authority": "අධිකාරිය",
    "Belonging": "අයත්වීමේ හැඟීම",
    "Community": "ප්‍රජාව",
    "Compassion": "කරුණාව",
    "Contentment": "ලද දෙයින් සෑහීමට පත්වීම",
    "Culture and Tradition": "සංස්කෘතිය හා සම්ප්‍රදාය",
    "Determination": "අධිෂ්ඨානය",
    "Discipline": "විනය",
    "Environmentalism": "පරිසර සංරක්ෂණය",
    "Equality": "සමානාත්මතාවය",
    "Family": "පවුල",
    "Generosity": "ත්‍යාගශීලීත්වය",
    "Gratitude": "කෘතඥතාවය",
    "Happiness": "සතුට",
    "Health": "සෞඛ්‍යය",
    "Hospitality": "ආගන්තුක සත්කාරය",
    "Independence": "ස්වාධීනත්වය",
    "Justice": "යුක්තිය",
    "Minimalism": "සරල ජීවන රටාව",
    "Modernity": "නවීනත්වය",
    "Modesty": "නිහතමානිකම",
    "Morality": "සදාචාරය",
    "Multiculturalism": "බහු සංස්කෘතිකත්වය",
    "Non-violence": "අවිහිංසාව",
    "Patriotism": "දේශප්‍රේමය",
    "Peace": "සාමය",
    "Personal Growth": "පෞද්ගලික වර්ධනය",
    "Politeness": "ආචාරශීලීත්වය",
    "Political Freedom": "දේශපාලන නිදහස",
    "Pride": "අභිමානය",
    "Resilience": "ඔරොත්තු දීමේ හැකියාව",
    "Respect": "ගෞරවය",
    "Responsibility": "වගකීම",
    "Security": "ආරක්ෂාව",
    "Self-expression": "ස්වයං ප්‍රකාශනය",
    "Shamefulness": "ලැජ්ජාව",
    "Social Standing": "සමාජ තත්ත්වය",
    "Spirituality": "ආධ්‍යාත්මිකත්වය",
    "Stability": "ස්ථාවරත්වය",
    "Sustainability": "තිරසාරත්වය",
    "Tolerance": "ඉවසීම",
    "Trustworthiness": "විශ්වාසනීයත්වය",
    "Wealth": "ධනය",
}

PROMPT = """You are an expert on Sri Lankan societal values and the Sri Lankan school curriculum.

Below is a school question in Sinhala, its four answer options, and the correct answer. Your task is to identify which SINGLE societal value the question is fundamentally testing.

Question (Sinhala):
{question}

Options (Sinhala):
{options}

Correct answer: {answer}

You must choose exactly one value from this list of Sri Lankan societal values. Each is given as "English term (Sinhala term)":

{value_list}

Guidance:
- Choose the value that the CORRECT ANSWER embodies, not merely the topic of the question.
- Judge it in a Sri Lankan cultural context.
- If several values fit, pick the most specific one rather than a broad one.
- Set confidence to "low" if the question is purely factual and tests no societal value.

Return the English term of the chosen value. Strickly return only the English term of the chosen value."""


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def value_vocabulary(si_dev_path):
    dev = load_jsonl(si_dev_path)
    return sorted({d["value_english"].strip() for d in dev if d.get("value_english")})


def build_prompt(item, values):
    opts = "\n".join(f"{i+1}. {c}" for i, c in enumerate(item.get("choices", [])))
    choices = item.get("choices", [])
    ai = item.get("answer")
    answer_text = (choices[ai - 1] if isinstance(ai, int) and 1 <= ai <= len(choices)
                   else str(ai))
    value_list = "\n".join(
        f"- {v} ({VALUE_SINHALA.get(v, '?')})" for v in values)
    return PROMPT.format(question=item.get("question", ""), options=opts,
                         answer=answer_text, value_list=value_list)


def call_gemini(api_key, model, prompt, values, timeout=60, retries=4):
    """One tagging call with structured output; enum guarantees a legal value."""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "value_english": {"type": "STRING", "enum": values},
                    "confidence": {"type": "STRING",
                                   "enum": ["high", "medium", "low"]},
                    "reason": {"type": "STRING"},
                },
                "required": ["value_english", "confidence"],
            },
        },
    }
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
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            last_err = f"HTTP {e.code}: {detail}"
            # 429/5xx are transient — back off and retry
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            break
        except Exception as e:  # network/JSON hiccups
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
    raise RuntimeError(f"Gemini call failed: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", default=os.environ.get("GEMINI_API_KEY"),
                    help="defaults to $GEMINI_API_KEY (preferred — do not "
                         "hardcode a key into this repo)")
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--mmlu", default=str(ROOT / "sinhala_mmlu.json"))
    ap.add_argument("--si_dev", default=str(ROOT / "processed" / "sri_lankan.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "processed" / "mmlu_value_tags.json"))
    ap.add_argument("--subjects", nargs="+", default=["Civics"])
    ap.add_argument("--no_drop", action="store_true",
                    help="tag known duplicates too (breaks CV integrity if trained on)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap items")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="pause between calls to stay under rate limits")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("no API key: set GEMINI_API_KEY or pass --api_key")

    values = value_vocabulary(args.si_dev)
    missing = [v for v in values if v not in VALUE_SINHALA]
    if missing:
        print(f"WARNING: no Sinhala gloss for: {missing} (will show '?')")
    print(f"value vocabulary: {len(values)} values")

    import collections
    mmlu = load_jsonl(args.mmlu)
    subset = select_items(mmlu, args.subjects, apply_drop=not args.no_drop)
    by_subj = collections.Counter(m.get("subject", "").strip() for m in subset)
    print(f"{len(subset)} items selected from {args.subjects}: {dict(by_subj)}")
    if args.no_drop:
        print("  WARNING --no_drop: contaminated items included (breaks CV integrity)")
    if args.limit:
        subset = subset[: args.limit]

    # resumable: keep whatever is already tagged
    out_path = Path(args.out)
    tags = {}
    if out_path.exists():
        tags = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"resuming: {len(tags)} already tagged")

    import collections
    dist = collections.Counter(t["value"] for t in tags.values())
    conf = collections.Counter(t.get("confidence") for t in tags.values())
    n_new = 0
    for i, item in enumerate(subset):
        key = f"{item.get('subject','').strip()}|{item.get('q_no')}"
        if key in tags:
            continue
        prompt = build_prompt(item, values)
        try:
            res = call_gemini(args.api_key, args.model, prompt, values)
        except RuntimeError as e:
            print(f"  FAILED {key}: {e}")
            continue
        v = res.get("value_english")
        if v not in values:  # belt-and-braces; enum should prevent this
            print(f"  SKIP {key}: illegal value {v!r}")
            continue
        tags[key] = {"value": v, "confidence": res.get("confidence"),
                     "reason": res.get("reason", "")[:300],
                     "q_no": item.get("q_no"),
                     "subject": item.get("subject", "").strip(),
                     "tagger": f"gemini:{args.model}"}
        dist[v] += 1
        conf[res.get("confidence")] += 1
        n_new += 1
        out_path.write_text(json.dumps(tags, ensure_ascii=False, indent=2),
                            encoding="utf-8")  # checkpoint every item
        if n_new % 10 == 0:
            print(f"  {n_new} new tags ({i+1}/{len(subset)} scanned)")
        time.sleep(args.sleep)

    print(f"\ntagged {len(tags)} items total ({n_new} new) -> {out_path}")
    print(f"confidence: {dict(conf)}")
    print(f"distinct values used: {len(dist)} / {len(values)}")
    print("top assignments:")
    for v, c in dist.most_common(15):
        print(f"   {v:26s} {c}")
    low = [k for k, t in tags.items() if t.get("confidence") == "low"]
    if low:
        print(f"\nlow-confidence items ({len(low)}) — consider excluding: "
              f"{', '.join(low[:10])}{' ...' if len(low) > 10 else ''}")


if __name__ == "__main__":
    main()
