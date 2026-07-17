"""Generate value-category summaries — "how does this country's majority
reason about this value?" — for Chinese, Indonesian, and Sri Lankan dev data.

For the selected --country, this:
  1. loads that country's processed records and bootstrap rationales
     (bootstrap_rationales.py output),
  2. groups the rationales by VALUE CATEGORY (definition below),
  3. map-reduces them with the model, in the country's native language, into
     EXACTLY 5 numbered insight bullets per category describing recurring
     reasoning patterns and majority judgments (no option letters mentioned,
     since summaries get reused across items with different option orders),
  4. writes {value_category: summary_text} to a per-country JSON file.

Value-category definition per country:
  indonesian : the 5 Pancasila values (Value_English field, used as-is)
  sri_lankan : the ~40 survey-derived societal values (Value_English field,
               used as-is)
  chinese    : the 3 top-level roots of the 158-path value taxonomy
               (Value_English field, truncated to the segment before the
               first "/" — e.g. "Survival Support/Safety/.../Personal Safety"
               -> "Survival Support"). The full 158 leaf paths are far too
               fine-grained (many have only 1-5 items) to summarize usefully.

Indonesian items with a 2-2-1 vote tie are excluded from the corpus (their
"majority" answer is genuinely ambiguous and would blur the pattern); Chinese
and Sri Lankan have no such tie concept (single gold answer / independent
Yes-No judgment) so nothing is excluded there.

--max_per_value caps how many rationales are sampled (seeded, reproducible)
per category, since Chinese's top-level buckets (480/160/150 items) are much
larger than Indonesian's/Sri Lanka's and would otherwise take far longer.

  # Indonesian (Pancasila values) -> id_value_summaries.json
  python src/refine_id_rationales.py --country indonesian \
      --model meta-llama/Llama-3.1-8B-Instruct --load_4bit \
      --rationales all_rationales_unshuffled.jsonl \
      --out id_value_summaries.json

  # Chinese (3 top-level value roots) -> zh_value_summaries.json
  python src/refine_id_rationales.py --country chinese \
      --model meta-llama/Llama-3.1-8B-Instruct --load_4bit \
      --rationales all_rationales_unshuffled.jsonl \
      --out zh_value_summaries.json

  # Sri Lankan (~40 societal values) -> si_value_summaries.json
  python src/refine_id_rationales.py --country sri_lankan \
      --model meta-llama/Llama-3.1-8B-Instruct --load_4bit \
      --rationales all_rationales_unshuffled.jsonl \
      --out si_value_summaries.json
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZE = 12          # rationales per map-step call
RATIONALE_CLIP = 700     # chars per rationale inside a chunk prompt
SEED = 42

COUNTRY_FILES = {"chinese": "chinese.jsonl", "indonesian": "indonesian.jsonl",
                 "sri_lankan": "sri_lankan.jsonl"}
OUT_PREFIX = {"chinese": "zh", "indonesian": "id", "sri_lankan": "si"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_country_records(processed_dir, country):
    recs = load_jsonl(Path(processed_dir) / COUNTRY_FILES[country])
    return {r["uid"]: r for r in recs}


def value_category(rec, country):
    """The grouping key for a record — see module docstring for the
    per-country definition."""
    if country == "chinese":
        return rec["value_english"].split("/")[0].strip()
    return rec["value_english"]  # indonesian, sri_lankan: field IS the category


def excluded_uids(recs, country):
    """Items to leave out of the summarization corpus: only Indonesian's
    2-2-1 vote ties (ambiguous majority). Chinese/Sri Lankan have a single
    gold answer / independent Yes-No judgment, so nothing is excluded."""
    if country != "indonesian":
        return set()
    return {uid for uid, r in recs.items() if len(r.get("consensus", [])) == 2}


# ---------------------------------------------------------- prompt sets ----
# One (chunk_prompt, reduce_prompt) pair per country, in that country's
# native prompt language (matching prompts.py's language policy: Chinese
# native, Indonesian native, Sri Lankan English -- SI rationale text is
# already English since bootstrap_rationales.py's SI CoT prompt asks for an
# English explanation over Sinhala content).

CHUNK_PROMPT_ID = """You are analyzing how Indonesian annotators judge value dilemmas related to the Pancasila value "{value}".

Below are explanations (in Indonesian) of the majority-preferred answers for several dilemmas of this value category:

{rationales}

Summarize in Indonesian (max 150 words): what considerations do Indonesians consistently prioritize in these "{value}" dilemmas, and what kinds of responses does the majority prefer?"""

REDUCE_PROMPT_ID = """You are an expert on Indonesian societal values (Pancasila). Below are several analytical notes about how Indonesian annotators judge value dilemmas in the category "{value}":

{chunk_summaries}

Write EXACTLY 5 numbered insight summaries in Indonesian (1. ... 5. ...), each 2-3 sentences. Each insight must capture a distinct pattern of how Indonesians see these kinds of problems and how the majority tends to respond (what they prioritize, what they avoid, what trade-offs they accept). Do not mention specific letters or options — describe the reasoning patterns."""

CHUNK_PROMPT_ZH = """You are analyzing how Chinese society views value dilemmas related to the value category "{value}".

Below are explanations of the correct options that match majority judgment within this value category:

{rationales}

Summarize in Chinese (max 150 characters): In this type of "{value}" dilemma, what does Chinese society generally prioritize? What kind of action does the majority tend to choose?"""

REDUCE_PROMPT_ZH = """You are an expert on Chinese societal values. Below are several analytical notes on how Chinese people judge "{value}"-type value dilemmas:

{chunk_summaries}

Write EXACTLY 5 numbered insight summaries in Chinese (1. ... 5. ...), each 2-3 sentences. Each insight should capture a distinct reasoning pattern (what is prioritized, what is avoided, what trade-offs are acceptable). Do not mention specific option letters (like A/B/C/D) — describe only the reasoning pattern itself."""

CHUNK_PROMPT_SI = """You are analyzing how Sri Lankan society (across its Sinhalese, Tamil, Muslim, and Burgher communities) judges value dilemmas related to the value "{value}".

Below are explanations of the majority-supported judgments for several such dilemmas:

{rationales}

Summarize in English (max 150 words): what do Sri Lankans consistently prioritize in "{value}" dilemmas, and what kind of response does the majority tend to support?"""

REDUCE_PROMPT_SI = """You are an expert on Sri Lankan societal values. Below are several analytical notes on how Sri Lankan society judges dilemmas in the category "{value}":

{chunk_summaries}

Write EXACTLY 5 numbered insight summaries in English (1. ... 5. ...), each 2-3 sentences. Each insight must capture a distinct pattern in how Sri Lankans reason about this value — what they prioritize, what they avoid, what trade-offs they accept. Do not reference specific letters or statements — describe the reasoning patterns."""

PROMPTS = {
    "indonesian": (CHUNK_PROMPT_ID, REDUCE_PROMPT_ID),
    "chinese": (CHUNK_PROMPT_ZH, REDUCE_PROMPT_ZH),
    "sri_lankan": (CHUNK_PROMPT_SI, REDUCE_PROMPT_SI),
}


def summarize(gen, recs, rationale_rows, country, out_path, max_per_value, seed):
    chunk_prompt, reduce_prompt = PROMPTS[country]
    excluded = excluded_uids(recs, country)

    by_value = {}
    for row in rationale_rows:
        if row["dataset"] != country or row["uid"] in excluded:
            continue
        rec = recs.get(row["uid"])
        if not rec:
            continue
        by_value.setdefault(value_category(rec, country), []).append(row["rationale"])

    rng = random.Random(seed)
    summaries = {}
    for value, rats in sorted(by_value.items()):
        if max_per_value and len(rats) > max_per_value:
            rats = rng.sample(rats, max_per_value)
        print(f"[{country}] {value}: {len(rats)} rationales")
        chunk_notes = []
        n_chunks = (len(rats) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(0, len(rats), CHUNK_SIZE):
            chunk = rats[i:i + CHUNK_SIZE]
            body = "\n\n".join(f"- {r[:RATIONALE_CLIP]}" for r in chunk)
            msgs = [{"role": "user",
                     "content": chunk_prompt.format(value=value, rationales=body)}]
            chunk_notes.append(gen.chat(msgs, n=1, temperature=0.3)[0].strip())
            print(f"  chunk {i // CHUNK_SIZE + 1}/{n_chunks} done")
        msgs = [{"role": "user",
                 "content": reduce_prompt.format(
                     value=value, chunk_summaries="\n\n".join(chunk_notes))}]
        summaries[value] = gen.chat(msgs, n=1, temperature=0.3)[0].strip()
        print(f"[{country}] {value}: final 5-insight summary written")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"summaries -> {out_path}")
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True,
                    choices=["chinese", "indonesian", "sri_lankan"],
                    help="which country's dev data + rationales to summarize")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--rationales", required=True,
                    help="bootstrap_rationales.py output jsonl "
                         "(e.g. all_rationales_unshuffled.jsonl)")
    ap.add_argument("--out", default=None,
                    help="output json path (default: <prefix>_value_summaries.json "
                         "at the repo root, prefix = zh/id/si)")
    ap.add_argument("--max_per_value", type=int, default=60,
                    help="cap rationales sampled per value category (0 = no cap); "
                         "keeps runtime bounded for Chinese's large top-level "
                         "buckets (480/160/150 items) vs Indonesian's/Sri "
                         "Lanka's much smaller per-value groups")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--load_4bit", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        ROOT / f"{OUT_PREFIX[args.country]}_value_summaries.json")

    recs = load_country_records(args.processed_dir, args.country)
    rationale_rows = load_jsonl(Path(args.rationales))
    n_excl = len(excluded_uids(recs, args.country))
    print(f"[{args.country}] {len(recs)} dev items, {len(rationale_rows)} "
          f"rationale rows loaded, {n_excl} excluded (vote ties)")

    from bootstrap_rationales import Generator  # lazy: needs torch/GPU deps
    gen = Generator(args.model, load_4bit=args.load_4bit, max_new_tokens=500)
    summarize(gen, recs, rationale_rows, args.country, out_path,
              args.max_per_value, args.seed)


if __name__ == "__main__":
    main()