"""Refine Indonesian rationales for tied-consensus (2-2-1) items using
value-level summaries distilled from the rest of the dataset.

Pipeline (both stages use the same <=8B model, Track-1 legal):

  stage 1 `summarize`:
    * take the Indonesian rationales EXCLUDING the 72 tied (2-2-1) items,
    * group them by the 5 Pancasila value categories (Democracy, Social
      Justice, Humanity, Unity, Religion),
    * map-reduce with the model: chunk-summaries -> exactly 5 insight
      summaries per value, in Indonesian, focused on how Indonesians frame
      such dilemmas and how the majority responded,
    * save to processed/id_value_summaries.json.

  stage 2 `rewrite`:
    * copy the input rationales file,
    * for each of the 72 tied Indonesian items, build the original CoT prompt
      PLUS the matching value summary as guidance, sample k chains, and accept
      the first whose answer lands in the consensus set (either tied letter),
    * fallback (no chain lands): rationalize a DETERMINISTIC uid-hash pick
      between the two tied letters — unlike the old consensus[0] fallback this
      does not systematically favour the alphabetically-first letter,
    * write the updated copy (source="summary_guided" on replaced rows).

  python src/refine_id_rationales.py all \
      --model meta-llama/Llama-3.1-8B-Instruct --load_4bit \
      --rationales all_rationales_unshuffled.jsonl \
      --out all_rationales_summary_guided.jsonl
"""
import argparse
import json
import zlib
from pathlib import Path

# NOTE: bootstrap_rationales (which imports torch) is imported lazily inside
# the GPU stages, so the CPU-only `fix_shifts` stage runs without torch.
from prompts import build_messages

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
PANCASILA = ["Democracy", "Social Justice", "Humanity", "Unity", "Religion"]

CHUNK_SIZE = 12          # rationales per map-step call
RATIONALE_CLIP = 700     # chars per rationale inside a chunk prompt


def load_indonesian(processed_dir):
    recs = load_jsonl(Path(processed_dir) / "indonesian.jsonl")
    return {r["uid"]: r for r in recs}


def tied_uids(id_recs):
    return {uid for uid, r in id_recs.items() if len(r["consensus"]) == 2}


# ------------------------------------------------------------- stage 1 ------
CHUNK_PROMPT = """You are analyzing how Indonesian annotators judge value dilemmas related to the Pancasila value "{value}".

Below are explanations (in Indonesian) of the majority-preferred answers for several dilemmas of this value category:

{rationales}

Summarize in Indonesian (max 150 words): what considerations do Indonesians consistently prioritize in these "{value}" dilemmas, and what kinds of responses does the majority prefer?"""

REDUCE_PROMPT = """You are an expert on Indonesian societal values (Pancasila). Below are several analytical notes about how Indonesian annotators judge value dilemmas in the category "{value}":

{chunk_summaries}

Write EXACTLY 5 numbered insight summaries in Indonesian (1. ... 5. ...), each 2-3 sentences. Each insight must capture a distinct pattern of how Indonesians see these kinds of problems and how the majority tends to respond (what they prioritize, what they avoid, what trade-offs they accept). Do not mention specific letters or options — describe the reasoning patterns."""


def summarize(gen, id_recs, rationale_rows, out_path):
    excluded = tied_uids(id_recs)
    by_value = {v: [] for v in PANCASILA}
    for row in rationale_rows:
        if row["dataset"] != "indonesian" or row["uid"] in excluded:
            continue
        rec = id_recs.get(row["uid"])
        if rec and rec["value_english"] in by_value:
            by_value[rec["value_english"]].append(row["rationale"])

    summaries = {}
    for value, rats in by_value.items():
        print(f"[summarize] {value}: {len(rats)} rationales")
        chunk_notes = []
        for i in range(0, len(rats), CHUNK_SIZE):
            chunk = rats[i:i + CHUNK_SIZE]
            body = "\n\n".join(f"- {r[:RATIONALE_CLIP]}" for r in chunk)
            msgs = [{"role": "user",
                     "content": CHUNK_PROMPT.format(value=value, rationales=body)}]
            chunk_notes.append(gen.chat(msgs, n=1, temperature=0.3)[0].strip())
            print(f"  chunk {i // CHUNK_SIZE + 1}/"
                  f"{(len(rats) + CHUNK_SIZE - 1) // CHUNK_SIZE} done")
        msgs = [{"role": "user",
                 "content": REDUCE_PROMPT.format(
                     value=value,
                     chunk_summaries="\n\n".join(chunk_notes))}]
        summaries[value] = gen.chat(msgs, n=1, temperature=0.3)[0].strip()
        print(f"[summarize] {value}: final 5-insight summary written")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"summaries -> {out_path}")
    return summaries


# ------------------------------------------------------------- stage 2 ------
GUIDED_SUFFIX = """

Sebagai panduan, berikut pola umum bagaimana mayoritas masyarakat Indonesia menilai dilema nilai "{value}":
{summary}

Gunakan pola-pola ini untuk memutuskan pilihan yang paling sesuai dengan penilaian mayoritas masyarakat Indonesia."""


def uid_pick(uid, consensus):
    """Deterministic, alphabet-unbiased pick between tied letters."""
    return sorted(consensus)[zlib.crc32(uid.encode("utf-8")) % len(consensus)]


def rewrite(gen, id_recs, rationale_rows, summaries, out_path, k, temperature):
    from bootstrap_rationales import parse_answer, strip_final_answer
    tied = tied_uids(id_recs)
    n_guided = n_fallback = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rationale_rows:
            if not (row["dataset"] == "indonesian" and row["uid"] in tied):
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            rec = id_recs[row["uid"]]
            consensus = set(rec["consensus"])
            summary = summaries.get(rec["value_english"], "")
            messages = build_messages(rec, mode="cot")
            messages[-1]["content"] += GUIDED_SUFFIX.format(
                value=rec["value_english"], summary=summary)

            rationale, answer, source = None, None, None
            for text in gen.chat(messages, n=k, temperature=temperature):
                a = parse_answer(text)
                if a is not None and a in consensus:
                    rationale, answer, source = (strip_final_answer(text), a,
                                                 "summary_guided")
                    n_guided += 1
                    break
            if rationale is None:
                target = uid_pick(rec["uid"], rec["consensus"])
                msgs = [dict(m) for m in messages]
                msgs[-1]["content"] += (
                    f"\n\nJawaban yang benar diketahui adalah {target}. Jelaskan "
                    f"secara singkat mengapa pilihan ini paling sesuai dengan "
                    f"penilaian mayoritas, lalu akhiri dengan baris "
                    f"\"Answer: {target}\".")
                text = gen.chat(msgs, n=1, temperature=0.3)[0]
                rationale, answer, source = (strip_final_answer(text), target,
                                             "summary_guided_rationalized")
                n_fallback += 1

            fout.write(json.dumps({
                "key": row["key"], "uid": row["uid"], "dataset": "indonesian",
                "rationale": rationale, "answer": answer, "source": source,
                # always 0: messages/rationale above were built from the
                # CANONICAL (unpermuted) rec, regardless of what shift the
                # original (now-replaced) rationale row carried
                "shift": 0,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[rewrite] {row['uid']} -> {answer} ({source})")
    print(f"rewrite done: {n_guided} guided, {n_fallback} fallback -> {out_path}")


def fix_shifts(id_recs, rationale_rows, out_path):
    """Deterministic repair for summary-guided files produced by the pre-fix
    version of `rewrite` (which copied the stale shift tag from the replaced
    row). The rewritten rationales were always generated from the CANONICAL
    option order, so the correct shift for every summary-guided tied row is 0.
    No GPU / no model needed; output is byte-reproducible."""
    tied = tied_uids(id_recs)
    n_fixed = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rationale_rows:
            if (row["dataset"] == "indonesian" and row["uid"] in tied
                    and row.get("source", "").startswith("summary_guided")):
                if row.get("shift", 0) != 0:
                    n_fixed += 1
                row = dict(row)
                row["shift"] = 0
                assert row["answer"] in id_recs[row["uid"]]["consensus"], \
                    f"{row['uid']}: answer {row['answer']!r} not in consensus"
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"fix_shifts: corrected {n_fixed} stale shift tags -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["summarize", "rewrite", "all", "fix_shifts"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--processed_dir", default=str(ROOT / "processed"))
    ap.add_argument("--rationales", required=True,
                    help="input rationales jsonl (e.g. all_rationales_unshuffled.jsonl; "
                         "for fix_shifts: the broken summary-guided output file)")
    ap.add_argument("--summaries", default=str(ROOT / "processed" / "id_value_summaries.json"))
    ap.add_argument("--out", default=str(ROOT / "all_rationales_summary_guided.jsonl"),
                    help="rewrite/fix_shifts: corrected copy of the rationales file")
    ap.add_argument("--k", type=int, default=8, help="rewrite: CoT samples per item")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--load_4bit", action="store_true")
    args = ap.parse_args()

    id_recs = load_indonesian(args.processed_dir)
    rationale_rows = load_jsonl(Path(args.rationales))
    print(f"{len(rationale_rows)} rationale rows loaded; "
          f"{len(tied_uids(id_recs))} tied Indonesian items")

    if args.stage == "fix_shifts":  # CPU-only repair, no model load
        fix_shifts(id_recs, rationale_rows, Path(args.out))
        return

    from bootstrap_rationales import Generator  # lazy: needs torch/GPU deps
    gen = Generator(args.model, load_4bit=args.load_4bit, max_new_tokens=500)

    summaries = None
    if args.stage in ("summarize", "all"):
        summaries = summarize(gen, id_recs, rationale_rows, Path(args.summaries))
    if args.stage in ("rewrite", "all"):
        if summaries is None:
            summaries = json.load(open(args.summaries, encoding="utf-8"))
        rewrite(gen, id_recs, rationale_rows, summaries, Path(args.out),
                args.k, args.temperature)


if __name__ == "__main__":
    main()
