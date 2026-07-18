"""Prompt templates for PlurVA-LLM Track 1.

Language policy (see METHODOLOGY.md §2.4/§2.5):
  Chinese    -> native Chinese instructions (base model strongest in zh)
  Indonesian -> native Indonesian instructions
  Sri Lankan -> English instructions + original Sinhala content (English-pivoted
                reasoning helps low-resource inputs; content stays native)

All templates end with a uniform machine-parseable marker:  "Answer: X"
(X in {A,B,C,D} for MCQ, {Yes,No} for the Sri Lankan binary decomposition).
Every template is value-conditioned: the annotated value is part of the input
schema and is present at test time, so we always show it to the model.

mode="direct" -> answer marker only (used for constrained letter scoring + SFT targets)
mode="cot"    -> brief reasoning first, then the marker (used for rationale
                 bootstrapping and self-consistency later)
"""

ANSWER_RE = r"Answer:\s*([ABCD]|Yes|No)"

# ---------------------------------------------------------------- Chinese ---
ZH_SYSTEM = (
    "你是一位深谙中国社会价值观与日常伦理困境的助手。"
    "题目描述了一个第一人称的价值冲突情境，四个选项都是现实中可能的做法。"
    "你的任务是选出最能践行指定核心价值、同时符合中国社会主流判断的选项。"
)

ZH_USER = """本题考察的核心价值：{value_native}（{value_english}）

情境与问题：
{question}

选项：
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

{instruction}"""

ZH_INSTR_DIRECT = "请直接给出最终选择，只输出一行，格式为 “Answer: X”（X 为 A/B/C/D）。"
ZH_INSTR_COT = (
    "请先简要分析各选项如何践行或违背该核心价值（不超过150字），"
    "然后在最后单独一行给出最终选择，格式为 “Answer: X”（X 为 A/B/C/D）。"
)

# ------------------------------------------------------------- Indonesian ---
ID_SYSTEM = (
    "Anda adalah asisten yang memahami nilai-nilai Pancasila dan norma masyarakat "
    "Indonesia. Tugas Anda adalah memilih tindakan yang paling sesuai dengan "
    "penilaian mayoritas masyarakat Indonesia terhadap nilai yang diuji."
)

ID_USER = """Nilai yang diuji: {value_native} ({value_english})

Skenario:
{scenario}

Pertanyaan:
{question}

Pilihan:
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

{instruction}"""

ID_INSTR_DIRECT = ("Berikan jawaban akhir Anda dalam satu baris saja dengan format "
                   "\"Answer: X\" (X adalah A/B/C/D).")
ID_INSTR_COT = (
    "Analisislah secara singkat setiap pilihan berdasarkan nilai yang diuji "
    "(maksimal 100 kata), lalu tuliskan jawaban akhir pada baris terakhir dengan "
    "format \"Answer: X\" (X adalah A/B/C/D)."
)

# Optional value-context blocks (<prefix>_value_summaries.json): prepended to
# the user turn when a summaries dict is available for that record's value.
# Absent by default -> byte-identical to the original prompt. Each country's
# block is written in that country's prompt language (Chinese native,
# Indonesian native, Sri Lankan English), matching the language policy above.
ID_VALUE_CONTEXT_BLOCK = """Konteks nilai — pola umum bagaimana mayoritas masyarakat Indonesia menilai dilema nilai "{value}":
{summary}

"""

ZH_VALUE_CONTEXT_BLOCK = """价值背景——中国社会在面对"{value}"类价值困境时的常见判断模式：
{summary}

"""

SI_VALUE_CONTEXT_BLOCK = """Value context — common patterns in how Sri Lankan society judges "{value}" dilemmas:
{summary}

"""

# -------------------------------------------------- Sri Lankan (binary) -----
SI_BIN_SYSTEM = (
    # "You are an assistant with deep familiarity with Sri Lankan societal values "
    # "across its Sinhalese, Tamil, Muslim and Burgher communities. You will read a "
    # "question and one candidate statement, both in Sinhala. Judge whether the "
    # "statement is a correct and socially appropriate answer to the question "
    # "according to the given Sri Lankan value."
    """You are an assistant with deep familiarity with Sri Lankan societal values across. You will read a question and one candidate statement, both in Sinhala. 
    Judge whether the statement is a correct and socially appropriate answer to the question according to the given Sri Lankan value."""
)

SI_BIN_USER = """Value being tested: {value_english}

Question (Sinhala):
{question}

Candidate statement (Sinhala):
{statement}

{instruction}"""

SI_BIN_INSTR_DIRECT = ("Reply with exactly one line: \"Answer: Yes\" if the statement is a "
                       "correct and appropriate answer to the question, or \"Answer: No\" "
                       "if it is not.")
SI_BIN_INSTR_COT = (
    "First, briefly explain in English what the question asks and whether the "
    "statement upholds or violates the value in a Sri Lankan context (max 80 words). "
    "Then give your final judgment on the last line as \"Answer: Yes\" or \"Answer: No\"."
)

# ---------------------------------------------- Sri Lankan (4-way, baseline) -
SI_4WAY_SYSTEM = (
    "You are an assistant with deep familiarity with Sri Lankan societal values. "
    "You will read a question and two candidate statements in Sinhala. Decide "
    "which statement(s) are correct answers according to the given value."
)

SI_4WAY_USER = """Value being tested: {value_english}

Question (Sinhala):
{question}

Statement A (Sinhala):
{opt_a}

Statement B (Sinhala):
{opt_b}

{instruction}"""

SI_4WAY_INSTR_DIRECT = ("Reply with exactly one line: \"Answer: A\" (only A is correct), "
                        "\"Answer: B\" (only B is correct), \"Answer: Both\", or "
                        "\"Answer: 0\" (neither is correct).")


def load_value_summaries(path):
    """Load a <prefix>_value_summaries.json file ({value_category -> summary})."""
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Backward-compatible alias (evaluate.py / make_submission.py import this name).
load_id_value_summaries = load_value_summaries


import os as _os

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_SUMMARY_FILES = {
    "chinese": "zh_value_summaries.json",
    "indonesian": "id_value_summaries.json",
    "sri_lankan": "si_value_summaries.json",
}
_summaries_cache = {}


def _auto_value_summaries(dataset):
    """Lazily load and cache <prefix>_value_summaries.json for a dataset from the
    repo root. Returns {} (no-op) if the file isn't present, so this never
    raises. Datasets without a summary file also get {}."""
    if dataset not in _summaries_cache:
        fname = _SUMMARY_FILES.get(dataset)
        try:
            _summaries_cache[dataset] = (
                load_value_summaries(_os.path.join(_REPO_ROOT, fname))
                if fname else {})
        except FileNotFoundError:
            _summaries_cache[dataset] = {}
    return _summaries_cache[dataset]


def _summary_key(rec):
    """Key used to look up a record's value summary. Chinese summaries are
    grouped by the top-level taxonomy root (segment before the first '/');
    Indonesian/Sri Lankan use value_english directly."""
    if rec["dataset"] == "chinese":
        return rec["value_english"].split("/")[0].strip()
    return rec["value_english"]


def build_messages(rec, mode="direct", si_statement=None, value_summaries="auto"):
    """Build chat messages for a processed record.

    For sri_lankan records, si_statement="A"|"B" selects the binary-decomposition
    prompt for that statement; si_statement=None gives the 4-way baseline prompt.

    value_summaries: controls the value-context block prepended to the user
    turn (per-country, keyed by the record's value category):
      "auto" (the default) -> auto-loads the record's country summary file from
        the repo root (chinese -> zh_value_summaries.json, indonesian ->
        id_value_summaries.json, sri_lankan -> si_value_summaries.json) and
        injects the matching value's summary. This is the standing default, so
        every caller that doesn't pass this argument (build_sft_data.py,
        bootstrap_rationales.py, train_dpo.py, self_consistency.py,
        make_submission.py, ...) embeds it automatically.
      a dict {value_category: summary_text} -> use that dict instead (e.g. to
        A/B a hand-written summary set).
      None -> explicitly disable injection, reproducing the original prompt
        byte-for-byte. Use this when evaluating an adapter that was trained
        BEFORE this default changed, to avoid a train/inference prompt mismatch.

    Returns a list of {"role", "content"} dicts (no assistant turn).
    """
    ds = rec["dataset"]
    summaries = (_auto_value_summaries(ds) if value_summaries == "auto"
                 else value_summaries)

    def _with_context(user, block):
        """Prepend the value-context block if a summary exists for this value."""
        if summaries:
            summary = summaries.get(_summary_key(rec))
            if summary:
                return block.format(value=_summary_key(rec), summary=summary) + user
        return user

    if ds == "chinese":
        instr = ZH_INSTR_DIRECT if mode == "direct" else ZH_INSTR_COT
        user = ZH_USER.format(
            value_native=rec["value_native"], value_english=rec["value_english"],
            question=rec["question"],
            opt_a=rec["options"]["A"], opt_b=rec["options"]["B"],
            opt_c=rec["options"]["C"], opt_d=rec["options"]["D"],
            instruction=instr)
        user = _with_context(user, ZH_VALUE_CONTEXT_BLOCK)
        return [{"role": "system", "content": ZH_SYSTEM},
                {"role": "user", "content": user}]

    if ds == "indonesian":
        instr = ID_INSTR_DIRECT if mode == "direct" else ID_INSTR_COT
        user = ID_USER.format(
            value_native=rec["value_native"], value_english=rec["value_english"],
            scenario=rec["scenario"] or "-", question=rec["question"],
            opt_a=rec["options"]["A"], opt_b=rec["options"]["B"],
            opt_c=rec["options"]["C"], opt_d=rec["options"]["D"],
            instruction=instr)
        user = _with_context(user, ID_VALUE_CONTEXT_BLOCK)
        return [{"role": "system", "content": ID_SYSTEM},
                {"role": "user", "content": user}]

    if ds == "sri_lankan":
        if si_statement in ("A", "B"):
            instr = SI_BIN_INSTR_DIRECT if mode == "direct" else SI_BIN_INSTR_COT
            user = SI_BIN_USER.format(
                value_english=rec["value_english"], question=rec["question"],
                statement=rec["options"][si_statement], instruction=instr)
            user = _with_context(user, SI_VALUE_CONTEXT_BLOCK)
            return [{"role": "system", "content": SI_BIN_SYSTEM},
                    {"role": "user", "content": user}]
        user = SI_4WAY_USER.format(
            value_english=rec["value_english"], question=rec["question"],
            opt_a=rec["options"]["A"], opt_b=rec["options"]["B"],
            instruction=SI_4WAY_INSTR_DIRECT)
        user = _with_context(user, SI_VALUE_CONTEXT_BLOCK)
        return [{"role": "system", "content": SI_4WAY_SYSTEM},
                {"role": "user", "content": user}]

    raise ValueError(f"unknown dataset {ds!r}")
