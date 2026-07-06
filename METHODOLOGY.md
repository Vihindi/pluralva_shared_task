# PlurVA-LLM 2026 Shared Task — Track 1 (Resource-Constrained) Methodology

**Goal:** maximize the macro-average of per-country accuracy (China, Indonesia, Sri Lanka) with a single unified system whose language-generating components are all ≤ 8B parameters, no closed APIs, no agents/tools.

---

## 1. What the data actually looks like (dev-set analysis)

All numbers computed from the released dev sets (`data/*.jsonl`).

### 1.1 Chinese (790 items, 4-way MCQ, single gold)

| Property | Finding |
|---|---|
| Gold distribution | A 30.0%, B 15.6%, C 17.2%, **D 37.2%** — strongly non-uniform |
| Value taxonomy | 158 unique fine-grained value paths (also given in English), 3 top-level roots: Social Order (480), Survival Support (160), Civilization & Progress (150) |
| Text length | Question ≈ 254 Chinese chars, options total ≈ 373 chars — long, dense dilemmas |
| Length artifact | **Gold is the longest option 51.8% of the time** (chance 25%); shortest only 12.8% |

**Specialities / issues**
- Every option is deliberately "reasonable" — options encode trade-offs (risk of retaliation, cost), so surface sentiment doesn't separate them. The gold option is typically the one that *practices the annotated value* while acknowledging realistic costs (see the example in the task description: the honest, legal, non-heroic option C).
- The per-item **value taxonomy is a strong label hint** and is part of the input schema, so it will be present at test time — condition on it.
- The longest-option artifact (~52%) means (a) a length-prior exists that a tuned model can absorb, and (b) length-based shortcuts must be controlled for when validating (a model that "wins" via length isn't learning values, and the hidden test may or may not preserve the artifact).
- Non-uniform gold positions mean naive uniform debiasing would *hurt* — calibrate to the dev prior instead.

### 1.2 Indonesian (366 items, 4-way MCQ, consensus of 5 annotator votes)

| Property | Finding |
|---|---|
| Gold format | Raw 5 votes, e.g. `"A, A, A, A, C"` — consensus = most-frequent label(s) |
| Consensus size | 294 items single-answer, **72 items have a 2-way tie (both count as correct)** |
| Agreement | Only 26/366 unanimous (5/5); 79 items 4/5; 158 items 3/5; **103 items with top vote = 2/5** (splits like 2-2-1 or 2-1-1-1) |
| Values | Democracy 110, Social Justice 94, Humanity 86, Unity 58, Religion 18 (Pancasila) |
| Baselines | always-C 36.9%, always-A 35.5%, random ≈ 29.9% (ties raise the floor) |

**Specialities / issues**
- This is a **human-disagreement dataset**, not a knowledge dataset. 28% of items have plurality support of only 2/5 — near-arbitrary consensus. Treat the vote vector as a *distribution* to be modeled, not a hard label (Plank, 2022; LeWiDi shared-task line of work).
- The tie rule is free accuracy: on tied items any of 2 labels scores 1. A distribution-aware decoder should exploit this (you only need to land inside the consensus set).
- Religion has only 18 dev items — per-value performance will be high-variance; don't over-tune per-value.
- Scenario field is populated only for Indonesian; prompts must concatenate Scenario + Question.

### 1.3 Sri Lankan (203 items, 2 statements → A / B / Both / 0)

| Property | Finding |
|---|---|
| Gold distribution | **A 52.2%**, Both 21.2%, B 16.7%, 0 9.9% |
| Values | 40 survey-derived societal values, ~3–9 items each (given in English too) |
| Text | Short (Q ≈ 98 chars); Options C/D always empty |
| Label naming | Dev gold uses `"0"`; the data readme says "None"; **the submission spec says `0` — emit `0`** |

**Specialities / issues**
- Sinhala is the **macro-average bottleneck**: on SinhalaMMLU, the best ≤8B open models score ~25–27% (near chance) — Qwen2.5-7B 27.2%, Llama-3.1-8B 25.3%, Aya-Expanse-8B 22.6% ([SinhalaMMLU](https://arxiv.org/abs/2509.03162)). The value-judgment task is linguistically easier than MMLU, but comprehension is still the limiting factor.
- Severe class skew (A 52%) plus two minority classes (Both, 0) that a naive 4-way prompt will under-predict. The task is really **two independent binary judgments** ("does statement X align with value V?") composed into {A, B, Both, 0} — decomposing it removes the skew problem and the A-position bias simultaneously.
- Only 203 dev items across 40 values → very little supervised signal per value; augmentation and cross-lingual transfer matter more here than anywhere else.

### 1.4 Cross-cutting
- Total supervised data is tiny: **1,359 labeled items**. Anything trained on it must be aggressively regularized and validated with cross-validation, never on the full dev set it was tuned on.
- Ranking is macro-average, so **+1 point on Sri Lanka is worth the same as +1 on China but is far cheaper to get** (it starts lowest). Allocate effort accordingly.
- The dataset name is part of each test instance, so per-country routing is trivial and explicitly allowed ("routing" is permitted in Track 1).

---

## 2. Proposed system: **VALORA** (Value-Aligned LoRA Routing Architecture)

One base model + per-country LoRA adapters + country-aware inference. Single system, single prediction file, everything ≤ 8B.

```
                       ┌────────────────────────────────────────────┐
 test item ── router ──┤  ZH: value-conditioned CoT-SFT LoRA        │
 (dataset field)       │  ID: soft-label (vote-distribution) LoRA   │──► calibrated
                       │  SI: translate-assist + binary-decompose   │    letter logits
                       └────────────────────────────────────────────┘        │
                              base: Qwen3-8B (≤8B, 119 languages)             ▼
                        permutation ensemble + prior calibration ──► predictions.jsonl
```

### 2.1 Base model: Qwen3-8B (with a validated fallback)

**Choice: [Qwen3-8B](https://arxiv.org/abs/2505.09388)** — rationale:
- Best-in-class Chinese among ≤8B open models (58% of the benchmark's items are Chinese-adjacent in difficulty weight since ZH is the largest dataset).
- Strong Indonesian (Qwen3 pretraining covers 119 languages/dialects incl. Indonesian and Sinhala).
- Hybrid thinking mode → free chain-of-thought at inference without a second model.

**Validation step (do first, cheap):** zero-shot dev accuracy of Qwen3-8B vs Llama-3.1-8B-Instruct vs Qwen2.5-7B-Instruct vs Aya-Expanse-8B vs [Sailor2-8B](https://arxiv.org/abs/2502.12982) (SEA-focused, strong Indonesian). Pick the best *macro* base; keep the runner-up for the ensemble ablation. Given SinhalaMMLU numbers, expect Qwen-family to win, but measure — the task style is different from MMLU.

### 2.2 Answer extraction: constrained letter-logit scoring, never free generation

Score `P(letter | prompt)` over the valid label set with a constrained one-token readout (plus a CoT variant where the model reasons first, then the readout is applied to the "final answer:" position). This guarantees zero invalid outputs (invalid = scored wrong per the task rules) and gives calibrated probabilities for the steps below. Fine-tuning likewise maximizes the gold-letter token likelihood (standard practice in [LoRA-ensemble MCQ setups](https://arxiv.org/abs/2310.00035)).

### 2.3 Debiasing: permutation ensemble + prior calibration (not uniform debiasing)

LLMs have strong option-ID/position bias ([Zheng et al., ICLR 2024 — PriDe](https://arxiv.org/abs/2309.03882)). Standard fixes assume the true answer distribution is uniform — **here it isn't** (ZH: D 37%; SI: A 52%). So:

1. **Permutation ensemble** at inference: evaluate each item under the 4 cyclic option orderings (2 orderings for SI), average the content-level probabilities. This removes position bias without assuming uniformity.
2. **Prior calibration**: after permutation averaging, multiply by the country-specific label prior estimated on held-out dev folds (a light-touch version of PriDe's prior separation, with the dev prior replacing the uniform assumption), then renormalize. Tune the calibration temperature per country on CV folds.
3. **Permutation augmentation in training** (§2.4) makes the fine-tuned model itself more order-invariant, reducing how much work steps 1–2 must do.

### 2.4 Fine-tuning stage 1 — multi-task LoRA SFT with rationale bootstrapping

Train LoRA adapters (r=16–32) on the dev data, one adapter per country plus one joint adapter (compare on CV; joint training may transfer across countries — the pluralistic-alignment literature suggests cultural signal transfers, e.g. [CultureLLM's](https://arxiv.org/abs/2402.10946) unified model matched its culture-specific ones).

**Targets:**
- **Rationale-augmented answers (STaR-style self-distillation, [Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)):** sample k=8 CoT solutions per dev item from the base model *conditioned on the annotated value*; keep only chains that reach the gold answer; SFT on (item → kept rationale + answer). For items where no chain succeeds, use rationalization (give the gold answer, ask the model to justify, keep the justification). This turns 1.3k labels into supervised *reasoning* data without any external model — fully Track-1 legal.
- **Value conditioning:** always include the `Value_English` (and native value) field in the prompt template — it's in the test schema and is the single strongest hint in the ZH data.

**Augmentation (all generation done by the ≤8B base model itself — no closed APIs):**
- **Option-permutation augmentation:** every training item appears under multiple option orders with relabeled gold — teaches order invariance and multiplies data 4×.
- **Semantic augmentation** à la [CultureLLM (NeurIPS 2024)](https://arxiv.org/abs/2402.10946): paraphrase scenarios while preserving the value conflict and label; validated to help cultural alignment from as few as 50 seeds. Apply most aggressively to Sri Lanka (203 items) and Indonesian Religion (18 items).
- **World Values Survey seeds** for China/Indonesia (public data; WVS covers both) converted to the task's MCQ style, following CultureLLM's recipe; Sri Lanka is not in recent WVS waves, so instead expand from the 40 annotated value names with self-generated scenario items (mirroring how the organizers built the dataset from value-tagged, AI-generated, human-validated items).

**Anti-overfit protocol:** 5-fold cross-validation on dev for *every* decision (adapter rank, epochs, prompt template, calibration temperature). The hidden test is 80% of the same distribution, so CV estimates transfer well; never report/tune on data a fold was trained on.

### 2.5 Fine-tuning stage 2 — country-specific objectives

**China — preference optimization.** Each item gives 1 chosen + 3 rejected options: natural preference pairs. Run [DPO](https://arxiv.org/abs/2305.18290) (or its variants; compare KTO) on (gold vs each distractor) pairs on top of the SFT adapter. Alternatively/additionally, **GRPO with a verifiable reward** (exact-match on the letter, [Shao et al. 2024](https://arxiv.org/abs/2402.03300)) on CoT rollouts — well suited because the reward is exact and cheap; watch for reward hacking via the length artifact (monitor whether the policy drifts toward always-longest).

**Indonesia — soft-label training.** Don't collapse the 5 votes: train the letter-logit distribution against the empirical vote distribution with KL/soft-CE loss. This is the established remedy when disagreement is signal, not noise ([Plank 2022](https://arxiv.org/abs/2211.02570); [SeedBERT](https://arxiv.org/abs/2211.13196); [LeWiDi-2025 findings](https://arxiv.org/abs/2510.12516) show test-time scaling alone doesn't fix disagreement — the training objective must change). At inference, predict argmax of the calibrated distribution; on near-ties the tie-scoring rule means both plausible labels are often *both* correct, so this directly buys accuracy on the 72/366-style tied items.

**Sri Lanka — decompose + translate-assist.**
1. **Binary decomposition:** ask two independent yes/no judgments — "Does statement A align with value V in this context?" and same for B — then map (yes,no)→A, (no,yes)→B, (yes,yes)→Both, (no,no)→0. This converts a skewed 4-way task into balanced binary tasks, directly attacks the under-prediction of Both/0, and gives a decision threshold to calibrate per class on dev folds.
2. **Translate-assist:** add an English gloss of the Sinhala item, produced by the base model itself or by NLLB-200 (≤3.3B, open weights, [NLLB](https://arxiv.org/abs/2207.04672)), *alongside* the original Sinhala (never replacing it — native content quality dominates translations per SinhalaMMLU, and English-pivoted reasoning helps low-resource inputs, [arXiv:2504.02890](https://arxiv.org/abs/2504.02890)). Validate on dev whether Sinhala-only, English-only, or bilingual prompting wins; keep the winner.
3. Fine-tune the binary judge on the 203 decomposed dev items (→ 406 binary examples) + semantic augmentations.

### 2.6 Inference-time ensembling

- **Self-consistency** ([Wang et al. 2022](https://arxiv.org/abs/2203.11171)): sample n=8–16 CoT chains per item, majority/probability-mass vote. Track 1 has no inference-compute limit.
- **Permutation ensemble** (§2.3) folded into the same batch.
- **K-fold LoRA ensemble**: the 5 CV adapters are free ensemble members ([LoRA ensembles](https://arxiv.org/abs/2310.00035)) — average their calibrated letter distributions for the final submission instead of retraining on full dev (keeps an honest ensemble and squeezes variance).

### 2.7 What we deliberately do NOT do

- **No retrieval/RAG at inference.** Track 1 permits "prompting, fine-tuning, preference optimization, and routing"; retrieval systems are only enumerated under Track 2, and "agents/tools" are banned in Track 1. In-corpus few-shot retrieval is a gray zone — the same benefit is captured legally by SFT, so don't risk disqualification. (If organizers clarify it's allowed, kNN few-shot by value-taxonomy match is a cheap add-on.)
- **No uniform-prior debiasing** (would fight the genuinely skewed gold distributions).
- **No manual per-item intervention** (banned in both tracks).

---

## 3. Expected performance & where the points come from

| Component | ZH | ID | SI |
|---|---|---|---|
| Zero-shot Qwen3-8B + value-conditioned CoT (est.) | ~55–65% | ~45–55% | ~45–55% |
| + constrained scoring & permutation/prior calibration | +2–4 | +2–4 | +3–6 |
| + LoRA SFT w/ rationales & augmentation | +5–10 | +4–8 | +8–15 |
| + DPO/GRPO (ZH), soft labels (ID), decomposition (SI) | +2–5 | +3–6 | +4–8 |
| + self-consistency & fold-ensemble | +1–3 | +1–2 | +1–3 |

(Estimates; the ID ceiling is structurally limited by 2/5-agreement items — treat ~75–80% as a realistic ceiling there. The SI column has the widest spread and the highest macro-leverage.)

**Sanity floors to beat:** ZH majority-class 37.2%, ID always-C 36.9%, SI always-A 52.2%. Any configuration below these on a CV fold is broken.

---

## 4. Execution plan

1. **Week 1 — harness + baselines.** Build the eval harness (JSONL in → predictions.jsonl out, CV splitter, per-country accuracy + macro). Zero-shot benchmark of the 4–5 candidate ≤8B models across prompt variants (native vs English instructions; with/without value hint; direct vs CoT). Lock the base model.
2. **Week 2 — SFT.** Rationale bootstrapping, augmentation pipelines, LoRA training with 5-fold CV, permutation augmentation. Lock prompt templates.
3. **Week 3 — country-specific objectives.** DPO/GRPO for ZH, soft-label loss for ID, binary decomposition + translate-assist for SI. Calibration temperatures per country.
4. **Week 4 — ensembling + freeze.** Self-consistency, fold-ensemble, final CV report, generate submission on hidden test, write system description (base model, params, hardware, compute — required for Track 1).

**Compute:** everything fits on a single 24–48GB GPU (8B in bf16 + LoRA via QLoRA if 24GB; inference with vLLM). If you only have smaller local hardware, Kaggle/Colab T4/A100 tiers suffice for LoRA on 1–5k examples.

**Submission hygiene:** emit `0` (not "None") for Sri Lankan "neither"; one `predictions.jsonl`, zipped; every hidden-test ID covered; labels restricted to the legal sets.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Hidden test lacks the ZH longest-option artifact | Never rely on length; check that CV accuracy holds when evaluating on length-controlled subsets |
| Overfitting 1.3k dev items | 5-fold CV for all decisions; fold-ensemble instead of full-dev retrain; early stopping on fold loss |
| Sinhala comprehension failure of chosen base | Measured in week 1; fallback = bilingual prompting with NLLB gloss + heavier SI augmentation |
| ID disagreement noise | Soft-label objective + accept the structural ceiling; don't chase fold noise |
| GRPO reward hacking (length) | Monitor length-of-chosen-option statistics of the policy vs gold |
| Rules ambiguity (retrieval, MoE, translation model) | Ask organizers; system as specified needs none of the ambiguous components except NLLB (≤8B, open — but confirm "all language-generating components" interpretation) |

---

## 6. Key references

- Sorensen et al., *A Roadmap to Pluralistic Alignment*, ICML 2024 — [arXiv:2402.05070](https://arxiv.org/abs/2402.05070)
- Feng et al., *Modular Pluralism*, EMNLP 2024 — multi-model pluralism (Track-2 flavored; motivates per-country specialization)
- Li et al., *CultureLLM*, NeurIPS 2024 — [arXiv:2402.10946](https://arxiv.org/abs/2402.10946) (semantic augmentation from small seeds)
- Zheng et al., *LLMs Are Not Robust Multiple Choice Selectors*, ICLR 2024 — [arXiv:2309.03882](https://arxiv.org/abs/2309.03882) (selection bias, PriDe)
- Wang et al., *Look at the Text*, 2024 — [arXiv:2404.08382](https://arxiv.org/abs/2404.08382) (text-answer robustness)
- Plank, *The 'Problem' of Human Label Variation*, EMNLP 2022 — [arXiv:2211.02570](https://arxiv.org/abs/2211.02570)
- Grimminger et al., SeedBERT — [arXiv:2211.13196](https://arxiv.org/abs/2211.13196); LeWiDi-2025 — [arXiv:2510.12516](https://arxiv.org/abs/2510.12516)
- Zelikman et al., *STaR* — [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- Wang et al., *Self-Consistency* — [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- Rafailov et al., *DPO* — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290); Shao et al., *GRPO/DeepSeekMath* — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- *LoRA Ensembles* — [arXiv:2310.00035](https://arxiv.org/abs/2310.00035)
- *SinhalaMMLU* — [arXiv:2509.03162](https://arxiv.org/abs/2509.03162)
- Qwen3 Technical Report — [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
- NLLB Team, *No Language Left Behind* — [arXiv:2207.04672](https://arxiv.org/abs/2207.04672)
- *Scaling Test-time Compute for Low-resource Languages / English-pivoted CoT* — [arXiv:2504.02890](https://arxiv.org/abs/2504.02890)
