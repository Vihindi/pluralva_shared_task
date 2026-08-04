# PlurVA-LLM Shared Task

The main end-to-end workflow is provided in
[`Plurvallm_main.ipynb`](Plurvallm_main.ipynb). Open the notebook in Google
Colab and run its cells from top to bottom to build the SFT data, train the
shared LoRA adapter, generate predictions, and package the submission.

## Clone the repository

```bash
git clone https://github.com/Vihindi/pluralva_shared_task.git
cd pluralva_shared_task
```

For Google Colab, select a GPU runtime before running the notebook. The Llama
3.1 checkpoint is gated, so accept its Hugging Face access conditions and add
your token to Colab Secrets using the name `HF_LLAMA_TOKEN`. Do not place the
token directly in the notebook.

## Install the training dependencies

```bash
pip install -U "bitsandbytes>=0.46.1" transformers peft datasets accelerate
```

The notebook mounts Google Drive and writes model checkpoints and submissions
under `/content/drive/MyDrive/pluralva_outputs/` so they survive a Colab runtime
disconnect.

## 1. Build the SFT data

Run this command from the repository root:

```bash
python src/build_sft_data.py \
  --processed_dir processed \
  --out_dir sft_data \
  --si_mode binary \
  --si_aux_files processed_mmlu_aux/sri_lankan.jsonl \
  --no_value_summaries
```

This command:

- creates four cyclic option orders for the Chinese and Indonesian examples;
- builds binary `Yes`/`No` Sinhala examples;
- adds the Sinhala MMLU auxiliary records from `processed_mmlu_aux`;
- disables value-summary injection to keep the training and prediction prompts
  consistent; and
- writes full-data and five-fold SFT files to `sft_data/`.

Rationales are disabled because `--rationales` is not supplied.

## 2. Train the shared LoRA adapter

```bash
python src/train_lora.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --train_files \
    sft_data/zh_train_full.jsonl \
    sft_data/id_train_full.jsonl \
    sft_data/si_train_full.jsonl \
  --output_dir /content/drive/MyDrive/pluralva_outputs/Joint_Adapter \
  --epochs 2 \
  --load_4bit
```

The current defaults use LoRA rank 16, alpha 32, dropout 0.05, learning rate
`1e-4`, physical batch size 2, gradient accumulation 10, and maximum sequence
length 1,536. Mixed-country batching is disabled by default, so rows are
normally shuffled. Add `--resume` to continue from the newest checkpoint in
the output directory.

## 3. Generate the initial submission

```bash
python src/make_submission.py predict \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapter /content/drive/MyDrive/pluralva_outputs/Joint_Adapter \
  --n_perms 4 \
  --prior_tau 0.5 \
  --th_a 0.5 \
  --th_b 0.5 \
  --no_value_summaries \
  --load_4bit \
  --out_dir /content/drive/MyDrive/pluralva_outputs/Submission_predictions
```

Prediction is resumable. Raw probabilities are written after every item to
`Submission_predictions/test_details.jsonl`. Rerunning the same command skips
completed IDs. The output directory also contains `predictions.jsonl` and
`predictions.zip`.

## 4. Apply conditional Sinhala calibration

Calibration changes only the final binary Sinhala label composition. It does
not rerun the model and does not affect Chinese or Indonesian predictions.

```bash
python src/make_submission.py compose \
  --details /content/drive/MyDrive/pluralva_outputs/Submission_predictions/test_details.jsonl \
  --th_b 0.49 \
  --th_a_if_b_no 0.24 \
  --th_a_if_b_yes 0.20 \
  --out_dir /content/drive/MyDrive/pluralva_outputs/final_calibrated_submission
```


## Repository layout

| Path | Purpose |
|---|---|
| `Plurvallm_main.ipynb` | Main Colab workflow |
| `processed/` | Processed Chinese, Indonesian, and Sinhala source data |
| `processed_mmlu_aux/` | Sinhala MMLU auxiliary records |
| `sft_data/` | Generated full-data and fold-specific SFT files |
| `src/build_sft_data.py` | Builds chat-formatted SFT datasets |
| `src/train_lora.py` | Trains a shared or monolingual LoRA adapter |
| `src/make_submission.py` | Predicts, composes, validates, and zips submissions |
| `src/tune_conditional_calibration.py` | Searches conditional Sinhala thresholds |
| `PlurVA-LLM_Test_Set/` | Official test inputs used during prediction |

