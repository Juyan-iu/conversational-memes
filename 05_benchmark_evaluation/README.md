# 05_benchmark_evaluation - VLM Benchmark Evaluation

This folder contains the evaluation pipeline used to run multimodal language
models on the meme-reply selection benchmark produced in
[`../04_benchmark`](../04_benchmark/README.md).

## What This Stage Does

`vlm_eval_pipeline.py` converts `benchmark_data` into the local VLM-eval item
format, loads each multiple-choice meme selection item, runs one or more model
runners, and writes prediction files plus summary metrics.

For each benchmark item, the model receives:

- the conversation context
- any available inline context images from the thread
- four candidate meme images labeled `A`, `B`, `C`, and `D`

The task is to select the meme image that was actually posted as the reply. The
three distractors are typed as lexical, visual, and random distractors.

## Files

```text
05_benchmark_evaluation/
├── vlm_eval_pipeline.py          # Convert benchmark data and run evaluations
├── runners/
│   ├── base.py                   # Dataset loading, prompts, image helpers
│   ├── openai_runner.py          # GPT-4o
│   ├── anthropic_runner.py       # Claude Sonnet 4.5
│   ├── google_runner.py          # Gemini 2.5 Pro
│   ├── together_runner.py        # Llama 4 Scout via OpenRouter
│   ├── reallms_runner.py         # IU REALLMS endpoint
│   ├── hf_runner_qwen.py         # Qwen2.5-VL-7B
│   ├── hf_runner_omni.py         # Qwen2.5-Omni-7B
│   ├── hf_runner_internvl.py     # InternVL3-8B
│   ├── hf_runner_qvq.py          # QVQ-72B-Preview
│   └── hf_runner_qwen3vl32b.py   # Qwen3-VL-32B
└── README.md
```

Generated outputs are written under local `data/`, `results/`, and optionally
`reports/` folders:

```text
data/vlmeval_converted/
└── <uid>/
    ├── info.txt
    ├── <uid>.jpg
    ├── B_text_distractor.jpg
    ├── C_visual_distractor.jpg
    ├── D_easy_distractor.jpg
    ├── labels.json
    ├── discourse.json
    └── context_images.json

results/
└── <run_id>/
    ├── <model>_predictions.jsonl
    └── summary.json
```

Circular runs write one raw prediction file per rotation plus an aggregated
summary:

```text
results/<run_id>/
├── <model>_circular_run0.jsonl
├── <model>_circular_run1.jsonl
├── <model>_circular_run2.jsonl
├── <model>_circular_run3.jsonl
├── <model>_circular_summary.jsonl
└── summary.json
```

## Converted Item Schema

Each converted item is a folder named by UID. The correct meme image must have
the same stem as the folder name:

```text
<uid>/
├── info.txt                 # Conversation text with inline [IMAGE:N] markers
├── <uid>.jpg                # Correct meme reply
├── B_text_distractor.jpg    # Lexical distractor
├── C_visual_distractor.jpg  # Visual distractor
├── D_easy_distractor.jpg    # Random/easy distractor
├── labels.json              # Distractor type labels
├── discourse.json           # Discourse and stance labels
└── context_images.json      # CDN URLs matched to [IMAGE:N] markers
```

`labels.json` maps distractor filenames to their distractor type:

```json
{
  "B_text_distractor.jpg": "lexical",
  "C_visual_distractor.jpg": "visual",
  "D_easy_distractor.jpg": "random"
}
```

Each prediction row contains the gold answer, model prediction, correctness,
and the distractor type selected when the model is wrong:

```json
{
  "id": "bsky_<did_suffix>_<rkey>",
  "gold": "C",
  "pred": "A",
  "correct": false,
  "picked_slot_type": "lexical",
  "slot_type_by_letter": {
    "A": "lexical",
    "B": "random",
    "C": "correct",
    "D": "visual"
  },
  "gold_filename": "bsky_<did_suffix>_<rkey>.jpg",
  "error": null
}
```

## Requirements

- Python 3.12+
- Benchmark data produced by
  [`../04_benchmark`](../04_benchmark/README.md)
- API keys for closed/API-hosted models
- CUDA-capable GPU for local Hugging Face models

If `ben_eval_env` already exists at the repository root, activate it before
running this stage:

```bash
source ben_eval_env/bin/activate
cd 05_benchmark_evaluation
```

If you need to create the environment from scratch:

```bash
python -m venv ben_eval_env
source ben_eval_env/bin/activate
pip install python-dotenv tqdm pillow openai anthropic google-genai
```

For local Hugging Face models, also install the GPU stack required by the model
you plan to run:

```bash
pip install torch torchvision transformers accelerate qwen-vl-utils
pip install qwen-omni-utils bitsandbytes
```

## Configure

Add API keys to `05_benchmark_evaluation/.env` or export them in the shell:

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
REALLMS_API_KEY=...
```

Available model keys:

```text
API / hosted:
  gpt-4o
  claude-sonnet-4-5
  gemini-2.5-pro
  llama-4-scout
  reallms

Local HF:
  qwen25-vl-7b
  qwen25-omni-7b
  internvl3-8b
  qvq-72b
  qwen3-vl-32b

Special:
  all
```

Optional local-model environment variables:

```text
QVQ_USE_4BIT=1             # Load QVQ-72B in 4-bit mode
QVQ_MAX_NEW_TOKENS=4096    # Generation budget for QVQ
QWEN_OMNI_USE_SC=1         # Enable self-consistency for Qwen2.5-Omni
QWEN3VL_THINKING=0         # Disable Qwen3-VL thinking mode
QWEN3VL_MAX_TOKENS=2048    # Generation budget for Qwen3-VL
```

## Run

From `05_benchmark_evaluation/`, convert benchmark data and run a small smoke
test:

```bash
python vlm_eval_pipeline.py \
  --data-root ../04_benchmark/benchmark_data \
  --model gpt-4o \
  --num 3
```

Run one full API-model evaluation:

```bash
python vlm_eval_pipeline.py \
  --data-root ../04_benchmark/benchmark_data \
  --model gpt-4o \
  --run-id gpt4o_full
```

Run multiple models in one command:

```bash
python vlm_eval_pipeline.py \
  --data-root ../04_benchmark/benchmark_data \
  --model gpt-4o claude-sonnet-4-5 gemini-2.5-pro
```

Run local models:

```bash
python vlm_eval_pipeline.py \
  --data-root ../04_benchmark/benchmark_data \
  --model qwen25-vl-7b internvl3-8b
```

Reuse already converted data:

```bash
python vlm_eval_pipeline.py \
  --skip-convert \
  --model gpt-4o \
  --num 10
```

Run circular evaluation, which rotates the correct answer through all four
positions to reduce position bias:

```bash
python vlm_eval_pipeline.py \
  --skip-convert \
  --model qwen25-vl-7b \
  --circular
```

Useful options:

```text
--model MODEL [MODEL ...]   Model key(s) to run. Use all to run every runner.
--data-root PATH            Source benchmark_data directory to convert.
--skip-convert              Use data/vlmeval_converted without reconverting.
--uid-list PATH             Evaluate only UIDs listed one per line.
--num N                     Limit to the first N items.
--run-id NAME               Custom output folder name.
--seed N                    Seed for A/B/C/D assignment.
--circular                  Run each item four times with answer rotation.
--prompt-version conv       Use full conversation text. This is the default.
--prompt-version disc       Use discourse labels instead of raw conversation.
--report                    Generate reports if report.py is available locally.
```

## Notes

- Candidate images are resized to 512 px on the longest edge and re-encoded as
  JPEG before being sent to model APIs or local processors.
- `--num` limits conversion when `--data-root` is provided. With
  `--skip-convert`, it limits the already converted item list.
- Context images are fetched from CDN URLs during evaluation. Deleted,
  restricted, or unavailable posts may appear as unavailable context images.
- API runners retry transient rate-limit and server errors. Permanent failures
  are recorded in the prediction row's `error` field.
- Circular evaluation costs roughly four times more compute/API calls than the
  standard evaluation.

