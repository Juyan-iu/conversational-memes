# 03_filter_and_label - Candidate Validation and Labeling

This folder contains the validation, meme-level labeling, cleanup, and refill
pipeline used after candidate meme replies are collected in
[`../01_collection`](../01_collection/README.md).

The public reproducibility workflow keeps the scripts needed to rebuild the
curated candidate pool. Legacy annotation resources from earlier experiments
are intentionally omitted because those labels are not part of the paper's
released task.

## What This Stage Does

`label_pipeline.py` reads candidate records from stage 01, validates that each
candidate image is a meme, downloads/cache media needed for annotation, and
writes one JSONL row per accepted record. For each accepted candidate, it adds
the meme validation result, downloaded image paths, stance attributes, and a
visual description.

`postprocess.py` checks the labeled output, removes clear non-English or
mass-mention records when requested, and writes `shortage.json` for months that
fall below the target count.

`refill_pipeline.py` reads `shortage.json`, samples additional candidates from
the original collection folders, and appends replacement records to the same
output directory using the same validation and labeling functions.

## Files

```text
03_filter_and_label/
├── label_pipeline.py                  # Main validation and labeling pipeline
├── postprocess.py                     # Language/mass-mention cleanup and shortage report
├── refill_pipeline.py                 # Monthly shortage refill pipeline
├── requirements.txt
└── README.md
```

## Input

By default, `label_pipeline.py` reads candidate records from:

```python
CONFIG = {
    "input_dirs": [
        "../01_collection/meme_dataset_24_06",
        "../01_collection/meme_dataset_25_02",
        "../01_collection/meme_dataset",
    ],
    "output_dir": "./labeled_dataset",
}
```

You can override the input folders with `--input`.

Expected input layout:

```text
meme_dataset_*/
├── records/
│   └── <uid>.json
└── index_YYYY-MM-DD.jsonl
```

## Requirements

- Python 3.10+
- OpenAI API key
- Candidate records from stage 01

Install dependencies:

```bash
python -m venv label_env
source label_env/bin/activate
pip install -r requirements.txt
```

Set your API key:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

## Run

Run the monthly balanced labeling job:

```bash
python label_pipeline.py \
  --monthly-total 20400 \
  --output ./labeled_final
```

Run a small sample:

```bash
python label_pipeline.py \
  --sample 50 \
  --output ./labeled_sample
```

Useful options:

```text
--input DIR [DIR ...]       Override candidate dataset folders.
--monthly-total N           Target total accepted records across available months.
--sample N                  Process a random sample when not using --all or --monthly-total.
--all                       Process all loaded records.
--model MODEL               Override main and visual model together.
--model-visual MODEL        Override only the visual model.
--uid-file PATH             Process only UIDs listed in a text file.
--save-uids PATH            Save sampled UIDs for reproducible reruns.
```

## Postprocess

Check status without modifying files:

```bash
python postprocess.py \
  --input ./labeled_final/labeled_memes.jsonl
```

Create a cleaned file and `shortage.json`:

```bash
python postprocess.py \
  --input ./labeled_final/labeled_memes.jsonl \
  --clean
```

Replace the original JSONL with the cleaned JSONL:

```bash
python postprocess.py \
  --input ./labeled_final/labeled_memes.jsonl \
  --clean --replace
```

## Refill

Use `refill_pipeline.py` after `postprocess.py` creates `shortage.json`:

```bash
python refill_pipeline.py \
  --shortage ./labeled_final/shortage.json \
  --output ./labeled_final
```

## Output

```text
labeled_final/
├── labeled_memes.jsonl
├── shortage.json
├── records/
│   └── <uid>.json
└── images/
```

Each accepted record preserves the original stage-01 fields and adds curation
metadata such as:

```json
{
  "uid": "...",
  "meme_validation": {
    "passed": true,
    "valid_ratio": 1.0,
    "validations": []
  },
  "downloaded_images": {
    "meme_reply": []
  },
  "meme_annotation": {
    "stance_labels": {
      "sarcastic": false,
      "humorous": true,
      "offensive": false
    },
    "visual_description": "..."
  },
  "stance_labels": {
    "sarcastic": false,
    "humorous": true,
    "offensive": false
  },
  "visual_description": "...",
  "labeled_at": "2025-..."
}
```

## Notes

- `postprocess.py` uses conservative language filtering: short text, slang, and
  emoji-heavy posts are usually kept, while clear non-English records and
  mass-mention posts are removed.
- `refill_pipeline.py` reuses the same validation and labeling functions as
  `label_pipeline.py`.
