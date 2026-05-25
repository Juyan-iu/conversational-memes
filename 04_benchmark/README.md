# 04_benchmark - Multiple-Choice Benchmark Construction

This folder contains the benchmark-construction pipeline used after candidate
meme replies are collected and labeled in
[`../03_filter_and_label`](../03_filter_and_label/README.md).

## What This Stage Does

`benchmark_pipeline.py` reads labeled meme-reply records, checks that required
posts and images are still available, extracts meme captions with OCR, and
creates one four-option visual question per accepted meme reply.

Each benchmark item contains:

```text
A) Original meme          # correct answer
B) Text distractor        # original caption removed and replaced with context keywords
C) Visual distractor      # similar non-meme image with the original caption inserted
D) Easy distractor        # random meme image
```

The pipeline supports balanced monthly top-liked sampling, CLIP-based visual
distractor search, LaMa inpainting, and resume-safe output appending.

## Files

```text
04_benchmark/
|-- benchmark_pipeline.py   # Benchmark item and distractor generation
|-- requirements.txt
`-- README.md
```

Generated outputs are written under the configured `output_dir` and are not
tracked in git:

```text
benchmark_data/
|-- benchmark_summary.jsonl
|-- pool_embeddings.npy
|-- pool_items.json
`-- <uid>/
    |-- A_original.jpg
    |-- B_text_distractor.jpg
    |-- C_visual_distractor.jpg
    |-- D_easy_distractor.jpg
    `-- meta.json
```

## Input

By default, `benchmark_pipeline.py` reads the labeled records from stage 03 and
image records from stage 01:

```python
CONFIG = {
    "labeled_jsonl": "../03_filter_and_label/labeled_final/labeled_memes.jsonl",
    "dataset_dirs": [
        "../01_collection/meme_dataset_24_06",
        "../01_collection/meme_dataset_25_02",
        "../01_collection/meme_dataset",
    ],
    "output_dir": "./benchmark_data",
    "date_from": "2023-09",
    "date_to": "2025-08",
}
```

Expected stage-01 input layout:

```text
meme_dataset_*/
|-- records/
|   `-- <uid>.json
`-- meme_images/
```

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended for full-scale runs
- Labeled records from stage 03
- Candidate image records from stage 01
- Network access for CDN availability checks and fallback image loading

Install dependencies:

```bash
python -m venv bech_env
source bech_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install simple-lama-inpainting
pip install -r requirements.txt
```

`requirements.txt` pins `Pillow<10` because `simple-lama-inpainting` expects a
Pillow 9.x-compatible environment.

## Run

Run a small test sample:

```bash
python benchmark_pipeline.py --sample 10
```

Build a balanced monthly benchmark with a target size:

```bash
python benchmark_pipeline.py --target 5000 --workers 4
```

Run one month:

```bash
python benchmark_pipeline.py --month 2023-09
```

Run all eligible labeled records:

```bash
python benchmark_pipeline.py --all
```

Useful options:

```text
--sample N       Process the top-liked N records for a test run.
--target N       Build a balanced monthly benchmark with N total target items.
--month YYYY-MM  Process one month only.
--all            Process all loaded records in descending like-count order.
--output DIR     Override the benchmark output directory.
--no-clip        Skip CLIP pool building; mainly useful for debugging.
--workers N      Set the parallel worker count.
--lama-res N     Override LaMa resolution, such as 256, 512, or 768.
```

## Output Schema

Each accepted item writes image options and a `meta.json` file:

```json
{
  "uid": "bsky_<did_suffix>_<rkey>",
  "month": "2024-06",
  "orig_url": "https://...",
  "captions": [
    {
      "text": "...",
      "bbox": [10, 20, 300, 80],
      "conf": 0.91,
      "position": "top"
    }
  ],
  "options": {
    "A": "A_original.jpg",
    "B": "B_text_distractor.jpg",
    "C": "C_visual_distractor.jpg",
    "D": "D_easy_distractor.jpg"
  },
  "answer": "A",
  "context": {
    "original_post_text": "...",
    "parent_reply_text": "...",
    "meme_text": "..."
  },
  "labels": {}
}
```

`benchmark_summary.jsonl` appends one metadata row per successful item. Existing
UIDs in that file are skipped on later runs, so interrupted jobs can be resumed
with the same output directory.

## Notes

- Public post and CDN image availability can change over time, especially for
  deleted or restricted content.
- `pool_embeddings.npy` and `pool_items.json` cache CLIP embeddings for the
  non-meme image pool. Delete both files if the source pool changes and you want
  to rebuild the cache.
- `--target` interleaves months after sorting each month by meme-reply like
  count, then stops adding records from months whose quota has already been met.

