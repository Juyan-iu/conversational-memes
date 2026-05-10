# 01_collection — Meme Reply Collection

> **Prerequisites:** Before running the collection pipeline, train the meme classifier first.
> See [`02_classifier/`](../02_meme_classification/README.md) for setup and training instructions.

This module collects meme reply posts from Bluesky firehose archives using a CLIP-based meme classifier.

## Overview

`meme_pipeline.py` scans daily Bluesky firehose archives, identifies image replies that are likely memes (via a fine-tuned CLIP classifier), and saves structured records for downstream labeling and benchmark construction.

## Directory Structure

```
01_collection/
├── meme_pipeline.py          # Main collection pipeline
├── requirements.txt          # Python dependencies
├── meme_dataset/             # Collected records (contents not tracked by git)
├── meme_dataset_24_06/       # Collection run: Jun 2024
└── meme_dataset_25_02/       # Collection run: Feb 2025
```

Each dataset folder follows this structure:
```
meme_dataset_*/
├── records/
│   └── <uid>.json            # One JSON file per collected meme reply
└── stats.json                # Per-date collection statistics
```

Each `<uid>.json` record contains:
```json
{
  "uid": "bsky_<did_suffix>_<rkey>",
  "uri": "at://did:.../app.bsky.feed.post/...",
  "original_post": { "uri", "text", "images", "created_at", ... },
  "parent_reply":  { "uri", "text", "images", ... },
  "meme_reply":    { "uri", "text", "images", "like_count", ... },
  "ancestor_chain": [...],
  "quoted_post":   { ... }
}
```

## Requirements

### System

- Python 3.12+
- CUDA-capable GPU (A100 recommended; CLIP inference)
- Access to Bluesky firehose archives (gzipped JSONL, one file per day)
- Fine-tuned CLIP meme classifier checkpoint

### Archive Structure

```
/path/to/firehose_archives/
├── 2023-09/
│   ├── 2023-09-01.json.gz
│   ├── 2023-09-02.json.gz
│   └── ...
└── 2024-06/
    └── ...
```

## Setup

```bash
# Create environment
python -m venv collect_env
source collect_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `CONFIG` at the top of `meme_pipeline.py`:

```python
CONFIG = {
    "archive_base": "/path/to/firehose_archives",  # Bluesky firehose archive root
    "model_dir":    "/path/to/checkpoints",        # CLIP classifier checkpoint
    "output_dir":   "./meme_dataset",              # Output directory
    "clip_name":    "openai/clip-vit-large-patch14-336",
    "meme_threshold": 0.5,
}
```

## Usage

```bash
source collect_env/bin/activate

# Preview a single date (no output written)
python meme_pipeline.py --preview --date 2023-09-01

# Run a single date
python meme_pipeline.py --run --date 2023-09-01

# Run a full month
python meme_pipeline.py --run --month 2023-09

# Run a date range
python meme_pipeline.py --run --start 2023-09-01 --end 2025-08-31
```

## Output

- One `.json` file per meme reply under `records/`
- `stats.json` with per-date counts (total posts, image replies, memes detected, meme ratio)

## Notes

- The `records/` directories are **not tracked by git** (see `.gitignore`). Only the folder structure is committed.
- The pipeline uses a TID-based date estimation to locate root posts in the firehose archives without full scans.
- Image downloads use CDN URLs (`cdn.bsky.app`). Posts deleted after collection will have inaccessible images.
