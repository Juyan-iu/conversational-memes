# 01_collection - Meme Reply Collection

This folder contains the archive-based collection pipeline used to reproduce the
candidate meme-reply collection stage from local Bluesky firehose archives.

## What This Stage Does

`meme_pipeline.py` scans daily Bluesky firehose archive files, keeps English
single-image replies, classifies each image with MemeTector v4, and writes one
JSON record per detected meme reply.

Each output record is keyed by the meme reply UID:

```text
bsky_<last_14_chars_of_did>_<rkey>
```

## Files

```text
01_collection/
├── meme_pipeline.py      # Archive scan, image download, CLIP inference
├── requirements.txt
└── README.md
```

Generated outputs are written under the configured `output_dir` and are not
tracked in git:

```text
meme_dataset/
├── meme_images/
│   └── <uid>/<cid>.jpg
├── original_post_images/
├── records/
│   └── <uid>.json
├── index_YYYY-MM-DD.jsonl
├── meme_index.jsonl
├── all_memes.json
└── stats.json
```

## Record Schema

Each saved record contains the detected meme reply, available conversation
context, the selected text comparison reply, and collection metadata:

```json
{
  "uid": "bsky_<did_suffix>_<rkey>",
  "uri": "at://did:.../app.bsky.feed.post/...",
  "meme_prob": 0.91,
  "is_meme": true,
  "threshold": 0.5,
  "meme_reply": {
    "uid": "...",
    "uri": "...",
    "text": "...",
    "images": [{"cid": "...", "local_path": "...", "source_url": "..."}],
    "root_uri": "...",
    "parent_uri": "..."
  },
  "original_post": {
    "uri": "...",
    "text": "...",
    "images": []
  },
  "parent_reply": null,
  "thread_structure": {
    "depth": 1,
    "label": "reply"
  },
  "best_reply_before_meme": null,
  "closest_text_reply": null,
  "closest_sibling_text_reply": null,
  "comparison_reply": {
    "uri": "...",
    "selected_by": "timely_structural"
  },
  "source_file": "2024-06-01.json.gz",
  "process_date": "2024-06-01"
}
```

Context fields may be `null` when the corresponding post is not available in the
local archive window used for lookup.

## Requirements

- Python 3.12+
- CUDA-capable GPU for full-scale collection
- Bluesky firehose archives in daily `.json.gz` files
- MemeTector v4 checkpoints prepared in
  [`../02_meme_classification`](../02_meme_classification/README.md)

Before running this collection stage, go to `../02_meme_classification` and
either train the MemeTector classifier or download the released checkpoints.
Then set `--model-dir` to the directory containing the fold checkpoints
(`fold1_best.pth` through `fold5_best.pth`).

Install dependencies:

```bash
python -m venv collect_env
source collect_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Configure

You can edit `CONFIG` at the top of `meme_pipeline.py` or override common paths
and parameters from the command line.

```python
CONFIG = {
    "archive_base": "/path/to/firehose_archives",
    "model_dir": "/path/to/memetector_v4/checkpoints",
    "output_dir": "./meme_dataset",
    "clip_name": "openai/clip-vit-large-patch14-336",
    "meme_threshold": 0.5,
    "lookup_max_days_back": 0,
    "lang_filter": ["en"],
}
```

Expected archive layout:

```text
/path/to/firehose_archives/
├── 2024-06/
│   ├── 2024-06-01.json.gz
│   └── ...
└── 2025-02/
    └── ...
```

## Run

```bash
python meme_pipeline.py --run --date 2024-06-01 \
  --archive-base /path/to/firehose_archives \
  --model-dir /path/to/memetector_v4/checkpoints \
  --output-dir ./meme_dataset
```

Useful options:

```text
--preview          Preview archive parsing without writing records.
--month YYYY-MM    Run a full month instead of one day.
--start / --end    Run a date range instead of one day.
--monthly-quota N   Stop after N detected memes per month. Use 0 for no quota.
--daily-sample N    Randomly sample at most N image replies per day.
--lookup-days N     Search up to N earlier archive days for root/parent context.
--lang all          Disable the default English-language filter.
--single-fold N     Use one MemeTector fold instead of the 5-fold ensemble.
```

## Notes

- Public post/image availability can change over time, especially for deleted or
  restricted content.
- Thread-depth percentages should use the full collected record count as the
  denominator. Context availability should be reported separately.
