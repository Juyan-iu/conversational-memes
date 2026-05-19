# 01_collection - Meme Reply Collection

This folder contains the original archive-based collection pipeline used to
identify meme replies in Bluesky firehose archives and construct conversation
context records.

For public reproducibility, there are two supported paths:

1. Re-run this folder against local Bluesky firehose archives and the released
   MemeTector checkpoint.
2. Use the released UID manifest with the lightweight hydration scripts in
   [`../download`](../download/README.md). This reproduces the same benchmark
   records by UID without re-running the classifier.

## What This Stage Does

`meme_pipeline.py` scans daily firehose archive files, keeps English image
replies, classifies each image with MemeTector v4, and writes one JSON record per
detected meme reply.

Each output record is keyed by the meme reply UID:

```text
bsky_<last_14_chars_of_did>_<rkey>
```

The UID is stable inside this project, but it is not reversible by itself because
it stores only the DID suffix. Any public reproducibility file should therefore
store both `uid` and the full `at://...` URI.

## Directory Layout

```text
01_collection/
├── meme_pipeline.py              # Archive scan, image download, CLIP inference
├── export_uid_manifest.py        # Export public UID/URI manifest from records
├── requirements.txt
├── meme_dataset/                 # Collection run output; records are not tracked
├── meme_dataset_24_06/
└── meme_dataset_25_02/
```

Each dataset folder follows this layout:

```text
meme_dataset_*/
├── records/
│   └── <uid>.json
├── images/
└── stats.json
```

## Record Schema

The final record used downstream is a JSON object with the following main
fields. `meme_pipeline.py` creates the meme reply, root post, parent reply, and
comparison fields from the archive. If context enrichment has been run, the same
record may also include `quoted_post`; the public hydration script in
`../download` reconstructs these fields from URI references.

```json
{
  "uid": "bsky_<did_suffix>_<rkey>",
  "uri": "at://did:.../app.bsky.feed.post/...",
  "meme_prob": 0.91,
  "is_meme": true,
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
  "parent_reply": {
    "uri": "...",
    "text": "...",
    "images": []
  },
  "quoted_post": null,
  "thread_structure": {
    "depth": 1,
    "label": "reply"
  },
  "comparison_reply": {
    "uri": "...",
    "selected_by": "timely_structural"
  }
}
```

`parent_reply` and `quoted_post` depend on whether the corresponding public
archive/API content is still available. The published dataset statistics should
report structural reply depth separately from context availability.

## Requirements

- Python 3.12+
- CUDA-capable GPU for full archive collection
- Bluesky firehose archives in daily `.json.gz` files
- MemeTector v4 checkpoints from [`../02_meme_classification`](../02_meme_classification/README.md)

Install dependencies:

```bash
python -m venv collect_env
source collect_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Configure Collection

Edit `CONFIG` at the top of `meme_pipeline.py`:

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

## Run Collection

```bash
# Preview one day without writing records
python meme_pipeline.py --preview --date 2024-06-01

# Run one day
python meme_pipeline.py --run --date 2024-06-01

# Run one month
python meme_pipeline.py --run --month 2024-06

# Run a date range
python meme_pipeline.py --run --start 2024-06-01 --end 2025-02-28
```

## Export the Public UID Manifest

After collection and any context patching are complete, export the manifest used
by the public download pipeline. Exporting from all collection record folders
creates the upstream candidate-pool manifest. In the release described here,
this pool contains 119,848 candidate meme replies before downstream validation,
labeling, and benchmark mapping.

```bash
python export_uid_manifest.py \
  --records-dir meme_dataset_24_06/records \
  --records-dir meme_dataset_25_02/records \
  --records-dir meme_dataset/records \
  --out ../download/data/meme_reply_uid_manifest.jsonl
```

Small sample export:

```bash
python export_uid_manifest.py \
  --records-dir meme_dataset_24_06/records \
  --out ../download/data/sample_uid_manifest.jsonl \
  --limit 25
```

The manifest intentionally contains only identifiers and URI pointers. Later
stages should map classifier metadata, validation labels, benchmark splits, and
provided distractors back onto records by `uid`.

If you need a manifest for only the final labeled dataset, export from the
post-filtering JSONL instead:

```bash
python export_uid_manifest.py \
  --input ../03_filter_and_label/labeled_final/labeled_memes.jsonl \
  --out ../download/data/validated_dataset_uid_manifest.jsonl
```

## Manifest Fields

Each JSONL row contains enough information to hydrate the same public record:

```json
{
  "uid": "bsky_<did_suffix>_<rkey>",
  "uri": "at://did:.../app.bsky.feed.post/...",
  "meme_reply_uri": "at://did:.../app.bsky.feed.post/...",
  "root_post_uri": "at://did:.../app.bsky.feed.post/...",
  "reply_parent_uri": "at://did:.../app.bsky.feed.post/...",
  "parent_reply_uri": null,
  "quoted_post_uri": null,
  "best_reply_before_meme_uri": null,
  "closest_text_reply_uri": null,
  "closest_sibling_text_reply_uri": null,
  "comparison_reply_uri": "at://did:.../app.bsky.feed.post/...",
  "thread_depth": 1,
  "thread_label": "reply"
}
```

`reply_parent_uri` is the direct AT Protocol parent reference. For first-level
replies it usually equals `root_post_uri`. `parent_reply_uri` is populated only
when the direct parent is itself a reply.

## Reproducibility Notes

- Full archive collection is needed only to regenerate meme candidates from
  scratch. Public users can start from the UID manifest in `../download`.
- The public API can no longer return deleted or unavailable posts/images. The
  hydration scripts keep the UID and URI and record failures in a report.
- Percentages for structural reply depth should use the full record count as the
  denominator. Parent-content availability should be reported separately among
  structurally nested replies.
- Original meme images, root posts, parent replies, and quoted posts are
  hydrated by URI. Labels and benchmark metadata should be joined later by `uid`.
