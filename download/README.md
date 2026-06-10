# download - Dataset and Benchmark Access

This folder describes how to access the public MEMECONV artifacts without
re-running the full collection pipeline. The released artifacts are separated by
task:

| Artifact | Purpose | Path |
| --- | --- | --- |
| Full hydrated dataset | Reconstruct public posts, conversation context, and meme images by UID from the released manifest. | `download/hydrate_from_uid_manifest.py` |
| Multiple-choice benchmark | Download or assemble the fixed recognition benchmark with answer and distractor images for evaluation. | Released `benchmark_data/` archive and `download/map_benchmark_assets.py` |
| Generation task | Reproduce the context-conditioned meme-generation protocol. | `06_meme_generation/generate_memes_compare.py` |

The full dataset and the multiple-choice benchmark are intentionally separated.
Hydration recovers public Bluesky records and image media by UID; the benchmark
also includes constructed distractors and metadata that are task artifacts, not
posts that can be recovered from the public API.

## Full Dataset Hydration

Use this path when you want the broad released dataset: one hydrated JSON record
per UID, plus optional local image files. This step does not re-run the meme
classifier. Instead, it reads a UID manifest exported from `01_collection` and
hydrates the same posts and conversation context from the public Bluesky API.

Later stages can map classifier metadata, labels, benchmark membership, and
provided distractors back onto the hydrated records by `uid`.

### Why a Manifest Is Needed

Project UIDs look like this:

```text
bsky_<last_14_chars_of_did>_<rkey>
```

That UID is stable for joining labels and benchmark rows, but it is not enough
to call the Bluesky API because the full DID is truncated. The manifest must
therefore include both `uid` and the full `at://...` URI.

Create the manifest from the original collected records:

```bash
python download_manifest.py
```

This downloads `data/collection_pool_uid_manifest.jsonl` from Google Drive:

```text
https://drive.google.com/file/d/1z-4NvEMU5asI0j5IX9dMahbVzzNmZ_Gd/view?usp=sharing
```

The released manifest is the upstream candidate-pool manifest. In the release
described here, that pool contains 119,848 candidate meme replies before
downstream validation, labeling, and benchmark mapping.

To regenerate the manifest from local collection records instead:

```bash
cd ../01_collection
python export_uid_manifest.py \
  --records-dir meme_dataset_24_06/records \
  --records-dir meme_dataset_25_02/records \
  --records-dir meme_dataset/records \
  --out ../download/data/collection_pool_uid_manifest.jsonl
```

### Files

```text
download/
├── download_manifest.py           # Download released manifest from Google Drive
├── hydrate_from_uid_manifest.py   # Hydrate records and optional images by UID
├── map_benchmark_assets.py        # Join released labels to benchmark/distractor assets by UID
├── sample_uid_manifest.jsonl      # Tiny dry-run sample of the manifest format
└── data/                          # Put released manifests here; not required by git
```

### Manifest Format

Each JSONL row should contain:

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

Required fields are `uid` and either `meme_reply_uri` or `uri`. The other fields
are URI pointers or structural indexes that make context recovery more exact
when public thread traversal is incomplete.

### Setup

The hydration script uses only the Python standard library.

```bash
python --version  # Python 3.10+ recommended
```

No Bluesky login is required. The script calls:

```text
https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread
```

### Sample Runs

Validate the sample manifest without network calls:

```bash
python hydrate_from_uid_manifest.py \
  --manifest sample_uid_manifest.jsonl \
  --out sample_hydrated \
  --dry-run
```

Run a small real hydration job from a released manifest:

```bash
python download_manifest.py

python hydrate_from_uid_manifest.py \
  --manifest data/collection_pool_uid_manifest.jsonl \
  --out hydrated_sample \
  --limit 10 \
  --download-images meme
```

Resume or re-run a subset:

```bash
python hydrate_from_uid_manifest.py \
  --manifest data/collection_pool_uid_manifest.jsonl \
  --out hydrated_records \
  --offset 100 \
  --limit 100 \
  --download-images none
```

Download root/parent/quoted/comparison images as well as the meme image:

```bash
python hydrate_from_uid_manifest.py \
  --manifest data/collection_pool_uid_manifest.jsonl \
  --out hydrated_context_sample \
  --limit 10 \
  --download-images context
```

Hydrate one UID:

```bash
python hydrate_from_uid_manifest.py \
  --manifest data/collection_pool_uid_manifest.jsonl \
  --out hydrated_one \
  --uid bsky_example_3abc123 \
  --download-images all
```

### Output

```text
hydrated_records/
├── records/
│   └── <uid>.json
├── images/
│   ├── meme_reply/<uid>/*.jpg
│   ├── original_post/<uid>/*.jpg
│   ├── parent_reply/<uid>/*.jpg
│   └── ...
└── hydration_report.json
```

Each hydrated record contains:

- `meme_reply`: the correct answer post and its image metadata
- `original_post`: root post context
- `parent_reply`: direct parent reply when the meme reply is structurally nested
- `quoted_post`: embedded/quoted post from the root context, if available
- `best_reply_before_meme`: most-liked sibling reply before the meme reply, if
  present in the manifest
- `closest_text_reply`: temporally closest text reply in the root thread, if
  present in the manifest
- `closest_sibling_text_reply`: temporally closest text sibling under the same
  parent, if present in the manifest
- `comparison_reply`: the text comparison reply URI from the original collection,
  if present in the manifest
- `thread_structure`: structural depth and label copied from the manifest when
  available
- `hydration_metadata`: manifest and API recovery details

Each hydrated post object also includes current public engagement metadata when
available from the API:

```text
like_count
reply_count
repost_count
quote_count
```

These counts are captured at hydration time and may differ from counts observed
during the original firehose collection.

### Release Hydration Run Summary

The following summary comes from the completed release hydration run reported in
`hydration_report.json`. This run reflects public API and CDN availability as of
2026-05-20. It started on 2026-05-19 20:39:53 UTC and finished on 2026-05-20
06:37:12 UTC, for a total runtime of about 9 hours and 57 minutes.

| Metric | Count | Rate |
| --- | ---: | ---: |
| Selected manifest rows | 22,031 | 100.0% |
| Hydrated records | 21,671 | 98.4% |
| Failed rows | 360 | 1.6% |
| Image downloads attempted | 35,524 | 100.0% |
| Images downloaded | 35,475 | 99.9% |
| Image downloads failed | 49 | 0.1% |

All hydrated records contain the same top-level schema fields. Optional context
fields are present for every record but are `null` when the referenced public
post was unavailable or the manifest did not include that context.

| Field | Present | Non-null | Non-null rate |
| --- | ---: | ---: | ---: |
| `uid` | 21,671 | 21,671 | 100.0% |
| `uri` | 21,671 | 21,671 | 100.0% |
| `original_post` | 21,671 | 21,671 | 100.0% |
| `parent_reply` | 21,671 | 7,776 | 35.9% |
| `quoted_post` | 21,671 | 3,461 | 16.0% |
| `meme_reply` | 21,671 | 21,671 | 100.0% |
| `thread_structure` | 21,671 | 21,671 | 100.0% |
| `best_reply_before_meme` | 21,671 | 10,513 | 48.5% |
| `closest_text_reply` | 21,671 | 18,365 | 84.7% |
| `closest_sibling_text_reply` | 21,671 | 12,350 | 57.0% |
| `comparison_reply` | 21,671 | 18,526 | 85.5% |

### Image Options

`--download-images` controls how much media is fetched:

```text
none      Do not download images; keep source URLs only.
meme      Download only the meme reply image. This is the default.
context   Download meme reply plus root/parent/quoted/comparison images.
all       Same as context; reserved for future extra media roles.
```

Root and parent images are downloaded only with `--download-images context` or
`--download-images all`. With the default `meme` option, their image metadata and
source URLs remain in the JSON record, but the files are not saved locally.

For benchmark reproduction, the correct answer image and conversation context
can be hydrated here. Visual distractors can be distributed separately and joined
later by UID.

### Availability Notes

Hydration is best-effort because public posts and CDN images can disappear after
the original collection. Failed rows are recorded in `hydration_report.json`.
Records that can be identified but not retrieved keep their UID/URI and receive
an `in_archive: false` stub.

`hydration_report.json` also includes `hydrated_field_present` and
`hydrated_field_non_null` counts for the context fields produced by this
collection step. Optional context fields such as `parent_reply`, `quoted_post`,
and auxiliary comparison replies can be present with `null` values when the
manifest or public API does not provide them.

## Multiple-Choice Benchmark Download

Use this path when you want the fixed recognition benchmark used for
multiple-choice VLM evaluation. The benchmark package should be downloaded as a
prebuilt `benchmark_data/` directory and placed under `download/`:

```text
download/
└── benchmark_data/
    ├── benchmark_summary.jsonl
    ├── <uid>/
    │   ├── A_original.jpg
    │   ├── B_text_distractor.jpg
    │   ├── C_visual_distractor.jpg
    │   ├── D_easy_distractor.jpg
    │   └── meta.json
    └── ...
```

The full hydrated dataset can recover the correct answer image and conversation
context, but it does not recreate the exact benchmark distractors. For
comparable evaluation numbers, use the released benchmark download.

### Benchmark UID Hydration

Use `hydrate_benchmark_from_uids.py` when you have the fixed `benchmark_data/`
directory, benchmark UID list, or `benchmark_summary.jsonl` and want to hydrate
only those benchmark records from the larger released collection manifest. The
benchmark input can be the benchmark output directory itself, a plain text list,
JSONL, JSON, CSV, TSV, or the benchmark summary file as long as each row
contains a `uid`.

The lightweight benchmark UID list is included in this repository:

```text
download/data/benchmark_uids.txt
```

The larger 22,031-row release manifest is distributed separately as
`labeled_release.jsonl`. Download it from the release link and place it here:

```text
download/data/labeled_release.jsonl
```

`labeled_release.jsonl` contains project UIDs, public Bluesky AT URIs, labels,
validation metadata, and derived visual descriptions. It does not include raw
Bluesky post text, image files, image URLs, local image paths, user handles, or
display names.

```bash
python hydrate_benchmark_from_uids.py \
  --benchmark-uids data/benchmark_uids.txt \
  --manifest data/labeled_release.jsonl \
  --out benchmark_hydrated \
  --download-images context
```

The output includes:

```text
benchmark_hydrated/
├── benchmark_uid_manifest.jsonl
├── benchmark_hydration_report.json
├── records/<uid>.json
└── images/
```

The script still requires the full UID/URI manifest. A project UID alone keeps
only a truncated DID suffix, so it cannot be used directly with the public
Bluesky API.

This script restores the public-record side of the benchmark. It can recover
the correct answer image (`A_original.jpg`) and conversation context when the
original public Bluesky content is still available. The distractor images
(`B_text_distractor.jpg`, `C_visual_distractor.jpg`, and
`D_easy_distractor.jpg`) are not redistributed in this repository because they
are transformed from, or sampled from, public user images. To create a
functionally equivalent benchmark, rerun the benchmark construction pipeline in
[`../04_benchmark`](../04_benchmark/README.md). Regenerated distractors may not
be pixel-identical to the internal evaluation archive.

### Benchmark Hydration Run Summary

The following summary comes from the benchmark UID hydration run over the 5,000
rows in `data/benchmark_uids.txt`, joined against `data/labeled_release.jsonl`.
Failed rows are treated as posts that were deleted or otherwise unavailable
through Bluesky's public API at hydration time.

| Metric | Count | Rate |
| --- | ---: | ---: |
| Benchmark UIDs | 5,000 | 100.0% |
| Matched manifest rows | 5,000 | 100.0% |
| Hydrated or already available records | 4,948 | 99.0% |
| Failed rows | 52 | 1.0% |
| Images downloaded | 7,483 | - |
| Image downloads failed | 11 | - |

The 52 failed rows returned `HTTP 400` from Bluesky's public
`getPostThread` endpoint. Image download failures are counted separately from
row-level hydration failures.

### Benchmark Asset Mapping

`map_benchmark_assets.py` assembles benchmark-aligned release records by joining
UID-based label records with benchmark membership rows and provided distractor
asset rows. It accepts JSONL, JSON, CSV, or TSV mapping files as long as they
contain a UID column.

Use it when you have:

- a released labeled record file, such as `labeled_release.jsonl`
- a benchmark membership file, such as `benchmark_summary.jsonl`
- an optional distractor asset table keyed by `uid`

Example:

```bash
python map_benchmark_assets.py \
  --input data/labeled_release.jsonl \
  --benchmark-map benchmark_data/benchmark_summary.jsonl \
  --distractors data/distractor_assets.jsonl \
  --out benchmark_mapped \
  --no-copy-images
```

The output adds benchmark fields to each matched record:

```text
benchmark_mapped/
├── records/<uid>.json
├── benchmark_mapped_records.jsonl
└── mapping_report.json
```

Each written record includes:

- `benchmark_membership`: benchmark rows matched by UID
- `distractor_assets`: provided distractor rows matched by UID
- `mapping_metadata`: source paths, join key, and mapping timestamp

By default, the script also copies any local images referenced by `local_path`
into the output directory. Use `--no-copy-images` when mapping metadata only or
when the benchmark image archive is already distributed separately.

## Generation Task Code

For the generation task, provide code and protocol rather than treating generated
images as the primary downloadable dataset. Generated outputs depend on the
image model version, API behavior, prompts, random seed, and generation date, so
the most reproducible release artifact is the script plus enough metadata to
rerun the same setup.

The included implementation is
[`../06_meme_generation/generate_memes_compare.py`](../06_meme_generation/generate_memes_compare.py).
It generates paired outputs for each sampled record:

```text
A) w/o context: visual description only
B) w/ context: visual description + conversation context + stance labels
```

If generated samples are released, treat them as an optional snapshot and record
the model name, generation date, prompts, sample seed, and source UIDs alongside
the images. The code path in `06_meme_generation/` should remain the canonical
artifact for reproducing or extending the generation task.
