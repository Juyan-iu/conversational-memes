# download - Public UID Hydration Pipeline

This folder is for public reproduction of the released conversational meme
dataset by UID. It does not re-run the meme classifier. Instead, it reads a UID
manifest exported from `01_collection` and hydrates the same posts and
conversation context from the public Bluesky API.

This is the first public data-collection step. Later steps can map classifier
metadata, labels, benchmark membership, and provided distractors back onto the
hydrated records by `uid`.

## Why a Manifest Is Needed

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

## Files

```text
download/
├── download_manifest.py           # Download released manifest from Google Drive
├── hydrate_from_uid_manifest.py   # Hydrate records and optional images by UID
├── sample_uid_manifest.jsonl      # Tiny dry-run sample of the manifest format
└── data/                          # Put released manifests here; not required by git
```

## Manifest Format

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

## Setup

The hydration script uses only the Python standard library.

```bash
python --version  # Python 3.10+ recommended
```

No Bluesky login is required. The script calls:

```text
https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread
```

## Sample Runs

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

## Output

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

## Image Options

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

## Availability Notes

Hydration is best-effort because public posts and CDN images can disappear after
the original collection. Failed rows are recorded in `hydration_report.json`.
Records that can be identified but not retrieved keep their UID/URI and receive
an `in_archive: false` stub.

`hydration_report.json` also includes `hydrated_field_present` and
`hydrated_field_non_null` counts for the context fields produced by this
collection step. Optional context fields such as `parent_reply`, `quoted_post`,
and auxiliary comparison replies can be present with `null` values when the
manifest or public API does not provide them.
