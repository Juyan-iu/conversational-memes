# 07_analysis - Descriptive Dataset and Benchmark Analysis

This folder contains lightweight post-hoc analysis utilities for the meme-reply
dataset and the multiple-choice benchmark produced by earlier stages.

## What This Stage Does

`extract_dataset_stats.py` reports high-level descriptive statistics for the
curated meme-reply dataset from
[`../03_filter_and_label`](../03_filter_and_label/README.md), including thread
structure, multimodal context availability, engagement comparisons, relay or
cluster indicators, annotation coverage, and monthly counts.

`extract_benchmark_stats.py` summarizes the benchmark produced by
[`../04_benchmark`](../04_benchmark/README.md), including context-field
coverage, distractor availability, OCR-caption coverage, answer distribution,
discourse-label coverage if present, and monthly counts.

`analyze_meme_reply_characteristics.py` creates a richer set of CSV tables and a
Markdown summary for observable meme-reply characteristics. It intentionally
avoids `discourse_function` and stance labels, so its outputs can be used as
descriptive evidence without treating internal curation labels as ground truth.

## Files

```text
07_analysis/
|-- analyze_meme_reply_characteristics.py  # CSV/Markdown feature analysis
|-- extract_benchmark_stats.py             # Benchmark summary statistics
|-- extract_dataset_stats.py               # Dataset summary statistics
`-- README.md
```

Generated outputs are written to the path passed with `--out` or `--out-json`
and are not required for repository use.

## Requirements

- Python 3.10+
- Curated records from stage 03 for dataset analysis
- Benchmark data from stage 04 for benchmark analysis

These scripts use only the Python standard library.

## Run

From `07_analysis/`, summarize the curated dataset:

```bash
python extract_dataset_stats.py \
  --jsonl ../03_filter_and_label/labeled_final/labeled_memes.jsonl \
  --out-json ./dataset_stats.json
```

If the JSONL is not available, the script can fall back to a records directory:

```bash
python extract_dataset_stats.py \
  --records ../01_collection/meme_dataset/records \
  --out-json ./dataset_stats.json
```

Summarize benchmark coverage:

```bash
python extract_benchmark_stats.py \
  --summary ../04_benchmark/benchmark_data/benchmark_summary.jsonl \
  --out-json ./benchmark_stats.json
```

If `benchmark_summary.jsonl` is not present, point the script at the benchmark
directory so it can read `*/meta.json` files:

```bash
python extract_benchmark_stats.py \
  --bench-dir ../04_benchmark/benchmark_data \
  --out-json ./benchmark_stats.json
```

Generate detailed characteristic tables:

```bash
python analyze_meme_reply_characteristics.py \
  --input ../03_filter_and_label/labeled_final/labeled_memes.jsonl \
  --out ./analysis_out
```

To ignore `ancestor_chain` fields while preserving other context fields:

```bash
python analyze_meme_reply_characteristics.py \
  --input ../03_filter_and_label/labeled_final/labeled_memes.jsonl \
  --out ./analysis_out_no_ancestors \
  --ignore-ancestor-chain
```

`analyze_meme_reply_characteristics.py` accepts JSONL files, JSON files
containing one record or a list of records, directories containing
`records/*.json`, and directories containing top-level `*.jsonl` or `*.json`
files.

## Detailed Analysis Outputs

```text
analysis_out/
|-- record_features.csv
|-- summary.md
|-- table_basic_counts.csv
|-- table_monthly_counts.csv
|-- table_thread_structure.csv
|-- table_thread_depth.csv
|-- table_context_availability.csv
|-- table_meme_relay_features.csv
|-- table_direct_meme_relay_depth.csv
|-- table_comparison_selected_by.csv
|-- table_numeric_summary.csv
|-- table_engagement_comparison.csv
|-- table_template_names.csv
|-- top_threads_by_meme_count.csv
|-- top_parents_by_meme_count.csv
`-- meme_relay_candidates.jsonl
```

`record_features.csv` is the row-level feature table used to build the
aggregate tables. `summary.md` gives a compact report with basic counts, relay
features, engagement comparisons, numeric summaries, and caveats for interpreting
the tables.

## Notes

- The default paths point to the conventional local outputs from stages 03 and
  04. Override them when working with a different dataset snapshot.
- `extract_dataset_stats.py` includes annotation fields when they exist, but
  those fields should be described as internal curation metadata.
- `analyze_meme_reply_characteristics.py` does not use discourse-function or
  stance labels when building its descriptive tables.
- Relay and cluster indicators show co-occurrence or parent-chain structure;
  they do not prove direct conversational uptake on their own.
