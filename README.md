# **MemeConv**

Code release for **MemeConv: A Dataset and Benchmark for Meme Literacy in the Wild Online Conversations**.

Folders `01_collection/` through `07_analysis/` are for reproducing the paper's pipeline and results. The `download/` folder is for accessing the released dataset and benchmark artifact.

This repository contains the pipelines used to collect naturally occurring meme replies from Bluesky, curate the MemeConv dataset, build the contextual meme-selection benchmark, evaluate LVLMs, and run the context-conditioned meme-generation study.

For setup and run commands, see the README inside each stage folder.

## Repository Map

| Folder | Paper location | Purpose |
| --- | --- | --- |
| `01_collection/` | Sec. 3.1; App. C.1-C.3, C.5 | Collect single-image Bluesky replies from firehose archives, filter candidate meme replies with MemeTector, and save raw candidate records. |
| `02_meme_classification/` | Sec. 3.1 Meme Filtering; App. C.3 | Train or inspect MemeTector v4, the CLIP-based binary classifier used for first-pass meme filtering. |
| `03_filter_and_label/` | Sec. 3.1 Annotation; App. C.4, D, F | Validate collected candidates with an LVLM, add visual descriptions and stance labels, clean records, and refill monthly shortages. |
| `04_benchmark/` | Sec. 3.3.1; App. G.1 | Construct the 5,000-item four-choice contextual meme-selection benchmark with text, visual, and easy distractors. |
| `05_benchmark_evaluation/` | Sec. 4.1; App. G.1.8, H | Convert benchmark items into the evaluation format and run closed/API and open-weight LVLMs. |
| `06_meme_generation/` | Sec. 3.3.2, 4.2; App. G.2, I, J | Generate paired memes with and without conversational context for the RQ2 generation study. |
| `07_analysis/` | Sec. 3.2; App. B, E | Produce descriptive dataset and benchmark statistics, including structure, context coverage, monthly counts, and meme-reply characteristics. |
| `download/` | Sec. 3 release/reproducibility note; App. C.5 | Hydrate released public artifacts by UID and access dataset or benchmark resources without re-running the full pipeline. |

## Pipeline

```text
02_meme_classification -> 01_collection -> 03_filter_and_label
                                      \-> 04_benchmark -> 05_benchmark_evaluation
03_filter_and_label -------------------> 06_meme_generation
03_filter_and_label / 04_benchmark ----> 07_analysis
download ------------------------------> public artifact hydration
```

## Outputs

Large generated artifacts are intentionally not tracked in git. This includes collected records, downloaded images, trained checkpoints, labeled datasets, benchmark images, model predictions, generated memes, and analysis outputs.

## Data Access

The public release uses stable project UIDs plus hydration scripts rather than redistributing original Bluesky posts and images directly. See [`download/README.md`](download/README.md) for hydration and artifact-access instructions.

