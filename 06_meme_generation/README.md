# 06_meme_generation - Context-Conditioned Meme Generation

This folder contains the meme-generation comparison used for the MEMECONV
generation task. It follows the paper's controlled setup: for each sampled meme
record, generate one meme from the isolated visual description and one meme from
the same visual description plus conversational context and stance attributes.

The goal is to test whether reply context helps an image model produce a meme
that is more appropriate as a conversational response, while keeping the source
visual description fixed.

## What This Stage Does

`generate_memes_compare.py` reads labeled meme records from
[`../03_filter_and_label`](../03_filter_and_label/README.md), samples records
with a `visual_description` and available parent reply context, and generates
paired meme images:

```text
Condition A / w/o context:
visual_description -> GPT Image 2 -> generated meme

Condition B / w/ context:
visual_description + conversation text + stance labels -> GPT Image 2 -> generated meme
```

The comparison mirrors the paper's RQ2 generation task. Condition A receives
only a context-independent description of the original meme image. Condition B
adds the surrounding conversation, including available original-post text,
quoted-post text, ancestor replies, parent reply, meme reply text, and binary
stance labels for sarcasm, humor, and offensiveness.

## Files

```text
06_meme_generation/
|-- generate_memes_compare.py   # Paired meme generation and HTML viewer
|-- requirements.txt
`-- README.md
```

Generated outputs are written to the configured output directory and are not
part of the source workflow:

```text
generated/
|-- <uid>/
|   |-- A_gptimage2.png          # w/o context generated image
|   |-- A_gptimage2.txt          # prompt used for condition A
|   |-- B_gptimage2.png          # w/ context generated image
|   `-- B_gptimage2.txt          # prompt used for condition B
|-- comparison_<timestamp>.html  # side-by-side review interface
`-- results_<timestamp>.json     # generated file paths and prompts
```

## Input

By default, the script reads:

```python
LABELED_JSONL = "../03_filter_and_label/labeled_final/labeled_memes.jsonl"
```

Each usable record must contain:

- `discourse_labels.meme_reply.visual.visual_description`
- `parent_reply`
- conversation fields such as `original_post`, `quoted_post`,
  `ancestor_chain`, and `meme_reply` when available
- stance labels under `discourse_labels.meme_reply.stance`

The current implementation shuffles eligible records with a fixed seed and
generates the requested number of complete paired examples. If one image in a
pair fails, that record is skipped and the script continues through an expanded
candidate pool.

## Requirements

- Python 3.10+
- OpenAI API key
- Labeled records from stage 03
- Network access for the OpenAI Images API

Install dependencies:

```bash
cd 06_meme_generation
python -m venv meme_gen_env
source meme_gen_env/bin/activate
pip install -r requirements.txt
```

Set your API key in `.env` or export it in the shell:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

## Run

Run a small smoke test:

```bash
python generate_memes_compare.py --n 3
```

Generate 50 paired comparisons:

```bash
python generate_memes_compare.py \
  --n 50 \
  --out ./generated
```

Use a custom labeled JSONL file:

```bash
python generate_memes_compare.py \
  --jsonl ../03_filter_and_label/labeled_final/labeled_memes.jsonl \
  --n 50 \
  --seed 42
```

Useful options:

```text
--jsonl PATH   Source labeled JSONL file.
--n N          Number of complete paired examples to generate.
--out DIR      Output directory for images, prompts, HTML, and JSON.
--delay SEC    Delay between image API calls.
--seed N       Random seed for eligible-record sampling.
```

## Prompt Conditions

Condition A is intentionally context-free. It asks the image model to create an
internet meme from the visual description alone, with a short readable caption.

Condition B asks for a meme that would plausibly function as a reply in the
specific conversation. The prompt tells the model to react to the root post,
parent reply, or key conversational moment; reflect the stance labels; and avoid
copying conversation text verbatim.

The script currently sends the textual conversation context and stance labels
directly to the image model. Context images are displayed in the HTML viewer
when their URLs are present in the record, but they are not passed as generation
inputs in the current code path.

## Review Output

The HTML viewer shows, for each completed UID:

- the conversation thread and available context images
- the annotated visual description
- the w/o-context and w/context generated images
- the truncated prompt used for each generated image

This viewer is intended for human review of the same dimensions used in the
paper's generation evaluation: image quality, conversational relevance, and
humor.

## Notes

- The paper reports that adding context improves conversational relevance but
  can reduce image quality because models may over-copy conversation wording
  into captions. Keep this tradeoff in mind when reviewing outputs.
- Generated images and HTML files may contain offensive or harmful content,
  because the source records are collected from public, in-the-wild
  conversations.
- Existing output files are skipped, so interrupted runs can be resumed with
  the same output directory.
