# 06_meme_generation — Meme Generation

Generates meme images from labeled conversational meme data using GPT-4o and DALL-E 3.

Two generation strategies are compared:
- **Version A**: uses only the meme's visual description extracted during labeling
- **Version B**: uses the full conversation context (text + embedded images) and discourse metadata (stance, discourse function) to prompt GPT-4o, which then generates a DALL-E 3 prompt

## Setup

```bash
cd 06_meme_generation
python -m venv meme_gen_env
source meme_gen_env/bin/activate
pip install -r requirements.txt
```

Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Generate both versions (top 50 by like count)
python generate_memes.py

# Options
python generate_memes.py --n 50 --out ./generated
python generate_memes.py --version a   # Version A only
python generate_memes.py --version b   # Version B only
```

## Output

```
generated/
├── version_a/
│   ├── bsky_xxx_yyy_a.png   # visual_description → DALL-E 3
│   └── ...
├── version_b/
│   ├── bsky_xxx_yyy_b.png   # context + stance → GPT-4o → DALL-E 3
│   └── ...
└── results.json             # metadata
```

## Generation Strategies

### Version A — Description-only
```
visual_description → DALL-E 3 prompt → image
```
Uses the `visual.visual_description` field from `discourse_labels.meme_reply`.

Example description:
> "A shouting armored man faces a colossal dark fortress wall with tiny figures at its base, symbolizing a lone plea for justice against an overwhelming oppressive force."

### Version B — Context-aware
```
conversation text + embedded images + stance + discourse_function
    → GPT-4o (generates DALL-E prompt)
    → DALL-E 3
    → image
```

GPT-4o receives:
- Full conversation thread (root post → quoted post → thread replies → parent reply → meme text)
- Context images (original post images, embedded/quoted images)
- Discourse function (e.g. `React.Respond`)
- Stance tags (sarcastic / humorous / offensive)

## Cost Estimate (50 items)

| Step | Model | Cost |
|------|-------|------|
| Version A image | DALL-E 3 standard | $0.04 × 50 = **$2.00** |
| Version B GPT prompt | GPT-4o | ~$0.01 × 50 = **$0.50** |
| Version B image | DALL-E 3 standard | $0.04 × 50 = **$2.00** |
| **Total** | | **~$4.50** |

## Sampling

Top 50 records by `meme_reply.like_count` from `labeled_memes.jsonl`.
Only records with a `visual_description` field are included.
