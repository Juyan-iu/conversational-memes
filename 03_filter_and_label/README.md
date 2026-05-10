# Meme Discourse Labeling Pipeline

Automatic discourse function (Speech Function) and stance labeling pipeline for Bluesky post–meme reply conversational data.

> The dataset will be released separately.

---

## File Structure

```
03_filter_and_label/
├── label_pipeline.py               # Main labeling pipeline
├── postprocess.py                  # Post-processing (non-English removal, shortage generation)
├── refill_pipeline.py              # Monthly shortage refill pipeline
├── create_tree.py                  # Discourse function tree builder
├── three_levels.tsv                # 3-level taxonomy (for parent replies)
├── two_levels_no_sustain.tsv       # 2-level taxonomy (for meme replies, Sustain excluded)
├── three_levels_tree.json          # 3-level tree (for parent replies)
├── two_levels_no_sustain_tree.json # 2-level tree (for meme replies)
├── description_three_levels.txt
├── description_two_levels.txt
├── requirements.txt
├── .env.example
├── .gitignore
└── prompts/
    ├── prompt_free_form_non_binary.py
    ├── prompt_add_label.py
    ├── prompt_scorer.py
    └── prompt_annotate_dialogs.py
```

---

## Labeling Design

### Input Data Structure

```
original_post       <- Source post (not labeled)
└── parent_reply    <- Parent reply, if any (3-level labeling)
    └── meme_reply  <- Meme reply (2-level labeling + Stance + Visual)
```

### Meme Reply Labels (2-level, Sustain excluded)

Discourse Function:
- `Open.Attend` / `Open.Command` / `Open.Demand` / `Open.Give`
- `React.Rejoinder` / `React.Respond`

Stance (independent binary Yes/No per label):
- `Sarcastic` — ironic or mocking tone
- `Humorous` — comedic or playful tone
- `Offensive` — aggressive or harmful content

Visual Description:
- One-sentence description of the meme image's visual elements, excluding caption text

### Parent Reply Labels (3-level)

- `Open.*` / `React.Rejoinder.*` / `React.Respond.*` / `Sustain.*`

---

## Usage

### Step 1: Environment Setup

```bash
python -m venv label_env
source label_env/bin/activate
pip install -r requirements.txt
```

### Step 2: API Key Configuration

```bash
cp .env.example .env
# Fill in your OpenAI API key in .env
```

`.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 3: (Optional) Rebuild Trees

Tree files are already included; rebuilding is only necessary if the taxonomy is modified.

```bash
# 3-level tree for parent replies
python create_tree.py \
  -c free_form_non_binary \
  -t three_levels.tsv \
  -d description_three_levels.txt \
  -o three_levels_tree.json

# 2-level tree for meme replies (Sustain excluded)
python create_tree.py \
  -c free_form_non_binary \
  -t two_levels_no_sustain.tsv \
  -d description_two_levels.txt \
  -o two_levels_no_sustain_tree.json
```

### Step 4: Run Labeling

```bash
# Monthly balanced collection (target: 20,400 validated memes = 850/month x 24 months)
nohup python -u label_pipeline.py \
  --monthly-total 20400 \
  --output ./labeled_final \
  > ./pipeline.log 2>&1 &

echo $! > ./pipeline.pid
tail -f ./pipeline.log
```

### Step 5: Post-processing

```bash
# Check status without modifying files
python postprocess.py \
  --input ./labeled_final/labeled_memes.jsonl

# Remove non-English records, replace original file, generate shortage.json
python postprocess.py \
  --input ./labeled_final/labeled_memes.jsonl \
  --clean --replace
```

### Step 6: Refill Monthly Shortages

```bash
nohup python -u refill_pipeline.py \
  --shortage ./labeled_final/shortage.json \
  --output ./labeled_final \
  > ./refill.log 2>&1 &

echo $! > ./refill.pid
tail -f ./refill.log
```

---

## Output Structure

```
labeled_final/
├── labeled_memes.jsonl     # Full labeling results (JSONL)
├── shortage.json           # Per-month shortage info
└── records/
    └── {uid}.json          # Individual records
```

Each record:
```json
{
  "uid": "...",
  "original_post": { "text": "...", ... },
  "parent_reply": { "text": "...", ... },
  "meme_reply": { "text": "...", "images": [...], ... },
  "meme_validation": {
    "passed": true,
    "valid_ratio": 1.0,
    "validations": [{ "template_name": "...", ... }]
  },
  "discourse_labels": {
    "meme_reply": {
      "discourse_function": "React.Respond",
      "stance": {
        "sarcastic": false,
        "humorous": true,
        "offensive": false
      },
      "visual": {
        "visual_description": "..."
      }
    },
    "parent_reply": {
      "discourse_function": "React.Rejoinder.Support"
    }
  },
  "labeled_at": "2025-..."
}
```

---

## Recommended .gitignore Entries

```
label_env/
.env
labeled_final/
labeled_test/
__pycache__/
*.log
*.pid
monthly_checks/
```

---

## References

- Petukhova & Kochmar (2025). A Fully Automated Pipeline for Conversational Discourse Annotation. ACL 2025.
- Calderon et al. (2025). The Alternative Annotator Test for LLM-as-a-Judge. ACL 2025.
- Sharma et al. (2020). SemEval-2020 Task 8: Memotion Analysis. SemEval 2020.
- Hwang & Shwartz (2023). MemeCap. EMNLP 2023.
