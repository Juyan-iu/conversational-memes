#!/usr/bin/env python3
"""
Meme Candidate Validation Pipeline

For collected meme-reply candidates, this script:
1. Validates whether the candidate image is an internet meme.
2. Downloads/cache related media for reproducibility.
3. Adds meme-level visual description and stance metadata.
4. Appends accepted records to an output JSONL and records directory.

Usage:
  python label_pipeline.py                  # Process a sample of 10 records.
  python label_pipeline.py --sample 50      # Process a sample of 50 records.
  python label_pipeline.py --all            # Process all loaded records.
"""

import os
import json
import time
import base64
import argparse
import requests
from pathlib import Path
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

CONFIG = {
    # Input data folders, relative to this script.
    "input_dirs": [
        "../01_collection/meme_dataset_24_06",
        "../01_collection/meme_dataset_25_02",
        "../01_collection/meme_dataset",
    ],

    # Output folder.
    "output_dir": "./labeled_dataset",

    # Model settings.
    "model_main":   "gpt-5.4-mini",
    "model_visual": "gpt-5.4-mini",

    # Meme validation threshold.
    "meme_valid_threshold": 0.8,

    # Sample size used unless --all or --monthly-total is provided.
    "sample_size": 10,

    # Skip summary when text is short and no image is present.
    "summary_skip_length": 50,

    # Delay between API requests, in seconds.
    "api_delay": 0.1,

    # GPT-5.4-mini price per 1M tokens.
    "price_input":  0.75,   # $0.75/1M input tokens
    "price_output": 4.50,   # $4.50/1M output tokens
}

# -- global cost tracker -----------------------------------------
class CostTracker:
    def __init__(self):
        self.input_tokens  = 0
        self.output_tokens = 0
        self.calls         = 0

    def add(self, response):
        usage = response.usage
        self.input_tokens  += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        self.calls         += 1

    @property
    def cost(self) -> float:
        return (
            self.input_tokens  / 1_000_000 * CONFIG["price_input"] +
            self.output_tokens / 1_000_000 * CONFIG["price_output"]
        )

    def summary(self) -> str:
        return (
            f"  API calls:     {self.calls}\n"
            f"  Input tokens:  {self.input_tokens:,}\n"
            f"  Output tokens: {self.output_tokens:,}\n"
            f"  Estimated cost: ${self.cost:.4f}"
        )

COST = CostTracker()

# ════════════════════════════════════════════════════════════════
#  OpenAI client
# ════════════════════════════════════════════════════════════════

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ════════════════════════════════════════════════════════════════
#  Utilities
# ════════════════════════════════════════════════════════════════

def load_records(input_dirs: list) -> list:
    """Load records/*.json from one or more dataset folders."""
    records = []
    for dir_str in input_dirs:
        d = Path(dir_str)
        if not d.exists():
            print(f"[SKIP] Missing folder: {d}")
            continue
        records_dir = d / "records"
        if not records_dir.exists():
            print(f"[SKIP] Missing records/ folder: {d}")
            continue
        files = sorted(records_dir.glob("*.json"))
        for fp in files:
            try:
                rec = json.loads(fp.read_text("utf-8"))
                rec["_source_dir"] = str(d)
                records.append(rec)
            except Exception as e:
                print(f"  [WARN] Failed to read {fp.name}: {e}")
        print(f"[LOAD] {d} -> {len(files)} files")
    return records


def load_records_by_month(input_dirs: list) -> dict:
    """
    Extract months from index_YYYY-MM-DD.jsonl files, map UIDs to
    records/{uid}.json, and return {YYYY-MM: [record, ...]}.
    """
    import re
    from collections import defaultdict
    by_month = defaultdict(list)
    uid_loaded = set()

    for dir_str in input_dirs:
        d = Path(dir_str)
        if not d.exists():
            continue
        records_dir = d / "records"

        # Find index_YYYY-MM-DD.jsonl files.
        index_files = sorted(d.glob("index_????-??-??.jsonl"))
        if not index_files:
            # Fall back to direct records/*.json loading when no index exists.
            print(f"[WARN] {d}: no index files found; loading records/ directly")
            for fp in sorted(records_dir.glob("*.json")):
                try:
                    rec = json.loads(fp.read_text("utf-8"))
                    uid = rec.get("uid", "")
                    if uid and uid not in uid_loaded:
                        rec["_source_dir"] = str(d)
                        by_month["unknown"].append(rec)
                        uid_loaded.add(uid)
                except Exception:
                    pass
            continue

        for idx_f in index_files:
            # Extract month from filename: index_2024-06-15.jsonl -> 2024-06.
            m = re.search(r"index_(\d{4}-\d{2})-\d{2}\.jsonl", idx_f.name)
            if not m:
                continue
            month = m.group(1)

            # Read UIDs from the index JSONL.
            try:
                with open(idx_f, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        uid = entry.get("uid", "")
                        if not uid or uid in uid_loaded:
                            continue
                        # Load records/{uid}.json.
                        rec_path = records_dir / f"{uid}.json"
                        if rec_path.exists():
                            try:
                                rec = json.loads(rec_path.read_text("utf-8"))
                                rec["_source_dir"] = str(d)
                                by_month[month].append(rec)
                                uid_loaded.add(uid)
                            except Exception:
                                pass
            except Exception as e:
                print(f"  [WARN] {idx_f.name}: {e}")

        months_found = sorted(set(
            re.search(r"index_(\d{4}-\d{2})", f.name).group(1)
            for f in index_files
            if re.search(r"index_(\d{4}-\d{2})", f.name)
        ))
        print(f"[LOAD] {d} -> {sum(len(by_month[m]) for m in months_found)} records ({len(months_found)} months)")

    return by_month


def download_image_to_base64(url: str, retries: int = 3, timeout: int = 15) -> str | None:
    """Download an image URL and return a base64 string."""
    session = requests.Session()
    session.headers.update({"User-Agent": "MemeResearchBot/1.0 (academic research)"})
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=(5, timeout), stream=True)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return base64.b64encode(r.content).decode("utf-8")
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def download_and_save_image(url: str, save_path: Path, retries: int = 3) -> bool:
    """Download an image URL to a local file."""
    session = requests.Session()
    session.headers.update({"User-Agent": "MemeResearchBot/1.0 (academic research)"})
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=(5, 15), stream=True)
            if r.status_code == 404:
                return False
            if r.status_code == 200:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return False


def build_image_content(images: list, output_dir: Path, uid: str, subfolder: str) -> list:
    """
    Convert image metadata to OpenAI image content items and cache local files.
    Returns: [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
    """
    content = []
    for img in (images or []):
        url = img.get("url") or img.get("source_url")
        cid = img.get("cid", "")
        if not url:
            continue

        # Local cache.
        save_path = output_dir / "images" / subfolder / uid / f"{cid}.jpg"
        if not save_path.exists():
            download_and_save_image(url, save_path)

        # Base64 encoding.
        b64 = download_image_to_base64(url)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
    return content


def gpt4o_call(messages: list, max_tokens: int = 200, use_visual_model: bool = False) -> str:
    """Call the configured chat completion model."""
    model = CONFIG["model_visual"] if use_visual_model else CONFIG["model_main"]
    time.sleep(CONFIG["api_delay"])
    # GPT-5.x style models use max_completion_tokens; older models use max_tokens.
    use_new_param = any(m in model for m in ["gpt-5", "o1", "o3"])
    token_param = "max_completion_tokens" if use_new_param else "max_tokens"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **{token_param: max_tokens},
        temperature=0,
    )
    COST.add(response)
    return response.choices[0].message.content.strip()


# ════════════════════════════════════════════════════════════════
#  Step 1: meme validation
# ════════════════════════════════════════════════════════════════

def validate_meme_image(image_b64: str) -> dict:
    """
    Validate whether an image is a real internet meme.
    Returns: {"is_valid_meme": bool, "confidence": float, "reason": str}
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image and determine if it is an internet meme.\n\n"
                        "A meme must meet ALL of the following criteria:\n"
                        "1. Has visible text/caption overlaid ON the image itself.\n"
                        "   Images with NO text at all are NOT memes.\n"
                        "2. The text is part of the meme format "
                        "(not just a watermark, logo, or news subtitle).\n"
                        "3. Uses a recognizable meme format or template "
                        "that is designed to be remixed or parodied.\n\n"
                        "Answer false if:\n"
                        "- No text is visible on the image\n"
                        "- It is a regular photo or screenshot with no meme format\n"
                        "- It is an advertisement or infographic\n\n"
                        "Text can be in any language. When in doubt, answer false.\n\n"
                        "Respond ONLY with a valid JSON object (no markdown):\n"
                        '{"is_valid_meme": true/false, "confidence": 0.0-1.0, '
                        '"template_name": "name of meme template or null", '
                        '"reason": "one sentence explanation"}'
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]
        }
    ]
    try:
        result = gpt4o_call(messages, max_tokens=150, use_visual_model=True)
        # Parse JSON.
        result = result.strip().strip("```json").strip("```").strip()
        return json.loads(result)
    except Exception as e:
        return {"is_valid_meme": False, "confidence": 0.0, "reason": f"parse_failed: {e}"}


def validate_record_memes(record: dict, output_dir: Path = None) -> dict:
    """
    Validate meme images in one record.
    The record passes when valid_ratio >= CONFIG["meme_valid_threshold"].
    Returns: {"passed": bool, "valid_ratio": float, "validations": [...]}
    """
    uid = record.get("uid", "unknown")
    meme_images = record.get("meme_reply", {}).get("images", [])
    if not meme_images:
        return {"passed": False, "valid_ratio": 0.0, "validations": [], "reason": "no_meme_image"}

    validations = []
    for img in meme_images:
        url = img.get("url") or img.get("source_url")
        cid = img.get("cid", "")
        if not url:
            continue

        b64 = None

        # Prefer cached output_dir image.
        if output_dir and cid:
            cached = output_dir / "images" / "meme_reply" / uid / f"{cid}.jpg"
            if cached.exists():
                try:
                    b64 = base64.b64encode(cached.read_bytes()).decode("utf-8")
                except Exception:
                    b64 = None

        # Download from URL if no cache is available.
        if not b64:
            b64 = download_image_to_base64(url)

        if not b64:
            validations.append({"url": url, "is_valid_meme": False, "confidence": 0.0, "reason": "download_failed"})
            continue
        result = validate_meme_image(b64)
        result["url"] = url
        validations.append(result)

    if not validations:
        return {"passed": False, "valid_ratio": 0.0, "validations": [], "reason": "no_validatable_image"}

    valid_count = sum(1 for v in validations if v.get("is_valid_meme", False))
    valid_ratio = valid_count / len(validations)
    passed = valid_ratio >= CONFIG["meme_valid_threshold"]

    return {
        "passed": passed,
        "valid_ratio": round(valid_ratio, 4),
        "validations": validations,
        "reason": f"{valid_count}/{len(validations)} images passed meme validation"
    }


# ════════════════════════════════════════════════════════════════
#  Step 2: image download
# ════════════════════════════════════════════════════════════════

def download_context_images(record: dict, output_dir: Path) -> dict:
    """
    Download context images from the original post, parent reply, comparison
    reply, and meme reply.
    Returns: {"original_post": [...], "parent_reply": [...], ...}
    """
    uid = record.get("uid", "unknown")
    downloaded = {}

    targets = {
        "original_post":          record.get("original_post", {}),
        "parent_reply":           record.get("parent_reply"),
        "best_reply_before_meme": record.get("best_reply_before_meme"),
        "comparison_reply":       record.get("comparison_reply"),
        "meme_reply":             record.get("meme_reply", {}),
    }

    for key, post in targets.items():
        if not post:
            downloaded[key] = []
            continue
        images = post.get("images", [])
        saved = []
        for img in (images or []):
            url = img.get("url") or img.get("source_url")
            cid = img.get("cid", "")
            if not url:
                continue
            save_path = output_dir / "images" / key / uid / f"{cid}.jpg"
            ok = False
            if save_path.exists():
                ok = True
            else:
                ok = download_and_save_image(url, save_path)
            if ok:
                saved.append({
                    "local_path": str(save_path.relative_to(output_dir)),
                    "cid": cid,
                    "url": url,
                    "alt": img.get("alt", "")
                })
        downloaded[key] = saved

    return downloaded


# ════════════════════════════════════════════════════════════════
#  Step 3: meme visual description
# ════════════════════════════════════════════════════════════════

def label_meme_visual(images_content: list, utterance_text: str = "") -> dict:
    """
    Describe the meme image visual elements in one sentence, excluding caption text.
    Returns: {"visual_description": str}
    """
    if not images_content:
        return {"visual_description": None}

    content = [
        {
            "type": "text",
            "text": (
                "Describe this meme image in ONE concise sentence that:\n"
                "1. Describes the visual elements (characters, objects, expressions, setting)\n"
                "2. Includes what those visual elements symbolize or represent as a metaphor\n"
                "3. Does NOT mention or quote any text/caption visible in the image\n"
                "4. Is self-contained enough that someone could recreate the meme "
                "just from reading your sentence (without seeing the image)\n\n"
                "Example: \"A dog sitting calmly in a burning room, "
                "symbolizing willful ignorance of an obvious crisis.\"\n\n"
                "Reply with ONE sentence only. No quotes around it."
            )
        }
    ] + images_content

    try:
        description = gpt4o_call(
            [{"role": "user", "content": content}],
            max_tokens=120,
            use_visual_model=True
        )
        return {"visual_description": description}
    except Exception:
        return {"visual_description": None}


# ════════════════════════════════════════════════════════════════
#  Step 4: meme stance metadata
# ════════════════════════════════════════════════════════════════

def classify_stance(utterance: str, images_content: list = None) -> dict:
    """
    Classify meme-reply stance metadata.
    Sarcastic / Humorous / Offensive are independent Yes/No decisions.
    """
    stance_questions = [
        ("sarcastic", "Is this utterance sarcastic or ironic? "
                      "Does it express the opposite of what it literally means, or mock someone/something?"),
        ("humorous",  "Is this utterance humorous or funny? "
                      "Does it use comedy, jokes, or playful language?"),
        ("offensive", "Is this utterance offensive or aggressive? "
                      "Does it attack, demean, or use hateful language toward someone or something?"),
    ]

    stance_result = {}
    for key, question in stance_questions:
        content = [
            {
                "type": "text",
                "text": (
                    f"Analyze the following meme utterance.\n\n"
                    f"Utterance: {utterance}\n\n"
                    f"Question: {question}\n\n"
                    f"Reply ONLY with \"Yes\" or \"No\"."
                )
            }
        ]
        if images_content:
            content.extend(images_content)

        messages = [{"role": "user", "content": content}]
        try:
            response = gpt4o_call(messages, max_tokens=5)
            stance_result[key] = "yes" in response.lower()
        except Exception:
            stance_result[key] = False

    return stance_result


# ════════════════════════════════════════════════════════════════
#  Optional summary helper
# ════════════════════════════════════════════════════════════════

def generate_summary(utterance_text: str, images_content: list = None) -> str | None:
    """
    Summarize literal visual/textual content in one sentence.
    Returns None for short text with no images.
    """
    text_len = len(utterance_text.strip())
    has_image = bool(images_content)

    # Skip short text with no images.
    if text_len <= CONFIG["summary_skip_length"] and not has_image:
        return None

    content = [
        {
            "type": "text",
            "text": (
                "Describe what this utterance literally shows or says in ONE short sentence. "
                "Focus ONLY on what is visually or textually present - "
                "do NOT interpret context, intent, or meaning. "
                "Do NOT include any caption text from meme images.\n\n"
                f"Utterance text: {utterance_text}\n\n"
                "Reply with ONE sentence only."
            )
        }
    ]
    if images_content:
        content.extend(images_content)

    messages = [{"role": "user", "content": content}]
    try:
        return gpt4o_call(messages, max_tokens=80)
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
#  Step 5: record processing
# ════════════════════════════════════════════════════════════════

def get_post_text(post: dict) -> str:
    """Extract text from a post/reply dictionary."""
    if not post:
        return ""
    return post.get("text", "") or ""


def get_post_images_content(post: dict, output_dir: Path = None, uid: str = None, key: str = None) -> list:
    """
    Extract OpenAI image content items from a post/reply dictionary.
    Reuse local files when available; otherwise download from URL.
    """
    if not post:
        return []
    images = post.get("images", []) or []
    content = []
    for img in images:
        url = img.get("url") or img.get("source_url")
        cid = img.get("cid", "")
        if not url:
            continue

        b64 = None

        # Prefer cached output_dir image.
        if output_dir and uid and key and cid:
            cached = output_dir / "images" / key / uid / f"{cid}.jpg"
            if cached.exists():
                try:
                    b64 = base64.b64encode(cached.read_bytes()).decode("utf-8")
                except Exception:
                    b64 = None

        # Download from URL if no cache is available.
        if not b64:
            b64 = download_image_to_base64(url)

        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
    return content


def process_record(record: dict, output_dir: Path) -> dict | None:
    """
    Process one record:
    1. Validate the meme image.
    2. Download/cache media.
    3. Add meme-level visual description and stance metadata.
    """
    uid = record.get("uid", "unknown")
    print(f"\n[PROCESS] {uid}")

    # 1. Download images first so later calls can use local cache.
    print("  [1] Downloading images...")
    downloaded_images = download_context_images(record, output_dir)

    # 2. Validate meme image using the local cache when possible.
    print("  [2] Validating meme image...")
    validation = validate_record_memes(record, output_dir=output_dir)
    print(f"      result: {validation['reason']} (valid_ratio={validation['valid_ratio']:.1%})")

    if not validation["passed"]:
        print("  [SKIP] Meme validation failed")
        return None

    # 3. Generate meme-level metadata only.
    print("  [3] Creating meme-level metadata...")
    meme_reply = dict(record.get("meme_reply") or {})
    meme_text = get_post_text(meme_reply)
    meme_images = get_post_images_content(
        meme_reply, output_dir=output_dir, uid=uid, key="meme_reply"
    )
    print(f"      [meme_reply] text={meme_text[:40]!r} images={len(meme_images)}")

    stance_labels = classify_stance(meme_text, meme_images if meme_images else None)
    visual_labels = (
        label_meme_visual(meme_images, meme_text)
        if meme_images else {"visual_description": None}
    )
    annotation = {
        "stance_labels": stance_labels,
        "visual_description": visual_labels.get("visual_description"),
    }
    meme_reply["annotation"] = annotation
    meme_reply["visual_description"] = annotation["visual_description"]

    print(f"      stance: {stance_labels}")
    print(f"      visual: {str(annotation['visual_description'] or '')[:60]}")

    # Combine result.
    result = {
        **record,
        "meme_reply": meme_reply,
        "meme_validation":  validation,
        "downloaded_images": downloaded_images,
        "meme_annotation": annotation,
        "stance_labels": stance_labels,
        "visual_description": annotation["visual_description"],
        "labeled_at": datetime.now(timezone.utc).isoformat(),
    }

    return result


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Meme candidate validation pipeline")
    parser.add_argument("--sample", type=int, default=CONFIG["sample_size"],
                        help=f"Number of sampled records to process (default: {CONFIG['sample_size']})")
    parser.add_argument("--all", action="store_true",
                        help="Process all loaded records without sampling.")
    parser.add_argument("--output", default=CONFIG["output_dir"],
                        help=f"Output folder (default: {CONFIG['output_dir']})")
    parser.add_argument("--input", default=None, nargs="+",
                        help="Input dataset folders (default: CONFIG input_dirs).")
    parser.add_argument("--uid-file", default=None,
                        help="Text file with one UID per line to process.")
    parser.add_argument("--save-uids", default=None,
                        help="Save sampled UIDs to a text file for reproducible reruns.")
    parser.add_argument("--model", default=None,
                        help="Override model_main and model_visual together. Example: gpt-5.4-mini")
    parser.add_argument("--model-visual", default=None,
                        help="Override only the visual-description model. Example: gpt-4o")
    parser.add_argument("--monthly-total", type=int, default=None,
                        help="Balanced accepted-record target across months. Example: 20000")
    args = parser.parse_args()

    # Model overrides.
    if args.model:
        CONFIG["model_main"]   = args.model
        CONFIG["model_visual"] = args.model
        print(f"  Model override: {args.model}")
    if args.model_visual:
        CONFIG["model_visual"] = args.model_visual
        print(f"  Visual model override: {args.model_visual}")

    # Output folder setup.
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records").mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    # Load records.
    input_dirs = args.input if args.input else CONFIG["input_dirs"]
    print("\n[LOAD] Loading records...")
    print(f"  Input folders: {input_dirs}")
    records = load_records(input_dirs)
    print(f"  Total records: {len(records)}")

    # Filter mass-mention posts (three or more @mentions).
    import re
    from collections import defaultdict

    def has_mass_mention(record: dict, threshold: int = 3) -> bool:
        text = (record.get("original_post") or {}).get("text") or ""
        mentions = re.findall(r"@[\w.\-]+", text)
        return len(mentions) >= threshold

    before = len(records)
    records = [r for r in records if not has_mass_mention(r)]
    print(f"  Mass-mention filter: removed {before - len(records)} -> {len(records)} remain")

    # Lightweight English filter for original posts and parent replies.
    def is_english(record: dict, threshold: float = 0.7) -> bool:
        texts = []
        orig_text = (record.get("original_post") or {}).get("text") or ""
        if orig_text.strip():
            texts.append(orig_text)
        parent_text = (record.get("parent_reply") or {}).get("text") or ""
        if parent_text.strip():
            texts.append(parent_text)

        for text in texts:
            alpha = sum(1 for c in text if c.isascii() and c.isalpha())
            total = sum(1 for c in text if c.isalpha())
            if total == 0:
                continue
            if alpha / total < threshold:
                return False
        return True

    before = len(records)
    records = [r for r in records if is_english(r)]
    print(f"  English filter: removed {before - len(records)} -> {len(records)} remain")



    # Use a fixed UID list when provided.
    if args.uid_file:
        with open(args.uid_file, encoding="utf-8") as f:
            target_uids = set(line.strip() for line in f if line.strip())
        records = [r for r in records if r.get("uid", "") in target_uids]
        print(f"  UID-file filter: {len(records)} records")
    elif not args.all and not args.monthly_total:
        import random
        sample_size = args.sample
        random.shuffle(records)
        records = records[:sample_size]
        print(f"  Processing random sample of {sample_size} records")

    # Save sampled UIDs if requested.
    if args.save_uids:
        with open(args.save_uids, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.get("uid", "") + "\n")
        print(f"  Saved UIDs: {args.save_uids} ({len(records)} records)")

    # Output JSONL is append-only.
    output_jsonl = output_dir / "labeled_memes.jsonl"
    print("\n  Processing from the beginning without duplicate checks")

    # Processing counters.
    success = 0
    skipped_validation = 0

    # -- monthly quota mode -----------------------------------------
    if args.monthly_total:
        import random

        # Load records by month from index files.
        print("\n[LOAD] Loading records by monthly index...")
        by_month_raw = load_records_by_month(input_dirs)

        # Apply mass-mention and English filters.
        by_month = {}
        total_filtered = 0
        for m, recs in by_month_raw.items():
            filtered = [r for r in recs if not has_mass_mention(r) and is_english(r)]
            total_filtered += len(recs) - len(filtered)
            by_month[m] = filtered
        print(f"  Applied filters: removed {total_filtered:,} records")

        months = sorted(k for k in by_month.keys() if k != "unknown")
        n_months = len(months)
        per_month = args.monthly_total // max(n_months, 1)

        print("\n  Monthly quota mode")
        print(f"  Target: {args.monthly_total:,} / {n_months} months = {per_month:,} accepted records per month")
        print(f"\n{'='*60}")

        # Shuffle records within each month.
        for m in months:
            random.shuffle(by_month[m])
        if "unknown" in by_month:
            random.shuffle(by_month["unknown"])

        month_passed = {m: 0 for m in months}
        month_passed["unknown"] = 0

        with open(output_jsonl, "a", encoding="utf-8") as out_f:
            for current_month in months + (["unknown"] if by_month.get("unknown") else []):
                quota = per_month if current_month != "unknown" else (args.monthly_total - sum(month_passed.values()))
                if quota <= 0:
                    continue
                month_records = by_month[current_month]
                print(f"\n  [{current_month}] start (quota: {quota:,}, candidates: {len(month_records):,})")
                i = 0
                for record in month_records:
                    if month_passed[current_month] >= quota:
                        print(f"  [{current_month}] quota reached ({quota:,}); moving to next month")
                        break
                    i += 1
                    uid = record.get("uid", "unknown")
                    print(f"\n  [{current_month} {month_passed[current_month]+1}/{quota}] uid={uid}")
                    try:
                        result = process_record(record, output_dir)
                        if result is None:
                            skipped_validation += 1
                            continue
                        out_f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                        out_f.flush()
                        record_path = output_dir / "records" / f"{uid}.json"
                        record_path.write_text(
                            json.dumps(result, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8"
                        )
                        month_passed[current_month] += 1
                        success += 1
                    except Exception as e:
                        import traceback
                        print(f"  [ERROR] {e}\n{traceback.format_exc()}")

        print("\n  Monthly collection result:")
        for m in months:
            print(f"    {m}: {month_passed[m]:,} records")

    else:
        # -- standard mode ------------------------------------------
        print(f"\n{'='*60}")
        print(f"  Processing {len(records)} records")
        print(f"{'='*60}")

        with open(output_jsonl, "a", encoding="utf-8") as out_f:
            for i, record in enumerate(records):
                uid = record.get("uid", "unknown")
                print(f"\n[{i+1}/{len(records)}] uid={uid}")

                try:
                    result = process_record(record, output_dir)

                    if result is None:
                        skipped_validation += 1
                        continue

                    # Append to JSONL.
                    out_f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                    out_f.flush()

                    # Also save an individual records/{uid}.json file.
                    record_path = output_dir / "records" / f"{uid}.json"
                    record_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8"
                    )
                    success += 1
                    print("  [DONE] saved")

                except Exception as e:
                    import traceback
                    print(f"  [ERROR] {e}\n{traceback.format_exc()}")

    # Summary.
    print(f"\n{'='*60}")
    print("  Processing complete")
    print(f"  Success:              {success}")
    print(f"  Validation failures:  {skipped_validation}")
    print(f"  Output: {output_dir.resolve()}")
    print("  ├─ labeled_memes.jsonl   (merged)")
    print("  ├─ records/{uid}.json    (individual)")
    print("  └─ images/               (downloaded images)")
    print(f"{'='*60}")
    print("\n  [API cost]")
    print(COST.summary())
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
