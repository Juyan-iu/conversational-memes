"""Shared utilities for the meme-selection task (paper RQ1 configuration).

Task format:
  - One configuration: 4 meme images labeled A/B/C/D (letters assigned by
    shuffle at load time), always with conversation context.
  - 4 options per item: 1 correct + 3 typed distractors (lexical / visual / random).
  - The model picks a single letter A/B/C/D.

Per-item on-disk schema:
    <item_id>/
      info.txt                         # Bluesky-style conversation text
      <item_id>.<ext>                  # the CORRECT meme (stem = folder name)
      <distractor_1>.<ext>             # 3 distractor images (arbitrary filenames)
      <distractor_2>.<ext>
      <distractor_3>.<ext>
      labels.json                      # {"<filename>": "lexical"|"visual"|"random"} per distractor

`labels.json` is optional. If missing, distractors are tagged "unlabeled" — the
pipeline still runs but per-distractor-type accuracy will collapse into one bucket.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

DISTRACTOR_TYPES = ("lexical", "visual", "random")
IMAGE_EXTS = (".jpeg", ".jpg", ".png")


# ---------------------------------------------------------------------------
# Item dataclass
# ---------------------------------------------------------------------------

@dataclass
class MCItem:
    """A single meme-selection MC item."""
    id: str                                           # folder name
    conversation_text: str                            # raw info.txt contents
    answer: str                                       # gold letter A/B/C/D (landed on the correct meme after shuffle)

    images: dict[str, Path] = field(default_factory=dict)
    # Maps each letter to what kind of slot landed there:
    #   "correct" | "lexical" | "visual" | "random" | "unlabeled"
    slot_type_by_letter: dict[str, str] = field(default_factory=dict)
    gold_filename: str = ""                           # filename of the correct meme (audit trail)

    meta: dict = field(default_factory=dict)
    discourse: dict = field(default_factory=dict)       # discourse labels from benchmark
    context_images: list[str] = field(default_factory=list)  # CDN URLs of original/parent images


# ---------------------------------------------------------------------------
# Dataset loader — walks a folder tree
# ---------------------------------------------------------------------------

def _load_labels_json(item_dir: Path) -> dict[str, str]:
    """Return {distractor_filename: type_tag}. Returns {} if labels.json missing.

    Unknown types are coerced to 'unlabeled'.
    """
    p = item_dir / "labels.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{p}: invalid JSON — {e}") from None
    out: dict[str, str] = {}
    for fname, tag in raw.items():
        out[fname] = tag if tag in DISTRACTOR_TYPES else "unlabeled"
    return out


def load_dataset(data_root: str | Path, seed: int = 0) -> list[MCItem]:
    """Walk `data_root` for per-item folders. Each folder must contain:
      - info.txt                      (conversation text)
      - <folder_name>.<ext>           (correct meme)
      - 3 other image files           (distractors)
      - labels.json (optional)        (distractor type tags)

    Letter assignment (A/B/C/D) is a deterministic shuffle keyed by `seed`.
    `MCItem.answer` records which letter landed on the correct meme.
    `MCItem.slot_type_by_letter` records what type (correct/lexical/visual/random/unlabeled)
    landed on each letter — used for per-distractor-type accuracy.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    rng_global = random.Random(seed)
    items: list[MCItem] = []

    for item_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        info_path = item_dir / "info.txt"
        if not info_path.exists():
            continue  # not an item folder; skip silently

        images = sorted([p for p in item_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if len(images) != 4:
            raise ValueError(
                f"{item_dir}: expected exactly 4 image files, found {len(images)}"
            )

        correct = next((p for p in images if p.stem == item_dir.name), None)
        if correct is None:
            raise ValueError(
                f"{item_dir}: no image with stem matching folder name '{item_dir.name}'. "
                "The correct meme must be named <folder_name>.<ext>."
            )

        distractor_labels = _load_labels_json(item_dir)  # {filename: type}

        # Shuffle, assign letters, record slot types.
        # Per-item random seed based on item id → position bias 제거
        rng = random.Random(seed ^ hash(item_dir.name) & 0xFFFFFFFF)
        shuffled = images[:]
        rng.shuffle(shuffled)
        labels = dict(zip("ABCD", shuffled))
        gold_letter = next(k for k, v in labels.items() if v == correct)

        slot_type_by_letter: dict[str, str] = {}
        for letter, path in labels.items():
            if path == correct:
                slot_type_by_letter[letter] = "correct"
            else:
                slot_type_by_letter[letter] = distractor_labels.get(path.name, "unlabeled")

        # Load discourse labels if available
        discourse = {}
        disc_path = item_dir / "discourse.json"
        if disc_path.exists():
            try:
                discourse = json.loads(disc_path.read_text())
            except Exception:
                pass

        # Load context image URLs (original post / parent reply images)
        context_images: list[str] = []
        ctx_img_path = item_dir / "context_images.json"
        if ctx_img_path.exists():
            try:
                context_images = json.loads(ctx_img_path.read_text())
            except Exception:
                pass

        items.append(MCItem(
            id=item_dir.name,
            conversation_text=info_path.read_text(),
            answer=gold_letter,
            images=labels,
            slot_type_by_letter=slot_type_by_letter,
            gold_filename=correct.name,
            discourse=discourse,
            context_images=context_images,
        ))
    return items


# ---------------------------------------------------------------------------
# Prompts  (two versions, selectable via PROMPT_VERSION env var)
#
#   PROMPT_VERSION=conv  (default) — full conversation text
#   PROMPT_VERSION=disc            — discourse-label based, no raw conversation
# ---------------------------------------------------------------------------

import os as _os

# ── Version A: raw conversation text ────────────────────────────────────────
# Structure: context → images → guidelines → final instruction
# (Final instruction last = stronger directive effect)
PROMPT_CONV = (
    "You are given a social media conversation and four candidate meme images.\n\n"
    "{ctx_images_note}"
    "Conversation:\n{conv}\n\n"
    "The four candidate meme images are labeled A, B, C, and D (shown below).\n\n"
    "To select the correct meme, follow these guidelines:\n"
    "- The meme\'s tone is indicated in the conversation ({tone_note}). "
    "Prioritize images whose humor and emotional register match this tone.\n"
    "- Textual overlap is NOT evidence of correctness. Meme captions frequently "
    "express sarcasm, irony, or indirect commentary — the correct meme will often "
    "contain words that do NOT appear in the conversation.\n"
    "- Some distractors are designed to look textually relevant. "
    "Do NOT be fooled by surface-level text similarity.\n"
    "- Judge by visual meaning, cultural connotation, and emotional fit: "
    "does this meme\'s message make sense as a reply to this specific moment?\n\n"
    "Which image (A, B, C, or D) was actually posted as the meme reply here?\n"
    "Respond with ONLY one letter. Do NOT explain."
)

# ── Version C: no-context ablation (candidate images ONLY) ──────────────────
# Used for the context-ablation experiment: no conversation text, no context
# images, no tone note (tone is context-derived). Deliberately contains no
# "Conversation:" marker so every runner falls into its no-context branch and
# skips context-image interleaving.
PROMPT_NOCTX = (
    "You are given four candidate meme images labeled A, B, C, and D (shown below).\n\n"
    "Exactly one of them was actually posted as a meme reply in a real social "
    "media conversation. The conversation itself is NOT shown to you.\n\n"
    "Which image (A, B, C, or D) was actually posted as the meme reply?\n"
    "Respond with ONLY one letter. Do NOT explain."
)

# ── Version B: discourse-label based (no raw conversation text) ──────────────
PROMPT_DISC = (
    "You are an expert in internet meme culture and conversational discourse.\n\n"
    "A meme image was used as a reply in a social media conversation. "
    "Below is a description of the communicative role this meme played — "
    "NOT the conversation itself.\n\n"
    "{disc_context}\n\n"
    "Your task: select the ONE image (out of A, B, C, D) that best fulfills "
    "this communicative role as a meme reply. Focus on the meme\'s rhetorical "
    "and emotional function, not textual content.\n\n"
    "Respond with ONLY one letter: A, B, C, or D. Do NOT explain."
)

# Discourse function descriptions (human-readable)
_DISC_DESCRIPTIONS = {
    "Open.Demand":      "The meme opens a new exchange by posing a question or request.",
    "Open.Give":        "The meme opens a new exchange by offering information or opinion.",
    "React.Demand":     "The meme reacts to the prior message by asking a follow-up question.",
    "React.Give":       "The meme reacts to the prior message by providing information, opinion, or commentary.",
    "React.Acknowledge":"The meme acknowledges or validates the prior message (e.g. agreement, empathy).",
    "Sustain.Demand":   "The meme sustains the conversation by prompting further engagement.",
    "Sustain.Give":     "The meme sustains the conversation by adding related content.",
}

_STANCE_DESCRIPTIONS = {
    "sarcastic": "sarcastic",
    "humorous":  "humorous",
    "offensive": "offensive",
}


def _build_disc_context(discourse: dict) -> str:
    """Build the discourse context string from discourse labels."""
    meme_labels  = discourse.get("meme_reply")  or {}
    parent_labels = discourse.get("parent_reply") or {}

    lines = []

    # Parent reply discourse role (if available)
    parent_func = parent_labels.get("discourse_function")
    if parent_func:
        parent_desc = _DISC_DESCRIPTIONS.get(parent_func, parent_func)
        lines.append(f"The message this meme is replying to served as: {parent_desc}")

    # Meme discourse role
    meme_func = meme_labels.get("discourse_function")
    if meme_func:
        meme_desc = _DISC_DESCRIPTIONS.get(meme_func, meme_func)
        lines.append(f"The meme\'s discourse function: {meme_desc}")

    # Stance
    stance = meme_labels.get("stance") or {}
    active_stances = [_STANCE_DESCRIPTIONS[k] for k in ("sarcastic", "humorous", "offensive")
                      if stance.get(k)]
    if active_stances:
        lines.append(f"Tone/stance of the meme: {', '.join(active_stances)}")
    else:
        lines.append("Tone/stance: neutral")

    if not lines:
        return "(No discourse information available — select the most contextually appropriate meme.)"

    return "\n".join(lines)


def build_prompt(item: MCItem, version: str | None = None) -> str:
    """Return the text prompt for the given version.

    version: "conv" (default) | "disc"
    Falls back to PROMPT_VERSION env var, then "conv".
    """
    if version is None:
        version = _os.environ.get("PROMPT_VERSION", "conv").lower()

    if version == "disc":
        disc_ctx = _build_disc_context(item.discourse)
        return PROMPT_DISC.format(disc_context=disc_ctx)
    elif version == "noctx":
        return PROMPT_NOCTX
    else:
        # Extract tone from conversation text (e.g. "[Tone: Sarcastic, Humorous]")
        import re as _re
        tone_match = _re.search(r"\[Tone:\s*([^\]]+)\]", item.conversation_text)
        tone_note = tone_match.group(1).strip() if tone_match else "not specified"

        # Context images note (shown inline in conversation via [IMAGE:N] placeholders)
        if item.context_images:
            note = ""  # no preamble needed; images appear inline
        else:
            note = ""

        return PROMPT_CONV.format(
            conv=item.conversation_text.strip(),
            ctx_images_note=note,
            tone_note=tone_note,
        )


def split_conv_by_images(conversation_text: str, image_urls: list[str]) -> list[dict]:
    """
    Split conversation_text by [IMAGE:N] placeholders.
    Returns a list of content blocks in order:
      {"type": "text", "text": "..."}
      {"type": "image_url", "url": "https://..."}  (for OpenAI)
      {"type": "image_bytes", "url": "https://..."}  (for Anthropic)

    If no placeholders or no images, returns a single text block.
    """
    import re as _re
    if not image_urls:
        return [{"type": "text", "text": conversation_text}]

    # Match both old [IMAGE:N] and new [Label: IMAGE:N] patterns
    parts = _re.split(r"(\[[^\]]*IMAGE:\d+\])", conversation_text)
    blocks = []
    for part in parts:
        m = _re.search(r"IMAGE:(\d+)", part)
        if m:
            idx = int(m.group(1))
            if idx < len(image_urls):
                # Extract label (everything before "IMAGE:")
                label_match = _re.match(r"\[([^:]+?)(?:\s+\d+)?:", part)
                label = label_match.group(1).strip() if label_match else "Context image"
                blocks.append({"type": "image_ref", "url": image_urls[idx],
                               "index": idx, "label": label})
        elif part.strip():
            blocks.append({"type": "text", "text": part})
    return blocks if blocks else [{"type": "text", "text": conversation_text}]


# Legacy alias
PROMPT = PROMPT_CONV


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_letter(text: str | None) -> str | None:
    if not text:
        return None
    m = _LETTER_RE.search(text.strip().upper())
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def image_to_data_url(path: str | Path) -> str:
    """LEGACY: encodes the image at `path` as-is, no resize.

    Kept for back-compat with smoke_test.py and any external callers.
    For all production runners use `prepare_image_data_url` instead — it
    enforces the 512 px / JPEG normalization that the budget plan assumes.
    """
    path = Path(path)
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Image preparation (per Juyoung's spec — May 2026)
# ---------------------------------------------------------------------------
# Every runner downscales each image to 512 px on the longest edge and
# re-encodes as JPEG before sending it to the model. This:
#   - matches the budget plan's per-provider token estimates,
#   - keeps requests well under every provider's per-image limit
#     (Anthropic 5 MB, OpenAI 20 MB, Gemini 20 MB total request),
#   - is roughly free for downstream quality on meme-selection (the captions
#     are still legible at 512 px JPEG-88; the budget assumes this).
#
# Three flavors so each caller picks the format it needs:
#   prepare_image_pil(path)        -> PIL.Image      (HF runners; processor takes PIL)
#   prepare_image_bytes(path)      -> (bytes, mime)  (Anthropic, Gemini)
#   prepare_image_data_url(path)   -> "data:image/jpeg;base64,..." (OpenAI, REALLMS)

DEFAULT_MAX_EDGE = 512
DEFAULT_JPEG_QUALITY = 88


def prepare_image_pil(path: str | Path, max_edge: int = DEFAULT_MAX_EDGE):
    """Open `path`, downscale so `max(W, H) == max_edge`, return a PIL.Image.

    No-op on the resize step if the image is already smaller than max_edge.
    Always converts to RGB so RGBA / palette images don't break JPEG encode
    downstream.
    """
    from PIL import Image

    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    scale = max_edge / max(w, h)
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return im


def prepare_image_bytes(
    path: str | Path,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> tuple[bytes, str]:
    """Return `(jpeg_bytes, mime_type)` after the standard 512 px resize."""
    import io

    im = prepare_image_pil(path, max_edge=max_edge)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), "image/jpeg"


def prepare_image_data_url(
    path: str | Path,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> str:
    """Return a `data:image/jpeg;base64,...` URL after the standard 512 px resize."""
    raw, mime = prepare_image_bytes(path, max_edge=max_edge, quality=quality)
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"
