#!/usr/bin/env python3
"""
Benchmark dataset generation pipeline (04_benchmark/)

Four-option multiple choice:
  A) Original meme          <- correct answer
  B) Text distractor        <- remove captions and replace them with post/reply keywords
  C) Visual distractor      <- use a similar image and insert the original caption
  D) Easy distractor        <- random meme

Improvements:
  - Inpainting: LaMa GPU batch processing (simple-lama-inpainting, CUDA)
    * A100 baseline: batch_size=64, resolution=768 for fast, high-quality output
  - CLIP: ViT-L-14 (optimized for A100 and better similarity accuracy)
  - EasyOCR: GPU multi-stream parallel processing
  - Text rendering: precise placement inside each bbox to avoid overlap
  - Balanced monthly top-liked sampling via --target
  - Resume support after interrupted runs

Usage:
  pip install simple-lama-inpainting torch torchvision open_clip_torch \
              easyocr pillow numpy requests python-dotenv yake opencv-python

  source bech_env/bin/activate
  python benchmark_pipeline.py --sample 10             # test run
  python benchmark_pipeline.py --target 5000           # balanced 5,000-item target
  python benchmark_pipeline.py --month 2023-09         # one specific month
  python benchmark_pipeline.py --all                   # all records
  python benchmark_pipeline.py --target 5000 --workers 4  # multiprocessing
"""

import os
import json
import random
import argparse
import requests
import numpy as np
import io
import torch
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

# ================================================================
#  CONFIG
# ================================================================
CONFIG = {
    "labeled_jsonl":  "../03_filter_and_label/labeled_final/labeled_memes.jsonl",
    "dataset_dirs": [
        "../01_collection/meme_dataset_24_06",
        "../01_collection/meme_dataset_25_02",
        "../01_collection/meme_dataset",
    ],
    "output_dir":     "./benchmark_data",
    "sample_size":    10,
    "target_size":    5000,

    # CLIP similarity range
    "visual_sim_min": 0.65,
    "visual_sim_max": 0.78,

    # Maximum n-gram size for keyword extraction
    "keyword_max_ngram": 2,
    "keyword_top_n":     3,

    # Caption position thresholds as image-height ratios
    "caption_top_ratio":    0.30,
    "caption_bottom_ratio": 0.70,

    # Minimum OCR confidence
    "ocr_min_conf": 0.4,

    # Maximum size of the non-meme image pool
    # A100 has enough memory for a larger pool.
    "pool_max_size": 50000,

    # Date filter (inclusive; the "unknown" month is always excluded)
    "date_from": "2023-09",
    "date_to":   "2025-08",

    # A100 optimization parameters
    # LaMa batch size: A100(40/80GB) -> 64, RTX4090(24GB) -> 32
    "lama_batch_size": 64,

    # LaMa resolution: 768 gives more precise inpainting (~30% slower than 512)
    # Meme text regions are usually small, so 768 tends to work well.
    "lama_resolution": 768,

    # CLIP model: ViT-L-14 (recommended on A100; better similarity than B-32)
    # Embedding dim: 768, memory ~1.7GB
    "clip_model":      "ViT-L-14",
    "clip_pretrained": "openai",

    # CLIP batch size: A100 -> 128
    "clip_batch_size": 128,

    # EasyOCR GPU batch size
    "ocr_gpu_batch":   8,

    # Multiprocessing worker count (--workers default)
    "default_workers": 4,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE}")

# ================================================================
#  Utilities
# ================================================================

def get_month(record: dict) -> str:
    val = (record.get("original_post") or {}).get("created_at", "")
    if not val:
        val = record.get("created_at", "")
    if val:
        try:
            ts = float(val)
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
        except (ValueError, TypeError):
            if len(str(val)) >= 7:
                return str(val)[:7]
    return "unknown"


def load_image(path: str = None, url: str = None) -> Image.Image | None:
    if path and Path(path).exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    if url:
        try:
            r = requests.get(url, timeout=10,
                             headers={"User-Agent": "MemeResearchBot/1.0"})
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
    return None


def check_cdn_url(url: str, timeout: int = 8) -> bool:
    """Check whether the CDN URL is actually reachable with a HEAD request."""
    if not url:
        return False
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD",
              headers={"User-Agent": "MemeResearchBot/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


# ================================================================
#  Data loading
# ================================================================

def load_labeled(jsonl_path: str) -> list:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    print(f"[LOAD] Labeled records: {len(records):,}")
    return records


def load_non_meme_pool(dataset_dirs: list, labeled_uids: set) -> list:
    pool = []
    for dir_str in dataset_dirs:
        d = Path(dir_str)
        records_dir = d / "records"
        if not records_dir.exists():
            continue
        for fp in records_dir.glob("*.json"):
            try:
                r = json.loads(fp.read_text("utf-8"))
                uid = r.get("uid", "")
                if uid in labeled_uids:
                    continue
                for img in (r.get("meme_reply") or {}).get("images", []) or []:
                    local = img.get("local_path", "")
                    url   = img.get("source_url") or img.get("url", "")
                    full_path = str(d / local) if local else None
                    if full_path and Path(full_path).exists():
                        pool.append({"path": full_path, "url": url, "uid": uid})
                    elif url:
                        pool.append({"path": None, "url": url, "uid": uid})
            except Exception:
                pass
    print(f"[LOAD] Non-meme image pool: {len(pool):,}")
    if len(pool) > CONFIG["pool_max_size"]:
        pool = random.sample(pool, CONFIG["pool_max_size"])
        print(f"[LOAD] Sampled pool: {len(pool):,}")
    return pool


# ================================================================
#  Balanced monthly top-N sampling
# ================================================================

def select_top_per_month(by_month: dict, total: int) -> list:
    """
    Return all candidates after sorting each month by descending like count.
    The actual per-month target count is tracked in the pipeline loop.
    Once a month reaches its quota, the loop stops trying records from that month.
    """
    months = sorted(k for k in by_month.keys() if k != "unknown")
    n_months = max(len(months), 1)
    per_month = max(1, total // n_months)

    print(f"[SELECT] {n_months} months | per-month target {per_month} | total target {total:,}")
    for m, c in sorted((m, len(v)) for m, v in by_month.items() if m != "unknown"):
        print(f"         {m}: {c} candidates")

    # Return all candidates sorted by descending likes within each month.
    # Interleave the monthly lists in round-robin order.
    sorted_by_month = {
        m: sorted(v, key=lambda r: (r.get("meme_reply") or {}).get("like_count", 0), reverse=True)
        for m, v in by_month.items() if m != "unknown"
    }
    candidates = []
    max_len = max((len(v) for v in sorted_by_month.values()), default=0)
    for i in range(max_len):
        for m in months:
            recs = sorted_by_month.get(m, [])
            if i < len(recs):
                candidates.append(recs[i])

    return candidates, per_month, months


# ================================================================
#  OCR
# ================================================================

_ocr_reader = None

def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        use_gpu = DEVICE == "cuda"
        _ocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        print(f"[OCR] EasyOCR loaded (gpu={use_gpu})")
    return _ocr_reader


def extract_captions(img: Image.Image) -> list:
    reader = get_ocr()
    arr = np.array(img)
    results = reader.readtext(arr)
    h = img.height
    captions = []
    for bbox, text, conf in results:
        if conf < CONFIG["ocr_min_conf"]:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        cy = (y1 + y2) / 2
        if cy / h <= CONFIG["caption_top_ratio"]:
            position = "top"
        elif cy / h >= CONFIG["caption_bottom_ratio"]:
            position = "bottom"
        else:
            position = "middle"
        captions.append({
            "text": text,
            "bbox": [x1, y1, x2, y2],
            "conf": float(conf),
            "position": position,
        })
    return captions


# ================================================================
#  LaMa Inpainting (GPU batch processing)
# ================================================================

_lama_model = None

def get_lama():
    global _lama_model
    if _lama_model is None:
        print(f"[LaMa] Loading model... (device={DEVICE})")
        from simple_lama_inpainting import SimpleLama
        _lama_model = SimpleLama()
        # Move to GPU.
        if DEVICE == "cuda" and hasattr(_lama_model, "model"):
            _lama_model.model = _lama_model.model.to(DEVICE)
        print("[LaMa] Loaded")
    return _lama_model


def build_mask(img: Image.Image, captions: list, pad: int = 10) -> Image.Image:
    """Create a caption-region mask (white = inpaint region)."""
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for cap in captions:
        x1, y1, x2, y2 = cap["bbox"]
        draw.rectangle([
            max(0, x1 - pad), max(0, y1 - pad),
            min(img.width, x2 + pad), min(img.height, y2 + pad)
        ], fill=255)
    return mask


def inpaint_single_lama(img: Image.Image, mask: Image.Image) -> Image.Image:
    """Inpaint a single image with LaMa."""
    lama = get_lama()
    orig_size = img.size
    res = CONFIG["lama_resolution"]

    # Resize to the recommended LaMa resolution.
    img_r  = img.resize((res, res), Image.LANCZOS)
    mask_r = mask.resize((res, res), Image.NEAREST)

    # Run LaMa.
    result = lama(img_r, mask_r)

    # Restore the original size.
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)
    return result


def inpaint_captions(img: Image.Image, captions: list) -> Image.Image:
    """
    Inpaint caption regions.
    Priority 1: LaMa GPU (simple-lama-inpainting)
    Priority 2: OpenCV TELEA (radius 3 to reduce bleeding)
    Priority 3: fill with the modal background color
    """
    if not captions:
        return img

    mask = build_mask(img, captions, pad=12)

    # Priority 1: LaMa GPU.
    try:
        return inpaint_single_lama(img, mask)
    except Exception as e:
        print(f"  [WARN] LaMa failed: {e}")

    # Priority 2: OpenCV TELEA with radius 3 to reduce bleeding.
    try:
        import cv2
        img_cv  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask_cv = np.array(mask)
        result_cv = cv2.inpaint(img_cv, mask_cv, 3, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"  [WARN] OpenCV failed: {e}")

    # Priority 3: fill with the modal background color.
    result = img.copy()
    arr    = np.array(img)
    draw   = ImageDraw.Draw(result)
    for cap in captions:
        x1, y1, x2, y2 = cap["bbox"]
        pad = 14
        border_pixels = []
        for region_slice in [
            arr[max(0, y1 - pad * 3):max(0, y1 - pad),   max(0, x1):min(img.width, x2)],
            arr[min(img.height, y2 + pad):min(img.height, y2 + pad * 3), max(0, x1):min(img.width, x2)],
            arr[max(0, y1):min(img.height, y2), max(0, x1 - pad * 3):max(0, x1 - pad)],
            arr[max(0, y1):min(img.height, y2), min(img.width, x2 + pad):min(img.width, x2 + pad * 3)],
        ]:
            if region_slice.size > 0:
                border_pixels.append(region_slice.reshape(-1, 3))

        if border_pixels:
            all_px = np.concatenate(border_pixels)
            # Modal color, usually the most accurate choice for flat backgrounds.
            vals, counts = np.unique(all_px.reshape(-1, 3), axis=0, return_counts=True)
            fill = tuple(vals[counts.argmax()].astype(int))
        else:
            fill = (255, 255, 255)

        draw.rectangle([
            max(0, x1 - pad), max(0, y1 - pad),
            min(img.width, x2 + pad), min(img.height, y2 + pad)
        ], fill=fill)
    return result


# ================================================================
#  Text insertion (precise bbox placement)
# ================================================================

def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Windows/Fonts/arialbd.ttf",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def analyze_caption_style(img: Image.Image, caption: dict) -> dict:
    arr = np.array(img)
    x1, y1, x2, y2 = caption["bbox"]
    h_img, w_img = arr.shape[:2]

    bbox_h = max(y2 - y1, 1)
    font_size = max(14, int(bbox_h * 0.85))
    font_size = min(font_size, max(14, int(h_img * 0.08)))

    region = arr[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
    if region.size == 0:
        return {"font_size": font_size, "text_color": (255, 255, 255),
                "bg_color": None, "position": caption.get("position", "bottom"),
                "bbox": caption["bbox"]}

    pixels = region.reshape(-1, 3)
    brightness = pixels.mean(axis=1)
    bright_pixels = pixels[brightness > 180]
    dark_pixels   = pixels[brightness < 80]
    text_color = (255, 255, 255) if len(bright_pixels) > len(dark_pixels) else (0, 0, 0)

    pad = 5
    above = arr[max(0, y1 - pad):y1, max(0, x1):min(w_img, x2)]
    bg_color = tuple(np.mean(above.reshape(-1, 3), axis=0).astype(int)) if above.size > 0 else None

    return {
        "font_size": font_size,
        "text_color": tuple(text_color),
        "bg_color": bg_color,
        "position": caption.get("position", "bottom"),
        "bbox": caption["bbox"],
    }


def wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        try:
            bbox = draw.textbbox((0, 0), test, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(test) * 8
        if tw <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [text]


def draw_caption_at_bbox(img: Image.Image, text: str, style: dict) -> Image.Image:
    """
    Insert text precisely inside the bbox to avoid overlap.
    This fixes the older draw_caption_on_image behavior that centered text globally.
    """
    result = img.copy()
    draw   = ImageDraw.Draw(result)

    x1, y1, x2, y2 = style["bbox"]
    font_size  = style.get("font_size", 16)
    text_color = style.get("text_color", (255, 255, 255))

    # Cap font size relative to image dimensions.
    max_by_height = max(14, int(img.height * 0.08))
    max_by_width  = max(14, int(img.width  * 0.05))
    font_size = min(font_size, max_by_height, max_by_width)
    font = get_font(font_size)

    # Wrap text by bbox width.
    max_w = max(x2 - x1, 40)
    display_text = text.upper() if text_color == (255, 255, 255) else text
    lines = wrap_text(display_text, font, max_w, draw)

    try:
        sample_bbox = draw.textbbox((0, 0), "Ag", font=font)
        line_h = sample_bbox[3] - sample_bbox[1] + 3
    except Exception:
        line_h = font_size + 3

    total_h = line_h * len(lines)

    # Vertically center inside the bbox.
    y_start = y1 + max(0, ((y2 - y1) - total_h) // 2)

    shadow_color = (0, 0, 0) if text_color[0] > 127 else (255, 255, 255)

    for i, line in enumerate(lines):
        try:
            lbbox = draw.textbbox((0, 0), line, font=font)
            lw = lbbox[2] - lbbox[0]
        except Exception:
            lw = len(line) * font_size // 2

        # Horizontally center inside the bbox.
        x = x1 + max(0, (max_w - lw) // 2)
        y = y_start + i * line_h

        # Eight-direction shadow.
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1),
                       (0, 2), (2, 0), (-2, 0), (0, -2)]:
            draw.text((x + dx, y + dy), line, font=font, fill=shadow_color)
        draw.text((x, y), line, font=font, fill=text_color)

    return result


# ================================================================
#  Keyword extraction
# ================================================================

def extract_keywords_multi(texts: list, n: int) -> list:
    """Return a list of n keywords."""
    combined = " ".join(t for t in texts if t and t.strip())
    if not combined.strip():
        return ["reaction"] * n
    try:
        import yake
        kw_extractor = yake.KeywordExtractor(
            n=CONFIG["keyword_max_ngram"],
            top=max(n, CONFIG["keyword_top_n"]),
            dedupLim=0.7,
        )
        keywords = [k for k, s in kw_extractor.extract_keywords(combined)]
        if keywords:
            return keywords
    except Exception:
        pass
    words = combined.split()
    return [" ".join(words[i:i+2]) for i in range(0, max(n*2, len(words)), 2)][:n] or ["reaction"] * n


# ================================================================
#  CLIP
# ================================================================

_clip_model = None
_clip_preprocess = None

def get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import open_clip
        print(f"[CLIP] Loading model... (device={DEVICE})")
        model_name = CONFIG["clip_model"]
        pretrained = CONFIG["clip_pretrained"]
        print(f"[CLIP] Loading {model_name}/{pretrained}... (device={DEVICE})")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _clip_model = _clip_model.to(DEVICE)
        _clip_model.eval()
        if DEVICE == "cuda":
            _clip_model = _clip_model.half()  # A100: fp16 roughly doubles throughput.
        print(f"[CLIP] Loaded (fp16={DEVICE=='cuda'})")
    return _clip_model, _clip_preprocess


def get_image_embedding(img: Image.Image) -> np.ndarray:
    model, preprocess = get_clip()
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    if DEVICE == "cuda":
        tensor = tensor.half()
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().float().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_visual_distractor(orig_emb: np.ndarray,
                           pool_embs: list,
                           pool_items: list) -> dict | None:
    candidates = []
    for emb, item in zip(pool_embs, pool_items):
        sim = cosine_sim(orig_emb, np.array(emb))
        if CONFIG["visual_sim_min"] <= sim <= CONFIG["visual_sim_max"]:
            candidates.append((sim, item))

    if not candidates:
        scored = sorted(
            [(cosine_sim(orig_emb, np.array(e)), it) for e, it in zip(pool_embs, pool_items)],
            key=lambda x: abs(x[0] - 0.72)
        )
        candidates = scored[:10]

    return random.choice(candidates)[1] if candidates else None


# ================================================================
#  Distractor generation
# ================================================================

def make_text_distractor(orig_img: Image.Image,
                         captions: list,
                         record: dict) -> Image.Image | None:
    """
    B) Text distractor:
    Remove original captions with LaMa and insert post/reply keywords at each bbox.
    """
    if not captions:
        return None

    # Analyze style before inpainting.
    styles = [analyze_caption_style(orig_img, cap) for cap in captions]

    # LaMa inpainting.
    cleaned = inpaint_captions(orig_img, captions)

    # Collect text.
    texts = []
    for field in ["original_post", "parent_reply", "meme_reply"]:
        t = (record.get(field) or {}).get("text") or ""
        if t:
            texts.append(t)

    keywords = extract_keywords_multi(texts, len(captions))

    # Insert text at the precise position for each bbox.
    result = cleaned
    for i, (cap, style) in enumerate(zip(captions, styles)):
        kw = keywords[i % len(keywords)] if keywords else "reaction"
        x1, y1, x2, y2 = cap["bbox"]
        max_chars = max(4, (x2 - x1) // max(style.get("font_size", 16) // 2, 1))
        kw = kw[:max_chars]
        result = draw_caption_at_bbox(result, kw, style=style)

    return result


def make_visual_distractor(orig_img: Image.Image,
                           orig_captions: list,
                           orig_emb: np.ndarray,
                           pool_embs: list,
                           pool_items: list) -> Image.Image | None:
    """
    C) Visual distractor:
    Use a similar non-meme image, remove its text with LaMa, and insert the original caption.
    """
    item = find_visual_distractor(orig_emb, pool_embs, pool_items)
    if not item:
        return None

    vis_img = load_image(item.get("path"), item.get("url"))
    if not vis_img:
        return None

    if not orig_captions:
        return vis_img

    # Remove OCR-detected text from the similar image.
    vis_captions = extract_captions(vis_img)
    if vis_captions:
        vis_img = inpaint_captions(vis_img, vis_captions)

    # Combine original caption text.
    caption_text = " ".join(c["text"] for c in orig_captions)[:60]

    # Insert with the original style.
    main_cap = max(orig_captions, key=lambda c: c.get("conf", 0))
    style = analyze_caption_style(orig_img, main_cap)
    # Rescale the bbox for the target visual-distractor image size.
    bbox_ratio = [
        main_cap["bbox"][0] / orig_img.width,
        main_cap["bbox"][1] / orig_img.height,
        main_cap["bbox"][2] / orig_img.width,
        main_cap["bbox"][3] / orig_img.height,
    ]
    style["bbox"] = [
        int(bbox_ratio[0] * vis_img.width),
        int(bbox_ratio[1] * vis_img.height),
        int(bbox_ratio[2] * vis_img.width),
        int(bbox_ratio[3] * vis_img.height),
    ]
    return draw_caption_at_bbox(vis_img, caption_text, style=style)


def make_easy_distractor(all_records: list,
                         exclude_uid: str,
                         dataset_dirs: list) -> Image.Image | None:
    """D) Easy distractor: random meme."""
    candidates = [r for r in all_records if r.get("uid") != exclude_uid]
    if not candidates:
        return None

    for _ in range(10):
        r = random.choice(candidates)
        imgs = (r.get("meme_reply") or {}).get("images", []) or []
        if not imgs:
            continue
        img_info = imgs[0]
        local = img_info.get("local_path", "")
        url   = img_info.get("source_url") or img_info.get("url", "")

        for d in dataset_dirs:
            candidate_path = Path(d) / local if local else None
            if candidate_path and candidate_path.exists():
                img = load_image(str(candidate_path), url)
                if img:
                    return img
        if url:
            img = load_image(None, url)
            if img:
                return img
    return None


# ================================================================
#  CLIP pool embeddings (cache, GPU batch)
# ================================================================

def build_pool_embeddings(pool_items: list, output_dir: Path):
    cache_path = output_dir / "pool_embeddings.npy"
    cache_meta = output_dir / "pool_items.json"

    if cache_path.exists() and cache_meta.exists():
        print("[CACHE] Loading pool embedding cache...")
        embs  = np.load(str(cache_path))
        items = json.loads(cache_meta.read_text())
        return embs.tolist(), items

    print(f"[CLIP] Building pool embeddings ({len(pool_items)} items, device={DEVICE})...")
    model, preprocess = get_clip()
    embs  = []
    items = []
    batch_size = CONFIG["clip_batch_size"] if DEVICE == "cuda" else 32

    imgs_batch  = []
    items_batch = []

    def flush_batch():
        if not imgs_batch:
            return
        try:
            tensors = torch.stack([preprocess(im) for im in imgs_batch]).to(DEVICE)
            if DEVICE == "cuda":
                tensors = tensors.half()
            with torch.no_grad():
                batch_emb = model.encode_image(tensors)
                batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
            for emb, it in zip(batch_emb.cpu().float().numpy(), items_batch):
                embs.append(emb)
                items.append(it)
        except Exception as e:
            print(f"  [WARN] Batch failed: {e}")
        imgs_batch.clear()
        items_batch.clear()

    for i, item in enumerate(pool_items):
        if i % 500 == 0:
            print(f"  {i}/{len(pool_items)}")
        img = load_image(item.get("path"), item.get("url"))
        if not img:
            continue
        imgs_batch.append(img)
        items_batch.append(item)
        if len(imgs_batch) >= batch_size:
            flush_batch()

    flush_batch()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), np.array(embs))
    cache_meta.write_text(json.dumps(items, ensure_ascii=False))
    print(f"  Done: {len(embs)}")
    return embs, items


# ================================================================
#  Single benchmark item generation
# ================================================================

def make_benchmark_item(record: dict,
                        all_records: list,
                        pool_items: list,
                        pool_embs: list,
                        output_dir: Path,
                        idx: int) -> dict | None:
    uid = record.get("uid", "unknown")
    print(f"\n[{idx}] {uid[:30]}")

    # Exclude deleted/private posts (in_archive=False means missing from the archive).
    orig = record.get("original_post") or {}
    parent = record.get("parent_reply") or {}
    if orig and orig.get("in_archive") is False:
        print("  [SKIP] original_post deleted/private")
        return None
    if parent and parent.get("in_archive") is False:
        print("  [SKIP] parent_reply deleted/private")
        return None

    imgs = (record.get("meme_reply") or {}).get("images", []) or []
    if not imgs:
        print("  [SKIP] No meme image")
        return None

    # Check CDN reachability. Images can disappear for deleted accounts/posts.
    meme_cdn = imgs[0].get("source_url") or imgs[0].get("url", "")
    if meme_cdn and not check_cdn_url(meme_cdn):
        print("  [SKIP] Meme image CDN is unreachable (deleted post)")
        return None
    # Check original-post images when present.
    orig_imgs = (orig.get("images") or [])
    if orig_imgs:
        orig_cdn = orig_imgs[0].get("source_url") or orig_imgs[0].get("url", "")
        if orig_cdn and not check_cdn_url(orig_cdn):
            print("  [SKIP] Original-post image CDN is unreachable (deleted post)")
            return None

    img_info = imgs[0]
    local    = img_info.get("local_path", "")
    url      = img_info.get("source_url") or img_info.get("url", "")

    orig_img = None
    for d in CONFIG["dataset_dirs"]:
        if local:
            p = Path(d) / local
            if p.exists():
                orig_img = load_image(str(p), url)
                break
    if not orig_img:
        orig_img = load_image(None, url)

    if not orig_img:
        print("  [SKIP] Failed to load original image")
        return None

    print("  [OCR] Extracting captions...")
    captions = extract_captions(orig_img)
    print(f"        {len(captions)} captions: {[c['text'][:20] for c in captions]}")

    print("  [CLIP] Embedding...")
    try:
        orig_emb = get_image_embedding(orig_img)
    except Exception as e:
        print(f"  [SKIP] CLIP failed: {e}")
        return None

    item_dir = output_dir / uid
    item_dir.mkdir(parents=True, exist_ok=True)

    # A) Original
    save_image(orig_img, item_dir / "A_original.jpg")
    print("  [A] Original saved")

    # B) Text distractor
    print("  [B] Creating text distractor (LaMa inpainting)...")
    text_dist = make_text_distractor(orig_img, captions, record)
    if text_dist:
        save_image(text_dist, item_dir / "B_text_distractor.jpg")
        print("  [B] Saved")
    else:
        print("  [SKIP] Failed to create B - excluding from benchmark")
        return None

    # C) Visual distractor
    vis_dist = None
    if pool_embs:
        print("  [C] Creating visual distractor...")
        vis_dist = make_visual_distractor(orig_img, captions, orig_emb,
                                          pool_embs, pool_items)
        if vis_dist:
            save_image(vis_dist, item_dir / "C_visual_distractor.jpg")
            print("  [C] Saved")
        else:
            print("  [SKIP] Failed to create C - excluding from benchmark")
            return None
    else:
        print("  [SKIP] No CLIP pool - excluding from benchmark")
        return None

    # D) Easy distractor
    easy_dist = make_easy_distractor(all_records, uid, CONFIG["dataset_dirs"])
    if easy_dist:
        save_image(easy_dist, item_dir / "D_easy_distractor.jpg")
        print("  [D] Saved")
    else:
        print("  [SKIP] Failed to create D - excluding from benchmark")
        return None

    meta = {
        "uid":    uid,
        "month":  get_month(record),
        "orig_url": url,
        "captions": captions,
        "options": {
            "A": "A_original.jpg",
            "B": "B_text_distractor.jpg",
            "C": "C_visual_distractor.jpg",
            "D": "D_easy_distractor.jpg",
        },
        "answer": "A",
        "context": {
            # Original post.
            "original_post_text":   (record.get("original_post") or {}).get("text", ""),
            "original_post_uri":    (record.get("original_post") or {}).get("uri", ""),
            "original_post_images": [
                img.get("source_url", "") or img.get("url", "")
                for img in ((record.get("original_post") or {}).get("images") or [])
                if img.get("source_url") or img.get("url")
            ],
            # Post quoted by the original post.
            "quoted_post_text":   (record.get("quoted_post") or {}).get("text", ""),
            "quoted_post_images": [
                img.get("source_url", "") or img.get("url", "")
                for img in ((record.get("quoted_post") or {}).get("images") or [])
                if img.get("source_url") or img.get("url")
            ],
            "quoted_post_external_title": (record.get("quoted_post") or {}).get("external_title", ""),
            "quoted_post_external_url":   (record.get("quoted_post") or {}).get("external_url", ""),
            # Original-post external link, such as a news/video title.
            "original_post_external_title": (record.get("original_post") or {}).get("external_title", ""),
            "original_post_external_url":   (record.get("original_post") or {}).get("external_url", ""),
            # Ancestor reply chain, oldest first.
            "ancestor_chain": [
                {
                    "text":           node.get("text", ""),
                    "images":         [
                        img.get("source_url", "") or img.get("url", "")
                        for img in (node.get("images") or [])
                        if img.get("source_url") or img.get("url")
                    ],
                    "external_title": node.get("external_title", ""),
                    "external_url":   node.get("external_url", ""),
                    "quoted_post_text": (node.get("quoted_post") or {}).get("text", ""),
                    "quoted_post_images": [
                        img.get("source_url", "") or img.get("url", "")
                        for img in ((node.get("quoted_post") or {}).get("images") or [])
                        if img.get("source_url") or img.get("url")
                    ],
                }
                for node in (record.get("ancestor_chain") or [])
            ],
            # Parent reply directly above the meme.
            "parent_reply_text":   (record.get("parent_reply") or {}).get("text", ""),
            "parent_reply_images": [
                img.get("source_url", "") or img.get("url", "")
                for img in ((record.get("parent_reply") or {}).get("images") or [])
                if img.get("source_url") or img.get("url")
            ],
            "parent_reply_external_title": (record.get("parent_reply") or {}).get("external_title", ""),
            # Meme reply text.
            "meme_text": (record.get("meme_reply") or {}).get("text", ""),
        },
        "labels": record.get("discourse_labels", {}),
    }
    (item_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


# ================================================================
#  Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate the benchmark dataset")
    parser.add_argument("--sample", type=int, default=CONFIG["sample_size"],
                        help=f"Test sample size (default: {CONFIG['sample_size']})")
    parser.add_argument("--target", type=int, default=None,
                        help="Balanced monthly target size (for example, 5000)")
    parser.add_argument("--all",    action="store_true", help="Process all records")
    parser.add_argument("--month",  default=None, help="Specific month (for example, 2023-09)")
    parser.add_argument("--output", default=CONFIG["output_dir"])
    parser.add_argument("--no-clip", action="store_true",
                        help="Skip CLIP pool building; mainly useful for debugging")
    parser.add_argument("--workers", type=int, default=CONFIG["default_workers"],
                        help=f"Parallel worker count (default: {CONFIG['default_workers']})")
    parser.add_argument("--lama-res", type=int, default=CONFIG["lama_resolution"],
                        help=f"LaMa resolution 256/512/768 (default: {CONFIG['lama_resolution']})")
    args = parser.parse_args()
    if args.lama_res != CONFIG["lama_resolution"]:
        CONFIG["lama_resolution"] = args.lama_res
        print(f"[CONFIG] LaMa resolution override: {args.lama_res}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled = load_labeled(CONFIG["labeled_jsonl"])
    labeled_uids = {r.get("uid", "") for r in labeled}

    by_month = defaultdict(list)
    for r in labeled:
        by_month[get_month(r)].append(r)

    # Apply the date filter.
    date_from = CONFIG["date_from"]
    date_to   = CONFIG["date_to"]
    filtered_months = {
        m: recs for m, recs in by_month.items()
        if m != "unknown" and date_from <= m <= date_to
    }
    removed = sum(len(v) for k, v in by_month.items() if k not in filtered_months)
    by_month = defaultdict(list, filtered_months)
    print(f"[FILTER] Date range {date_from} ~ {date_to}: {sum(len(v) for v in by_month.values()):,} records ({removed:,} excluded)")
    labeled = [r for recs in by_month.values() for r in recs]

    # Select targets.
    if args.month:
        candidates = by_month.get(args.month, [])
        candidates.sort(
            key=lambda r: (r.get("meme_reply") or {}).get("like_count", 0),
            reverse=True
        )
        per_month_quota = None
        print(f"[{args.month}] {len(candidates):,} records")
    elif args.target:
        candidates, per_month_quota, months_list = select_top_per_month(by_month, args.target)
    elif args.all:
        candidates = labeled
        candidates.sort(
            key=lambda r: (r.get("meme_reply") or {}).get("like_count", 0),
            reverse=True
        )
        per_month_quota = None
    else:
        candidates = sorted(
            labeled,
            key=lambda r: (r.get("meme_reply") or {}).get("like_count", 0),
            reverse=True
        )[:args.sample]
        per_month_quota = None

    print(f"Records to process: {len(candidates):,}")

    non_meme_pool = load_non_meme_pool(CONFIG["dataset_dirs"], labeled_uids)

    if not args.no_clip and non_meme_pool:
        pool_embs, pool_items = build_pool_embeddings(non_meme_pool, output_dir)
    else:
        pool_embs, pool_items = [], non_meme_pool

    # Skip already processed UIDs for resume support.
    done_uids = set()
    summary_path = output_dir / "benchmark_summary.jsonl"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_uids.add(json.loads(line)["uid"])
                except Exception:
                    pass
    if done_uids:
        print(f"[RESUME] Skipping already processed records: {len(done_uids):,}")
        candidates = [r for r in candidates if r.get("uid", "") not in done_uids]

    # ThreadPoolExecutor shares the GPU CUDA context and parallelizes I/O well.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    lock = threading.Lock()
    success = 0
    total_count = len(candidates)

    def process_one(args_tuple):
        i, record = args_tuple
        return make_benchmark_item(
            record, labeled, pool_items, pool_embs,
            output_dir, i + 1 + len(done_uids)
        )

    # GPU: thread parallelism with shared CUDA context; CPU: limit worker count.
    n_workers = args.workers if DEVICE == "cuda" else min(args.workers, 2)
    print(f"[RUN] Workers: {n_workers} (device={DEVICE})")

    # Track monthly targets only in --target mode.
    month_success = defaultdict(int)
    month_quota   = per_month_quota if args.target else None
    target_total  = args.target if args.target else None

    def quota_reached(record):
        """Return True if this record's month has already reached its quota."""
        if month_quota is None:
            return False
        return month_success[get_month(record)] >= month_quota

    with open(summary_path, "a", encoding="utf-8") as f_summary:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for i, rec in enumerate(candidates):
                if quota_reached(rec):
                    continue
                if target_total and (success + len(done_uids)) >= target_total:
                    break
                futures[executor.submit(process_one, (i, rec))] = rec

            for future in as_completed(futures):
                rec  = futures[future]
                meta = future.result()
                with lock:
                    if meta:
                        success += 1
                        month_success[get_month(rec)] += 1
                        f_summary.write(
                            json.dumps(meta, ensure_ascii=False, default=str) + "\n"
                        )
                        f_summary.flush()
                    done_so_far = success + len(done_uids)
                    if done_so_far % 100 == 0 and done_so_far > 0:
                        print(f"  [PROGRESS] {done_so_far:,} complete")
                        if month_quota:
                            for m in sorted(month_success):
                                print(f"         {m}: {month_success[m]}/{month_quota}")
                    if target_total and done_so_far >= target_total:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

    print(f"\n{'='*55}")
    print(f"  Done: {success}/{total_count} (cumulative: {success + len(done_uids):,})")
    print(f"  Output: {output_dir.resolve()}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
