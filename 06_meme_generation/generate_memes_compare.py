#!/usr/bin/env python3
"""
generate_memes_compare.py

Generate meme images for the top N sampled records from labeled_memes.jsonl
using Version A (visual_description only) and Version B (conversation context),
then write the results to an HTML viewer.

Total image count: N x 2 versions x number of configured models.

Usage:
  python generate_memes_compare.py --n 3
  python generate_memes_compare.py --n 50 --out ./generated
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

LABELED_JSONL = "../03_filter_and_label/labeled_final/labeled_memes.jsonl"
DEFAULT_N     = 3
DEFAULT_OUT   = "./generated"
IMAGE_SIZE    = "1024x1024"
MODELS        = ["gpt-image-2"]

# ════════════════════════════════════════════════════════════════
#  Data loading
# ════════════════════════════════════════════════════════════════

def load_top_n(jsonl_path: str, n: int, seed: int = 42) -> list[dict]:
    import random
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            vd = ((r.get("discourse_labels") or {}).get("meme_reply") or {}).get("visual") or {}
            if not vd.get("visual_description"):
                continue
            # Keep only records with richer conversational context.
            if not r.get("parent_reply"):
                continue
            records.append(r)
    random.seed(seed)
    random.shuffle(records)
    return records[:n]

# ════════════════════════════════════════════════════════════════
#  Context builders
# ════════════════════════════════════════════════════════════════

def build_context_text(record: dict) -> str:
    parts = []
    orig = record.get("original_post") or {}
    if orig.get("text"):
        parts.append(f"[Original post]: {orig['text']}")
    if orig.get("external_title"):
        parts.append(f"[Link]: {orig['external_title']}")
    quoted = record.get("quoted_post") or {}
    if quoted.get("text"):
        parts.append(f"[Quoted post]: {quoted['text']}")
    for i, node in enumerate(record.get("ancestor_chain") or []):
        if node.get("text"):
            parts.append(f"[Thread reply {i+1}]: {node['text']}")
    parent = record.get("parent_reply") or {}
    if parent.get("text"):
        parts.append(f"[Parent reply]: {parent['text']}")
    meme = record.get("meme_reply") or {}
    if meme.get("text"):
        parts.append(f"[Meme reply text]: {meme['text']}")
    return "\n".join(parts) if parts else "(no text context)"


def build_context_images(record: dict) -> list[str]:
    urls = []
    for field in ["original_post", "quoted_post"]:
        sub = record.get(field) or {}
        for img in (sub.get("images") or []):
            url = img.get("source_url") or img.get("url", "")
            if url and url not in urls:
                urls.append(url)
    for node in (record.get("ancestor_chain") or []):
        for img in (node.get("images") or []):
            url = img.get("source_url") or img.get("url", "")
            if url and url not in urls:
                urls.append(url)
    return urls[:4]


def fetch_image_b64(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            raw = r.read()
        return base64.b64encode(raw).decode()
    except Exception:
        return None


def build_thread_display(record: dict) -> list[dict]:
    thread = []
    orig = record.get("original_post") or {}
    if orig:
        imgs = [img.get("source_url") or img.get("url","") for img in (orig.get("images") or [])]
        thread.append({"role": "Original Post", "text": orig.get("text",""), "images": imgs, "color": "#1d4ed8"})
    quoted = record.get("quoted_post")
    if quoted and quoted.get("text"):
        imgs = [img.get("source_url") or img.get("url","") for img in (quoted.get("images") or [])]
        thread.append({"role": "Quoted Post", "text": quoted.get("text",""), "images": imgs, "color": "#7c3aed"})
    for i, anc in enumerate(record.get("ancestor_chain") or []):
        if anc.get("text"):
            imgs = [img.get("source_url") or img.get("url","") for img in (anc.get("images") or [])]
            thread.append({"role": f"Reply {i+1}", "text": anc.get("text",""), "images": imgs, "color": "#0f766e"})
    parent = record.get("parent_reply")
    if parent and parent.get("text"):
        imgs = [img.get("source_url") or img.get("url","") for img in (parent.get("images") or [])]
        thread.append({"role": "Parent Reply", "text": parent.get("text",""), "images": imgs, "color": "#b45309"})
    meme = record.get("meme_reply") or {}
    imgs = [img.get("source_url") or img.get("url","") for img in (meme.get("images") or [])]
    thread.append({"role": "Meme Reply ★", "text": meme.get("text",""), "images": imgs, "color": "#be123c", "is_meme": True})
    return thread

# ════════════════════════════════════════════════════════════════
#  Prompt builders
# ════════════════════════════════════════════════════════════════

def build_prompt_a(record: dict) -> str:
    """Version A: visual_description only (w/o context)"""
    vd = ((record.get("discourse_labels") or {}).get("meme_reply") or {}).get("visual") or {}
    description = vd.get("visual_description", "")
    return (
        f"Create an internet meme image based on this visual description:\n"
        f"{description}\n\n"
        f"Requirements:\n"
        f"- Include a short, punchy caption overlaid on the image (top/bottom text or speech bubble style)\n"
        f"- The caption and image must work together to deliver a single clear joke or reaction\n"
        f"- Use high contrast text that is easy to read at a glance\n"
        f"- Make it look like a real internet meme someone would actually share"
    )


def build_prompt_b(record: dict) -> str:
    """Version B: full conversation context (w/ context)"""
    ctx_text = build_context_text(record)
    dl = (record.get("discourse_labels") or {}).get("meme_reply") or {}
    stance = dl.get("stance") or {}
    stance_tags = [k for k, v in stance.items() if v]
    stance_str = ", ".join(stance_tags) if stance_tags else "neutral"

    vd = ((record.get("discourse_labels") or {}).get("meme_reply") or {}).get("visual") or {}
    description = vd.get("visual_description", "")
    return (
        f"Create an internet meme image based on this visual description:\n{description}\n\n"
        f"The meme should work as a reply in this social media conversation.\n\n"
        f"Conversation:\n{ctx_text}\n\n"
        f"The meme should:\n"
        f"- Directly react to the conversation above (the root post, parent reply, or key moment)\n"
        f"- Reflect this tone/stance: {stance_str}\n"
        f"- Include a short, punchy caption overlaid on the image\n"
        f"- Use unexpected humor: irony, exaggeration, reversal, or visual-verbal mismatch\n"
        f"- Feel specific to THIS conversation — not a generic meme that could go anywhere\n"
        f"- Do NOT copy or quote text directly from the conversation. Express the meaning visually and with original caption wording\n"
        f"- Make it look like a real internet meme someone would actually post as a reply"
    )

# ════════════════════════════════════════════════════════════════
#  Image generation (client.images.generate)
# ════════════════════════════════════════════════════════════════

def generate_image(
    prompt: str,
    model: str,
    client: OpenAI,
    out_path: Path,
) -> str | None:
    if out_path.exists():
        print(f"    already exists, skipping")
        return str(out_path)

    try:
        resp = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=IMAGE_SIZE,
            
        )
        # Prefer b64_json; fall back to downloading the returned URL.
        item = resp.data[0]
        if item.b64_json:
            out_path.write_bytes(base64.b64decode(item.b64_json))
        else:
            import urllib.request as _ur
            with _ur.urlopen(item.url, timeout=30) as r:
                out_path.write_bytes(r.read())
        out_path.with_suffix(".txt").write_text(prompt, encoding="utf-8")
        print(f"    ✓ saved → {out_path.name}")
        return str(out_path)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

# ════════════════════════════════════════════════════════════════
#  HTML generation
# ════════════════════════════════════════════════════════════════

def img_to_b64_src(path: str | None) -> str:
    if not path or not Path(path).exists():
        return ""
    data = Path(path).read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


def generate_html(results: list[dict]) -> str:
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    cards_html = ""
    for i, r in enumerate(results):
        uid           = r["uid"]
        thread        = r["thread"]
        visual_desc   = r.get("visual_description", "")

        # Thread HTML
        thread_html = ""
        for msg in thread:
            role    = msg["role"]
            text    = msg.get("text") or ""
            imgs    = msg.get("images") or []
            color   = msg["color"]
            is_meme = msg.get("is_meme", False)
            imgs_html = "".join(
                f'<img src="{url}" alt="" onerror="this.style.display=\'none\'">'
                for url in imgs if url
            )
            thread_html += f"""
            <div class="msg {'meme-msg' if is_meme else ''}">
              <div class="role-tag" style="background:{color}">{role}</div>
              {f'<p class="msg-text">{text}</p>' if text else ''}
              {f'<div class="msg-imgs">{imgs_html}</div>' if imgs_html else ''}
            </div>"""

        # Generated image grid: A/model, B/model.
        combos = [
            ("A", "gpt-image-2", "w/o context"),
            ("B", "gpt-image-2", "w/ context"),
        ]
        grid_html = ""
        for version, model, label in combos:
            key      = f"{version}_{model}"
            img_path = r["images"].get(key)
            prompt   = r["prompts"].get(key, "")
            src      = img_to_b64_src(img_path)
            img_tag  = (
                f'<img src="{src}" alt="generated meme">'
                if src else
                '<div class="no-img">Generation failed</div>'
            )
            grid_html += f"""
            <div class="gen-cell">
              <div class="gen-label">
                <span class="ver-badge ver-{'a' if version=='A' else 'b'}">{label}</span>
                <span class="model-tag">{model}</span>
              </div>
              {img_tag}
              <details class="prompt-detail">
                <summary>prompt</summary>
                <pre>{prompt[:400]}{'...' if len(prompt)>400 else ''}</pre>
              </details>
            </div>"""

        cards_html += f"""
        <div class="card">
          <div class="card-header">
            <span class="card-num">#{i+1}</span>
            <span class="card-uid">{uid}</span>
          </div>
          <div class="card-body">
            <div class="thread-col">
              <div class="col-title">Conversation Context</div>
              {f'<div class="visual-desc"><span class="vd-label">Visual Description</span>{visual_desc}</div>' if visual_desc else ""}
              <div class="thread">{thread_html}</div>
            </div>
            <div class="gen-col">
              <div class="col-title">Generated Memes — Quality · Relevance · Humor (1–5)</div>
              <div class="gen-grid">{grid_html}</div>
            </div>
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meme Generation Comparison</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #f5f0e8; --surface: #fff; --surface2: #f9f6f0;
    --border: #e0d8cc; --text: #1a1410; --dim: #7a6e62;
    --accent: #c84b2f;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Fraunces', Georgia, serif; }}

  header {{
    background: var(--text); color: var(--bg);
    padding: 20px 40px; display: flex; align-items: baseline; gap: 16px;
  }}
  header h1 {{ font-size: 1.4rem; font-weight: 700; }}
  header h1 em {{ color: #e8a87c; font-style: italic; }}
  .meta {{ font-family: 'DM Mono', monospace; font-size: 0.72rem; opacity: 0.6; margin-left: auto; }}

  .legend {{
    display: flex; gap: 24px; padding: 12px 40px;
    background: var(--surface); border-bottom: 2px solid var(--text);
    font-family: 'DM Mono', monospace; font-size: 0.75rem; flex-wrap: wrap; align-items: center;
  }}
  .ver-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-weight: 500; font-size: 0.7rem; font-family: 'DM Mono', monospace;
  }}
  .ver-a {{ background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }}
  .ver-b {{ background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }}

  main {{ padding: 32px 40px; display: flex; flex-direction: column; gap: 40px; max-width: 1600px; margin: 0 auto; }}

  .card {{
    background: var(--surface); border: 2px solid var(--text); border-radius: 4px; overflow: hidden;
  }}
  .card-header {{
    background: var(--text); color: var(--bg);
    padding: 10px 20px; display: flex; align-items: center; gap: 12px;
  }}
  .card-num {{ font-size: 1.1rem; font-weight: 700; color: #e8a87c; font-family: 'DM Mono', monospace; }}
  .card-uid {{ font-family: 'DM Mono', monospace; font-size: 0.7rem; opacity: 0.6; }}

  .card-body {{ display: grid; grid-template-columns: 340px 1fr; }}

  .thread-col {{
    border-right: 2px solid var(--text); padding: 20px;
    background: var(--surface2); max-height: 820px; overflow-y: auto;
  }}
  .gen-col {{ padding: 20px; }}

  .col-title {{
    font-family: 'DM Mono', monospace; font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--dim);
    margin-bottom: 14px; padding-bottom: 6px; border-bottom: 1px solid var(--border);
  }}

  .visual-desc {{
    background: #fefce8; border: 1px solid #fde68a; border-radius: 4px;
    padding: 8px 12px; margin-bottom: 12px; font-size: 0.78rem;
    line-height: 1.5; color: #78350f;
  }}
  .vd-label {{
    display: block; font-family: 'DM Mono', monospace; font-size: 0.62rem;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em;
    color: #92400e; margin-bottom: 4px;
  }}
  .thread {{ display: flex; flex-direction: column; gap: 10px; }}
  .msg {{ border-left: 3px solid var(--border); padding: 8px 10px; }}
  .meme-msg {{ border-left-color: #c84b2f; background: #fff5f3; }}
  .role-tag {{
    display: inline-block; padding: 1px 7px; border-radius: 2px;
    font-size: 0.62rem; font-weight: 500; color: #fff;
    margin-bottom: 5px; font-family: 'DM Mono', monospace;
  }}
  .msg-text {{ font-size: 0.82rem; line-height: 1.5; }}
  .msg-imgs {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }}
  .msg-imgs img {{ max-width: 120px; max-height: 120px; object-fit: cover; border-radius: 3px; border: 1px solid var(--border); }}

  .gen-grid {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px;
  }}
  .gen-cell {{
    display: flex; flex-direction: column; gap: 8px;
    border: 1px solid var(--border); border-radius: 4px;
    padding: 10px; background: var(--surface2);
  }}
  .gen-label {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
  .model-tag {{
    font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--dim);
    background: var(--bg); padding: 1px 6px; border-radius: 2px; border: 1px solid var(--border);
  }}
  .gen-cell img {{
    width: 100%; aspect-ratio: 1; object-fit: contain;
    border-radius: 3px; border: 1px solid var(--border); background: #f0ede8;
  }}
  .no-img {{
    width: 100%; aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
    background: #f0ede8; color: var(--dim); font-size: 0.8rem;
    border-radius: 3px; font-family: 'DM Mono', monospace;
  }}
  .prompt-detail summary {{
    font-family: 'DM Mono', monospace; font-size: 0.63rem; color: var(--dim);
    cursor: pointer; user-select: none; margin-top: 4px;
  }}
  .prompt-detail pre {{
    font-family: 'DM Mono', monospace; font-size: 0.63rem;
    white-space: pre-wrap; word-break: break-word;
    background: var(--bg); padding: 8px; border-radius: 3px;
    margin-top: 6px; line-height: 1.5; color: var(--dim);
  }}

  @media (max-width: 1100px) {{
    .card-body {{ grid-template-columns: 1fr; }}
    .thread-col {{ border-right: none; border-bottom: 2px solid var(--text); max-height: 400px; }}
    .gen-grid {{ grid-template-columns: repeat(2, 1fr); }}
    main, header, .legend {{ padding-left: 16px; padding-right: 16px; }}
  }}
</style>
</head>
<body>
<header>
  <h1>Meme Generation <em>Comparison</em></h1>
  <span class="meta">{timestamp}</span>
</header>
<div class="legend">
  <span><span class="ver-badge ver-a">w/o context</span> Version A — visual description only</span>
  <span><span class="ver-badge ver-b">w/ context</span> Version B — full conversation context</span>
  <span style="margin-left:auto;color:var(--dim);font-size:0.72rem">Evaluate: Quality · Relevance · Humor (1–5)</span>
</div>
<main>{cards_html}</main>
</body>
</html>"""

# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",  default=LABELED_JSONL)
    parser.add_argument("--n",      type=int, default=DEFAULT_N)
    parser.add_argument("--out",    default=DEFAULT_OUT)
    parser.add_argument("--delay",  type=float, default=2.0)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    client  = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = args.n
    pool_size = target * 3  # Use a larger pool to allow for generation failures.
    print(f"[LOAD] Targeting {target} complete samples from a pool of {pool_size}...")
    all_records = load_top_n(args.jsonl, pool_size, seed=args.seed)
    print(f"  Pool size: {len(all_records)}\n")

    results = []
    attempted = 0

    for record in all_records:
        if len(results) >= target:
            break

        attempted += 1
        uid = record.get("uid", f"sample_{attempted}")
        print(f"[{len(results)+1}/{target}] {uid} (attempt {attempted})")

        sample_dir = out_dir / uid
        sample_dir.mkdir(exist_ok=True)

        prompt_a = build_prompt_a(record)
        prompt_b = build_prompt_b(record)

        images  = {}
        prompts = {}
        success = True

        for model in MODELS:
            model_slug = model.replace("-","")

            # Version A
            key_a  = f"A_{model}"
            path_a = sample_dir / f"A_{model_slug}.png"
            print(f"  [A/{model}]", end=" ", flush=True)
            result_a = generate_image(prompt_a, model, client, path_a)
            images[key_a]  = result_a
            prompts[key_a] = prompt_a
            if not result_a:
                success = False
            time.sleep(args.delay)

            # Version B
            key_b  = f"B_{model}"
            path_b = sample_dir / f"B_{model_slug}.png"
            print(f"  [B/{model}]", end=" ", flush=True)
            result_b = generate_image(prompt_b, model, client, path_b)
            images[key_b]  = result_b
            prompts[key_b] = prompt_b
            if not result_b:
                success = False
            time.sleep(args.delay)

        if not success:
            print(f"  -> Partial failure, skipping ({uid})")
            print()
            continue

        vd = ((record.get("discourse_labels") or {}).get("meme_reply") or {}).get("visual") or {}
        results.append({
            "uid":            uid,
            "thread":         build_thread_display(record),
            "images":         images,
            "prompts":        prompts,
            "visual_description": vd.get("visual_description", ""),
        })
        print(f"  -> Complete ({len(results)}/{target})")
        print()

    print(f"\nCompleted: {len(results)}/{target} samples (attempted: {attempted})")

    # Save HTML
    ts       = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    html_out = out_dir / f"comparison_{ts}.html"
    html_out.write_text(generate_html(results), encoding="utf-8")

    # Save JSON
    json_out = out_dir / f"results_{ts}.json"
    json_out.write_text(
        json.dumps([{k: v for k, v in r.items() if k != "thread"} for r in results],
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

    print(f"\n{'='*55}")
    print(f"  HTML: {html_out.resolve()}")
    print(f"  JSON: {json_out.resolve()}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
