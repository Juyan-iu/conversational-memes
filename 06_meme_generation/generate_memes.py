#!/usr/bin/env python3
"""
06_meme_generation/generate_memes.py

labeled_memes.jsonl에서 좋아요 상위 50개를 뽑아
두 가지 방식으로 밈 이미지를 생성합니다.

Version A: visual_description만 사용 → DALL-E 3
Version B: 대화 컨텍스트 + stance + discourse_function → GPT-4o 프롬프트 생성 → DALL-E 3

사용법:
  python generate_memes.py
  python generate_memes.py --n 50 --out ./generated
  python generate_memes.py --version a  # Version A만
  python generate_memes.py --version b  # Version B만
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────
LABELED_JSONL  = "../03_filter_and_label/labeled_final/labeled_memes.jsonl"
DEFAULT_N      = 50
DEFAULT_OUT    = "./generated"
GPT_MODEL      = "gpt-4o"
IMAGE_SIZE     = "1024x1024"


# ════════════════════════════════════════════════════════════════
#  데이터 로딩 & 샘플링
# ════════════════════════════════════════════════════════════════

def load_top_n(jsonl_path: str, n: int) -> list[dict]:
    """좋아요 수 상위 n개 레코드 반환."""
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
            # visual_description 있는 것만
            vd = (r.get("discourse_labels", {}) or {})
            vd = (vd.get("meme_reply") or {})
            vd = (vd.get("visual") or {})
            if not vd.get("visual_description"):
                continue
            like_count = (r.get("meme_reply") or {}).get("like_count", 0) or 0
            records.append((like_count, r))

    records.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in records[:n]]


# ════════════════════════════════════════════════════════════════
#  컨텍스트 빌더
# ════════════════════════════════════════════════════════════════

def build_context_text(record: dict) -> str:
    """대화 컨텍스트 텍스트 구성."""
    ctx_parts = []

    orig = record.get("original_post") or {}
    if orig.get("text"):
        ctx_parts.append(f"[Root post]: {orig['text']}")
    if orig.get("external_title"):
        ctx_parts.append(f"[Link]: {orig['external_title']}")

    quoted = record.get("quoted_post") or {}
    if quoted.get("text"):
        ctx_parts.append(f"[Quoted post]: {quoted['text']}")

    ancestor_chain = record.get("ancestor_chain") or []
    for i, node in enumerate(ancestor_chain):
        if node.get("text"):
            ctx_parts.append(f"[Thread reply {i+1}]: {node['text']}")

    parent = record.get("parent_reply") or {}
    if parent.get("text"):
        ctx_parts.append(f"[Parent reply]: {parent['text']}")

    meme = record.get("meme_reply") or {}
    if meme.get("text"):
        ctx_parts.append(f"[Meme reply text]: {meme['text']}")

    return "\n".join(ctx_parts) if ctx_parts else "(no text context)"


def build_context_images(record: dict) -> list[str]:
    """대화 컨텍스트 이미지 URL 수집 (최대 4개)."""
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
    """CDN URL → base64."""
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            raw = r.read()
        return base64.b64encode(raw).decode()
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
#  Version A: visual_description → DALL-E 3
# ════════════════════════════════════════════════════════════════

def generate_version_a(record: dict, client: OpenAI, out_dir: Path) -> str | None:
    """visual_description으로 밈 생성 (GPT-4o image generation)."""
    uid = record.get("uid", "unknown")
    out_path = out_dir / f"{uid}_a.png"
    if out_path.exists():
        print(f"  [A] {uid[:30]} already exists, skipping")
        return str(out_path)

    dl = (record.get("discourse_labels") or {})
    mr = (dl.get("meme_reply") or {})
    visual = (mr.get("visual") or {})
    description = visual.get("visual_description", "")
    if not description:
        print(f"  [A] {uid[:30]} no visual_description, skipping")
        return None

    prompt = (
        f"Create an internet meme image based on this description:\n{description}\n\n"
        "Include a funny, culturally relevant caption as text on the image in meme style "
        "(e.g. top/bottom text, impact font style). Make it look like a real internet meme."
    )

    try:
        resp = client.responses.create(
            model=GPT_MODEL,
            input=prompt,
            tools=[{"type": "image_generation", "size": IMAGE_SIZE}],
        )
        img_b64 = next(
            item.result for item in resp.output
            if item.type == "image_generation_call"
        )
        img_bytes = base64.b64decode(img_b64)
        out_path.write_bytes(img_bytes)
        prompt_path = out_path.with_suffix(".txt")
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"  [A] {uid[:30]} ✓ saved")
        return str(out_path)
    except Exception as e:
        print(f"  [A] {uid[:30]} ERROR: {e}")
        return None


# ════════════════════════════════════════════════════════════════
#  Version B: context + stance → GPT-4o → DALL-E 3
# ════════════════════════════════════════════════════════════════

def generate_version_b(record: dict, client: OpenAI, out_dir: Path) -> str | None:
    """대화 컨텍스트 + stance + 이미지를 GPT-4o에 직접 넣어 밈 생성."""
    uid = record.get("uid", "unknown")
    out_path = out_dir / f"{uid}_b.png"
    if out_path.exists():
        print(f"  [B] {uid[:30]} already exists, skipping")
        return str(out_path)

    dl = (record.get("discourse_labels") or {})
    mr = (dl.get("meme_reply") or {})
    stance = mr.get("stance") or {}
    discourse_fn = mr.get("discourse_function", "unknown")
    stance_tags = [k for k, v in stance.items() if v]
    stance_str = ", ".join(stance_tags) if stance_tags else "neutral"

    ctx_text = build_context_text(record)
    ctx_img_urls = build_context_images(record)

    prompt = (
        f"Create an internet meme image that fits as a reply to this social media conversation.\n\n"
        f"Conversation:\n{ctx_text}\n\n"
        f"The meme should have these characteristics:\n"
        f"- Discourse function: {discourse_fn}\n"
        f"- Tone/stance: {stance_str}\n\n"
        "Include a funny, culturally relevant caption as text on the image in meme style. "
        "Make it look like a real internet meme that someone would actually post."
    )

    # 컨텍스트 이미지 포함 (Fix: input_text/input_image는 message의 content 안에 넣어야 함)
    content_items: list[dict] = [{"type": "input_text", "text": prompt}]
    for url in ctx_img_urls:
        b64 = fetch_image_b64(url)
        if b64:
            content_items.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}"
            })
    input_content = [{"role": "user", "content": content_items}]

    # 프롬프트 저장
    prompt_path = out_path.with_suffix(".txt")
    prompt_path.write_text(prompt, encoding="utf-8")

    try:
        resp = client.responses.create(
            model=GPT_MODEL,
            input=input_content,
            tools=[{"type": "image_generation", "size": IMAGE_SIZE}],
        )
        img_b64 = next(
            item.result for item in resp.output
            if item.type == "image_generation_call"
        )
        img_bytes = base64.b64decode(img_b64)
        out_path.write_bytes(img_bytes)
        print(f"  [B] {uid[:30]} ✓ saved")
        return str(out_path)
    except Exception as e:
        print(f"  [B] {uid[:30]} ERROR: {e}")
        return None


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Meme Generation Pipeline")
    parser.add_argument("--jsonl",    default=LABELED_JSONL)
    parser.add_argument("--n",        type=int, default=DEFAULT_N)
    parser.add_argument("--out",      default=DEFAULT_OUT)
    parser.add_argument("--version",  choices=["a", "b", "both"], default="both")
    parser.add_argument("--delay",    type=float, default=1.0,
                        help="Delay between API calls (seconds)")
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    out_dir = Path(args.out)
    (out_dir / "version_a").mkdir(parents=True, exist_ok=True)
    (out_dir / "version_b").mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] Loading top {args.n} records by like count...")
    records = load_top_n(args.jsonl, args.n)
    print(f"  Loaded: {len(records)} records")

    results = {"version_a": [], "version_b": []}

    for i, record in enumerate(records):
        uid = record.get("uid", "?")
        like_count = (record.get("meme_reply") or {}).get("like_count", 0)
        print(f"\n[{i+1}/{len(records)}] {uid[:35]} (likes: {like_count})")

        if args.version in ("a", "both"):
            path_a = generate_version_a(record, client, out_dir / "version_a")
            results["version_a"].append({"uid": uid, "path": path_a})
            time.sleep(args.delay)

        if args.version in ("b", "both"):
            path_b = generate_version_b(record, client, out_dir / "version_b")
            results["version_b"].append({"uid": uid, "path": path_b})
            time.sleep(args.delay)

    # 결과 저장
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )

    # 요약
    a_ok = sum(1 for r in results["version_a"] if r["path"])
    b_ok = sum(1 for r in results["version_b"] if r["path"])
    print(f"\n{'='*50}")
    print(f"  Version A: {a_ok}/{len(records)} generated")
    print(f"  Version B: {b_ok}/{len(records)} generated")
    print(f"  Results:   {results_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
