#!/usr/bin/env python3
"""
Convert benchmark_data/ → vlm-eval compatible dataset format.

benchmark_data layout (input):
    <uid>/
        A_original.jpg
        B_text_distractor.jpg
        C_visual_distractor.jpg
        D_easy_distractor.jpg
        meta.json

vlm-eval layout (output):
    <output_root>/
        <uid>/
            info.txt                  # conversation context text
            <uid>.jpg                 # correct meme (stem = folder name)
            B_text_distractor.jpg     # lexical distractor
            C_visual_distractor.jpg   # visual distractor
            D_easy_distractor.jpg     # random distractor
            labels.json               # distractor type tags

info.txt format (matches placeholder example):
    ===== ROOT POST =====
    <original_post_text>

    ===== REPLY POST =====
    <meme_reply_text>

    Parent Comment:
    <parent_reply_text>

    ===== URL =====
    https://bsky.app/profile/.../post/...

Usage:
    python convert_to_vlmeval.py --input ./benchmark_data --output ./vlmeval_data
    python convert_to_vlmeval.py --input ./benchmark_data --output ./vlmeval_data --limit 100
"""

import json
import shutil
import argparse
from pathlib import Path


def uri_to_url(uri: str) -> str:
    """at://did/.../rkey → https://bsky.app/profile/did/post/rkey"""
    if not uri:
        return ""
    try:
        parts = uri.replace("at://", "").split("/")
        if len(parts) >= 3:
            return f"https://bsky.app/profile/{parts[0]}/post/{parts[2]}"
    except Exception:
        pass
    return uri


def build_info_txt(meta: dict) -> str:
    """
    Build info.txt from meta.json context.
    Full conversation thread: root post → quoted post → ancestor chain → parent reply → meme text
    NO discourse labels (those go to discourse.json for prompt version B).
    """
    ctx = meta.get("context") or {}

    orig_text        = (ctx.get("original_post_text") or ctx.get("original_post") or "").strip()
    orig_uri         = (ctx.get("original_post_uri") or "").strip()
    orig_ext_title   = (ctx.get("original_post_external_title") or "").strip()
    orig_ext_url     = (ctx.get("original_post_external_url") or "").strip()

    quoted_text      = (ctx.get("quoted_post_text") or "").strip()
    quoted_ext_title = (ctx.get("quoted_post_external_title") or "").strip()
    quoted_ext_url   = (ctx.get("quoted_post_external_url") or "").strip()

    ancestor_chain   = ctx.get("ancestor_chain") or []

    parent_text      = (ctx.get("parent_reply_text") or ctx.get("parent_reply") or "").strip()
    parent_ext_title = (ctx.get("parent_reply_external_title") or "").strip()

    meme_text        = (ctx.get("meme_text") or "").strip()

    orig_url = uri_to_url(orig_uri) or uri_to_url(meta.get("uri", ""))

    parts = []

    # 1. Root post
    if orig_text or orig_ext_title:
        parts.append("===== ROOT POST =====")
        if orig_text:
            parts.append(orig_text)
        if orig_ext_title:
            link = f" ({orig_ext_url})" if orig_ext_url else ""
            parts.append(f"[Link: {orig_ext_title}{link}]")

    # 2. Quoted post
    if quoted_text or quoted_ext_title:
        parts.append("\n===== QUOTED POST =====")
        if quoted_text:
            parts.append(quoted_text)
        if quoted_ext_title:
            link = f" ({quoted_ext_url})" if quoted_ext_url else ""
            parts.append(f"[Link: {quoted_ext_title}{link}]")

    # 3. Ancestor chain (오래된 순)
    for i, node in enumerate(ancestor_chain):
        n_text      = (node.get("text") or "").strip()
        n_ext_title = (node.get("external_title") or "").strip()
        n_ext_url   = (node.get("external_url") or "").strip()
        n_qt_text   = (node.get("quoted_post_text") or "").strip()
        if n_text or n_ext_title or n_qt_text:
            parts.append(f"\n===== THREAD REPLY [{i+1}] =====")
            if n_text:
                parts.append(n_text)
            if n_ext_title:
                link = f" ({n_ext_url})" if n_ext_url else ""
                parts.append(f"[Link: {n_ext_title}{link}]")
            if n_qt_text:
                parts.append(f"[Quoted: {n_qt_text[:150]}]")

    # 4. Parent reply
    if parent_text or parent_ext_title:
        parts.append("\n===== PARENT REPLY =====")
        if parent_text:
            parts.append(parent_text)
        if parent_ext_title:
            parts.append(f"[Link: {parent_ext_title}]")

    # 5. Meme reply text
    if meme_text:
        parts.append("\n===== MEME REPLY TEXT =====")
        parts.append(meme_text)

    # 6. URL
    if orig_url:
        parts.append(f"\n===== URL =====\n{orig_url}")

    if not parts:
        parts.append("(no text context — model should rely on images)")

    return "\n".join(parts) + "\n"


def convert(input_dir: Path, output_dir: Path, limit: int | None = None):
    summary_path = input_dir / "benchmark_summary.jsonl"
    if not summary_path.exists():
        print(f"[ERROR] {summary_path} not found")
        return

    # Load all metas
    all_metas = []
    with open(summary_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_metas.append(json.loads(line))
                except Exception:
                    pass

    print(f"[LOAD] {len(all_metas)} items")
    if limit:
        all_metas = all_metas[:limit]
        print(f"[LIMIT] {len(all_metas)} items")

    output_dir.mkdir(parents=True, exist_ok=True)

    success = skip = 0
    for meta in all_metas:
        uid      = meta.get("uid", "")
        item_src = input_dir / uid
        item_dst = output_dir / uid

        # Check all source files exist
        orig_src = item_src / "A_original.jpg"
        lex_src  = item_src / "B_text_distractor.jpg"
        vis_src  = item_src / "C_visual_distractor.jpg"
        rnd_src  = item_src / "D_easy_distractor.jpg"

        if not all(p.exists() for p in [orig_src, lex_src, vis_src, rnd_src]):
            print(f"  [SKIP] {uid} — missing image files")
            skip += 1
            continue

        item_dst.mkdir(parents=True, exist_ok=True)

        # 1. info.txt
        info_txt = build_info_txt(meta)
        (item_dst / "info.txt").write_text(info_txt, encoding="utf-8")

        # 2. Correct meme: A_original.jpg → <uid>.jpg (stem must match folder name)
        shutil.copy2(orig_src, item_dst / f"{uid}.jpg")

        # 3. Distractors (keep original filenames)
        shutil.copy2(lex_src, item_dst / "B_text_distractor.jpg")
        shutil.copy2(vis_src, item_dst / "C_visual_distractor.jpg")
        shutil.copy2(rnd_src, item_dst / "D_easy_distractor.jpg")

        # 4. labels.json
        labels = {
            "B_text_distractor.jpg":   "lexical",
            "C_visual_distractor.jpg": "visual",
            "D_easy_distractor.jpg":   "random",
        }
        (item_dst / "labels.json").write_text(
            json.dumps(labels, indent=2) + "\n", encoding="utf-8"
        )

        success += 1

    print(f"\n{'='*50}")
    print(f"  Converted: {success} items")
    print(f"  Skipped:   {skip} items")
    print(f"  Output:    {output_dir.resolve()}")
    print(f"{'='*50}")
    print(f"\nRun eval:")
    print(f"  cd vlm-eval/")
    print(f"  python run_eval.py --model gpt-4o --data-root ../{output_dir} --out results/gpt4o_predictions.jsonl")
    print(f"  python run_eval.py --model claude-sonnet-4-5 --data-root ../{output_dir} --out results/claude_predictions.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark_data to vlm-eval format")
    parser.add_argument("--input",  default="./benchmark_data", help="benchmark_data directory")
    parser.add_argument("--output", default="./vlmeval_data",   help="output directory")
    parser.add_argument("--limit",  type=int, default=None,     help="limit number of items")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output), args.limit)


if __name__ == "__main__":
    main()
