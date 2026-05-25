#!/usr/bin/env python3
"""
MEMECONV Benchmark Statistics Extractor
Usage:
  python3 extract_benchmark_stats.py
  python3 extract_benchmark_stats.py --summary ../04_benchmark/benchmark_data/benchmark_summary.jsonl
  python3 extract_benchmark_stats.py --bench-dir ../04_benchmark/benchmark_data/
  python3 extract_benchmark_stats.py --out-json bench_stats.json
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

DEFAULT_SUMMARY  = "../04_benchmark/benchmark_data/benchmark_summary.jsonl"
DEFAULT_BENCHDIR = "../04_benchmark/benchmark_data/"

VALID_MONTHS = [
    "2023-09", "2023-10", "2023-11", "2023-12",
    "2024-01", "2024-02", "2024-03", "2024-04",
    "2024-05", "2024-06", "2024-07", "2024-08",
    "2024-09", "2024-10", "2024-11", "2024-12",
    "2025-01", "2025-02", "2025-03", "2025-04",
    "2025-05", "2025-06", "2025-07", "2025-08",
]



def load_summary_jsonl(path):
    """benchmark_summary.jsonl 로드"""
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] line {i+1}: {e}", file=sys.stderr)
    return records


def load_from_bench_dir(bench_dir):
    """benchmark_data/ 폴더에서 UID 하위 meta.json 파일들을 모두 읽음"""
    records = []
    bench_path = Path(bench_dir)

    # benchmark_summary.jsonl 우선 시도
    summary = bench_path / "benchmark_summary.jsonl"
    if summary.exists():
        print(f"  Found benchmark_summary.jsonl")
        return load_summary_jsonl(summary)

    # fallback: 각 UID 폴더의 meta.json
    for meta_fp in sorted(bench_path.glob("*/meta.json")):
        try:
            records.append(json.loads(meta_fp.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [warn] {meta_fp}: {e}", file=sys.stderr)
    return records


# ════════════════════════════════════════════════════════════════
#  통계 계산
# ════════════════════════════════════════════════════════════════

def compute_benchmark_stats(records, source_label):
    N = len(records)
    if N == 0:
        print("  [warn] 벤치마크 레코드가 없습니다.", file=sys.stderr)
        return {}

    month_counts = defaultdict(int)

    # context field coverage
    n_has_original_text    = 0
    n_has_quoted_text      = 0
    n_has_parent_text      = 0
    n_has_meme_text        = 0
    n_has_ancestor_chain   = 0
    n_has_ctx_image        = 0   # original_post_images OR quoted_post_images OR parent_reply_images
    n_has_original_image   = 0
    n_has_quoted_image     = 0
    n_has_parent_image     = 0
    n_has_external_link    = 0   # original/quoted external title

    # distractor presence
    n_has_text_distractor   = 0
    n_has_visual_distractor = 0
    n_has_easy_distractor   = 0
    n_all_four_options      = 0

    # captions (OCR)
    n_has_caption    = 0
    caption_lengths  = []

    # answer distribution (sanity check — should all be A)
    answer_counts = defaultdict(int)

    # labels (discourse)
    label_keys = defaultdict(int)

    for rec in records:
        # month
        m = rec.get("month", "unknown")
        if m in VALID_MONTHS:
            month_counts[m] += 1

        # context fields
        ctx = rec.get("context") or {}

        orig_text   = ctx.get("original_post_text") or ""
        quoted_text = ctx.get("quoted_post_text") or ""
        parent_text = ctx.get("parent_reply_text") or ""
        meme_text   = ctx.get("meme_text") or ""
        ancestors   = ctx.get("ancestor_chain") or []

        if orig_text.strip():   n_has_original_text  += 1
        if quoted_text.strip(): n_has_quoted_text    += 1
        if parent_text.strip(): n_has_parent_text    += 1
        if meme_text.strip():   n_has_meme_text      += 1
        if ancestors:           n_has_ancestor_chain += 1

        orig_imgs   = ctx.get("original_post_images") or []
        quoted_imgs = ctx.get("quoted_post_images") or []
        parent_imgs = ctx.get("parent_reply_images") or []

        if orig_imgs:   n_has_original_image += 1
        if quoted_imgs: n_has_quoted_image   += 1
        if parent_imgs: n_has_parent_image   += 1
        if orig_imgs or quoted_imgs or parent_imgs:
            n_has_ctx_image += 1

        ext_title = (ctx.get("original_post_external_title") or
                     ctx.get("quoted_post_external_title") or "")
        if ext_title.strip():
            n_has_external_link += 1

        # options / distractors
        opts = rec.get("options") or {}
        has_b = bool(opts.get("B"))
        has_c = bool(opts.get("C"))
        has_d = bool(opts.get("D"))
        if has_b: n_has_text_distractor   += 1
        if has_c: n_has_visual_distractor += 1
        if has_d: n_has_easy_distractor   += 1
        if opts.get("A") and has_b and has_c and has_d:
            n_all_four_options += 1

        # answer
        answer_counts[rec.get("answer", "?")] += 1

        # captions (OCR-detected)
        captions = rec.get("captions") or []
        if captions:
            n_has_caption += 1
            total_len = sum(len((c.get("text") or "")) for c in captions)
            caption_lengths.append(total_len)

        # discourse labels
        labels = rec.get("labels") or {}
        for k, v in labels.items():
            if v:
                label_keys[k] += 1

    months_sorted = [(m, month_counts.get(m, 0)) for m in VALID_MONTHS]
    n_valid_months = sum(1 for _, c in months_sorted if c > 0)

    avg_caption_len = (sum(caption_lengths) / len(caption_lengths)
                       if caption_lengths else 0)

    return {
        "source":          source_label,
        "total_items":     N,
        "n_valid_months":  n_valid_months,

        # context coverage
        "has_original_post_text":    n_has_original_text,
        "has_original_post_text_pct": round(n_has_original_text  / N * 100, 2),
        "has_quoted_post_text":      n_has_quoted_text,
        "has_quoted_post_text_pct":  round(n_has_quoted_text     / N * 100, 2),
        "has_parent_reply_text":     n_has_parent_text,
        "has_parent_reply_text_pct": round(n_has_parent_text     / N * 100, 2),
        "has_meme_text":             n_has_meme_text,
        "has_meme_text_pct":         round(n_has_meme_text       / N * 100, 2),
        "has_ancestor_chain":        n_has_ancestor_chain,
        "has_ancestor_chain_pct":    round(n_has_ancestor_chain  / N * 100, 2),

        "has_any_ctx_image":         n_has_ctx_image,
        "has_any_ctx_image_pct":     round(n_has_ctx_image       / N * 100, 2),
        "has_original_post_image":   n_has_original_image,
        "has_original_post_image_pct": round(n_has_original_image / N * 100, 2),
        "has_quoted_post_image":     n_has_quoted_image,
        "has_quoted_post_image_pct": round(n_has_quoted_image    / N * 100, 2),
        "has_parent_reply_image":    n_has_parent_image,
        "has_parent_reply_image_pct": round(n_has_parent_image   / N * 100, 2),
        "has_external_link":         n_has_external_link,
        "has_external_link_pct":     round(n_has_external_link   / N * 100, 2),

        # distractor coverage
        "has_text_distractor":    n_has_text_distractor,
        "has_text_distractor_pct": round(n_has_text_distractor   / N * 100, 2),
        "has_visual_distractor":  n_has_visual_distractor,
        "has_visual_distractor_pct": round(n_has_visual_distractor / N * 100, 2),
        "has_easy_distractor":    n_has_easy_distractor,
        "has_easy_distractor_pct": round(n_has_easy_distractor   / N * 100, 2),
        "all_four_options":       n_all_four_options,
        "all_four_options_pct":   round(n_all_four_options        / N * 100, 2),

        # OCR captions
        "has_ocr_caption":       n_has_caption,
        "has_ocr_caption_pct":   round(n_has_caption / N * 100, 2),
        "avg_caption_char_len":  round(avg_caption_len, 1),

        # answer distribution (sanity)
        "answer_distribution": dict(answer_counts),

        # discourse label counts (if any)
        "discourse_label_counts": dict(sorted(label_keys.items(),
                                              key=lambda x: -x[1])),

        "month_counts": dict(months_sorted),
    }


# ════════════════════════════════════════════════════════════════
#  콘솔 출력
# ════════════════════════════════════════════════════════════════

def print_benchmark_stats(s):
    N = s["total_items"]
    print("\n" + "=" * 62)
    print("  MEMECONV Benchmark Statistics")
    print("=" * 62)

    print(f"\n[SCALE]")
    print(f"  Total benchmark items        : {N:,}")
    print(f"  Months with items            : {s['n_valid_months']} / {len(VALID_MONTHS)}")

    print(f"\n[DISTRACTOR COVERAGE]")
    print(f"  Text distractor (B)          : {s['has_text_distractor']:,}  ({s['has_text_distractor_pct']}%)")
    print(f"  Visual distractor (C)        : {s['has_visual_distractor']:,}  ({s['has_visual_distractor_pct']}%)")
    print(f"  Easy distractor (D)          : {s['has_easy_distractor']:,}  ({s['has_easy_distractor_pct']}%)")
    print(f"  All 4 options complete       : {s['all_four_options']:,}  ({s['all_four_options_pct']}%)")

    print(f"\n[CONTEXT COVERAGE  (text)]")
    print(f"  Has original post text       : {s['has_original_post_text']:,}  ({s['has_original_post_text_pct']}%)")
    print(f"  Has quoted post text         : {s['has_quoted_post_text']:,}  ({s['has_quoted_post_text_pct']}%)")
    print(f"  Has parent reply text        : {s['has_parent_reply_text']:,}  ({s['has_parent_reply_text_pct']}%)")
    print(f"  Has meme reply text          : {s['has_meme_text']:,}  ({s['has_meme_text_pct']}%)")
    print(f"  Has ancestor chain           : {s['has_ancestor_chain']:,}  ({s['has_ancestor_chain_pct']}%)")
    print(f"  Has external link            : {s['has_external_link']:,}  ({s['has_external_link_pct']}%)")

    print(f"\n[CONTEXT COVERAGE  (images)]")
    print(f"  Has any context image        : {s['has_any_ctx_image']:,}  ({s['has_any_ctx_image_pct']}%)")
    print(f"  Original post image          : {s['has_original_post_image']:,}  ({s['has_original_post_image_pct']}%)")
    print(f"  Quoted post image            : {s['has_quoted_post_image']:,}  ({s['has_quoted_post_image_pct']}%)")
    print(f"  Parent reply image           : {s['has_parent_reply_image']:,}  ({s['has_parent_reply_image_pct']}%)")

    print(f"\n[OCR CAPTIONS]")
    print(f"  Items with OCR caption       : {s['has_ocr_caption']:,}  ({s['has_ocr_caption_pct']}%)")
    print(f"  Avg caption char length      : {s['avg_caption_char_len']}")

    ad = s["answer_distribution"]
    if ad:
        print(f"\n[ANSWER DISTRIBUTION  (sanity — should be all A)]")
        for k, v in sorted(ad.items()):
            print(f"  {k}  : {v:,}")

    dl = s["discourse_label_counts"]
    if dl:
        print(f"\n[DISCOURSE LABELS]")
        for k, v in dl.items():
            print(f"  {k:<30} : {v:,}  ({round(v/N*100,2)}%)")

    print(f"\n[MONTHLY DISTRIBUTION  ({s['n_valid_months']} months with data)]")
    mc     = s["month_counts"]
    months = [(m, mc[m]) for m in VALID_MONTHS]
    max_c  = max(c for _, c in months) if months else 1
    for m, c in months:
        bar = "#" * min(40, int(c / max(1, max_c) * 40))
        print(f"  {m}  {c:5,}  {bar}")

    print("\n" + "=" * 62)


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MEMECONV benchmark statistics extractor")
    parser.add_argument("--summary",   default=DEFAULT_SUMMARY,
                        help="benchmark_summary.jsonl 경로")
    parser.add_argument("--bench-dir", default=DEFAULT_BENCHDIR,
                        help="benchmark_data/ 폴더 경로 (summary 없을 때 fallback)")
    parser.add_argument("--out-json",  default=None,
                        help="결과를 JSON으로 저장")
    args = parser.parse_args()

    records      = []
    source_label = ""

    summary_path = Path(args.summary)
    bench_path   = Path(args.bench_dir)

    if summary_path.exists():
        print(f"Loading: {summary_path} ...")
        records      = load_summary_jsonl(summary_path)
        source_label = str(summary_path)
        print(f"  -> {len(records):,} items loaded")
    elif bench_path.exists():
        print(f"Loading bench dir: {bench_path} ...")
        records      = load_from_bench_dir(bench_path)
        source_label = str(bench_path)
        print(f"  -> {len(records):,} items loaded")
    else:
        print(f"[ERROR] '{summary_path}' 또는 '{bench_path}' 를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    s = compute_benchmark_stats(records, source_label)
    if not s:
        sys.exit(1)

    print_benchmark_stats(s)

    if args.out_json:
        out = Path(args.out_json)
        out.write_text(json.dumps({"benchmark_stats": s}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"JSON saved: {out}")


if __name__ == "__main__":
    main()
