#!/usr/bin/env python3
"""
MEMECONV Dataset Statistics Extractor
Usage:
  python3 extract_dataset_stats.py
  python3 extract_dataset_stats.py --jsonl ../03_filter_and_label/labeled_final/labeled_memes.jsonl
  python3 extract_dataset_stats.py --records ../01_collection/meme_dataset/records/
  python3 extract_dataset_stats.py --out-json stats_out.json
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

DEFAULT_JSONL   = "../03_filter_and_label/labeled_final/labeled_memes.jsonl"
DEFAULT_RECORDS = "../01_collection/meme_dataset/records/"
DEFAULT_STATS   = "../01_collection/meme_dataset/stats.json"

VALID_MONTHS = [
    "2023-09", "2023-10", "2023-11", "2023-12",
    "2024-01", "2024-02", "2024-03", "2024-04",
    "2024-05", "2024-06", "2024-07", "2024-08",
    "2024-09", "2024-10", "2024-11", "2024-12",
    "2025-01", "2025-02", "2025-03", "2025-04",
    "2025-05", "2025-06", "2025-07", "2025-08",
]


def load_jsonl(path):
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


def load_records_dir(path):
    records = []
    for fp in sorted(Path(path).glob("*.json")):
        try:
            records.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [warn] {fp.name}: {e}", file=sys.stderr)
    return records


def get_created_at(rec):
    mr = rec.get("meme_reply") or {}
    return mr.get("created_at") or rec.get("created_at")


def parse_month(ca):
    if not ca:
        return "unknown"
    try:
        ts = float(ca)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
    except (ValueError, TypeError):
        pass
    s = str(ca)
    if len(s) >= 7:
        return s[:7]
    return "unknown"


def compute_stats(records, source_label):
    N = len(records)
    if N == 0:
        print("  [warn] 레코드가 없습니다.", file=sys.stderr)
        return {}

    n_first_level = 0
    n_re_reply    = 0

    n_root_has_image   = 0
    n_parent_has_image = 0
    n_quoted_post      = 0
    n_quoted_has_image = 0

    n_comparison_available = 0
    meme_gt_nearby = 0
    n_nearby_valid = 0
    meme_gt_best   = 0
    n_best_valid   = 0

    n_sarcastic       = 0
    n_humorous        = 0
    n_offensive       = 0
    n_stance_labeled  = 0
    n_has_visual_desc = 0

    month_counts    = defaultdict(int)
    root_uris       = set()
    meme_uris       = set()
    parent_to_memes = defaultdict(list)
    root_to_memes   = defaultdict(list)

    for rec in records:
        uri = rec.get("uri", "")
        meme_uris.add(uri)

        ts    = rec.get("thread_structure") or {}
        label = ts.get("label") or ts.get("structure_label") or ""
        if label == "reply":
            n_first_level += 1
        elif label in ("re-reply", "re_reply"):
            n_re_reply += 1
        else:
            mr = rec.get("meme_reply") or {}
            if mr.get("is_re_reply"):
                n_re_reply += 1
            else:
                n_first_level += 1

        op = rec.get("original_post") or {}
        if op.get("has_image") or (op.get("images") and len(op["images"]) > 0):
            n_root_has_image += 1
        root_uri = op.get("uri") or (rec.get("meme_reply") or {}).get("root_uri")
        if root_uri:
            root_uris.add(root_uri)

        pr = rec.get("parent_reply") or {}
        if pr.get("has_image") or (pr.get("images") and len(pr.get("images", [])) > 0):
            n_parent_has_image += 1

        qp = rec.get("quoted_post") or rec.get("embed_post") or {}
        if qp:
            n_quoted_post += 1
            if qp.get("has_image") or (qp.get("images") and len(qp.get("images", [])) > 0):
                n_quoted_has_image += 1

        comp = rec.get("comparison_reply")
        if comp:
            n_comparison_available += 1
            meme_likes = (rec.get("meme_reply") or {}).get("like_count") or 0
            comp_likes = comp.get("like_count") or 0
            if meme_likes is not None and comp_likes is not None:
                n_nearby_valid += 1
                if meme_likes > comp_likes:
                    meme_gt_nearby += 1

        bb = rec.get("best_reply_before_meme")
        if bb:
            meme_likes = (rec.get("meme_reply") or {}).get("like_count") or 0
            bb_likes   = bb.get("like_count") or 0
            if meme_likes is not None and bb_likes is not None:
                n_best_valid += 1
                if meme_likes > bb_likes:
                    meme_gt_best += 1

        ann    = (rec.get("meme_reply") or {}).get("annotation") or rec.get("annotation") or {}
        stance = ann.get("stance_labels") or rec.get("stance_labels") or {}
        if stance:
            n_stance_labeled += 1
            if stance.get("sarcastic"):  n_sarcastic += 1
            if stance.get("humorous"):   n_humorous  += 1
            if stance.get("offensive"):  n_offensive += 1

        vd = (ann.get("visual_description") or
              rec.get("visual_description") or
              (rec.get("meme_reply") or {}).get("visual_description"))
        if vd and str(vd).strip():
            n_has_visual_desc += 1

        m = parse_month(get_created_at(rec))
        if m in VALID_MONTHS:
            month_counts[m] += 1

        mr2        = rec.get("meme_reply") or {}
        parent_uri = mr2.get("parent_uri") or ""
        if parent_uri:
            parent_to_memes[parent_uri].append(uri)
        if root_uri:
            root_to_memes[root_uri].append(uri)

    n_direct_relay = sum(
        1 for rec in records
        if ((rec.get("meme_reply") or {}).get("parent_uri") or "") in meme_uris
    )
    n_thread_cluster = sum(
        1 for root, uris_in_root in root_to_memes.items()
        if len(uris_in_root) >= 2
        for _ in uris_in_root
    )
    n_same_parent_cluster = sum(
        1 for parent, uris_in_parent in parent_to_memes.items()
        if len(uris_in_parent) >= 2
        for _ in uris_in_parent
    )

    months_sorted = [(m, month_counts.get(m, 0)) for m in VALID_MONTHS]

    return {
        "source":                source_label,
        "total_records":         N,
        "unique_root_threads":   len(root_uris),

        "first_level_replies":   n_first_level,
        "re_replies":            n_re_reply,
        "first_level_pct":       round(n_first_level / N * 100, 2),
        "re_reply_pct":          round(n_re_reply    / N * 100, 2),

        "root_post_has_image":        n_root_has_image,
        "root_post_has_image_pct":    round(n_root_has_image   / N * 100, 2),
        "parent_reply_has_image":     n_parent_has_image,
        "parent_reply_has_image_pct": round(n_parent_has_image / N * 100, 2),
        "quoted_post_available":      n_quoted_post,
        "quoted_post_pct":            round(n_quoted_post      / N * 100, 2),
        "quoted_post_has_image":      n_quoted_has_image,
        "quoted_post_has_image_pct":  round(n_quoted_has_image / N * 100, 2),

        "comparison_reply_available": n_comparison_available,
        "comparison_reply_pct":       round(n_comparison_available / N * 100, 2),
        "meme_gt_nearby_N":           n_nearby_valid,
        "meme_gt_nearby_count":       meme_gt_nearby,
        "meme_gt_nearby_pct":         round(meme_gt_nearby / max(1, n_nearby_valid) * 100, 2),
        "meme_gt_best_N":             n_best_valid,
        "meme_gt_best_count":         meme_gt_best,
        "meme_gt_best_pct":           round(meme_gt_best   / max(1, n_best_valid)   * 100, 2),

        "direct_meme_to_meme_relay":      n_direct_relay,
        "direct_meme_relay_pct":          round(n_direct_relay       / N * 100, 2),
        "thread_level_meme_cluster":      n_thread_cluster,
        "thread_cluster_pct":             round(n_thread_cluster      / N * 100, 2),
        "same_parent_meme_cluster":       n_same_parent_cluster,
        "same_parent_cluster_pct":        round(n_same_parent_cluster / N * 100, 2),
        "image_to_image_reply_candidate": n_parent_has_image,
        "image_to_image_pct":             round(n_parent_has_image    / N * 100, 2),

        "stance_labeled":         n_stance_labeled,
        "sarcastic_count":        n_sarcastic,
        "sarcastic_pct":          round(n_sarcastic / max(1, n_stance_labeled) * 100, 2),
        "humorous_count":         n_humorous,
        "humorous_pct":           round(n_humorous  / max(1, n_stance_labeled) * 100, 2),
        "offensive_count":        n_offensive,
        "offensive_pct":          round(n_offensive / max(1, n_stance_labeled) * 100, 2),
        "has_visual_description": n_has_visual_desc,
        "visual_description_pct": round(n_has_visual_desc / N * 100, 2),

        "month_counts":     dict(months_sorted),
        "n_months_covered": len(VALID_MONTHS),
    }


def print_stats(s, collection_stats=None):
    N = s["total_records"]
    print("\n" + "=" * 62)
    print("  MEMECONV Dataset Statistics")
    print("=" * 62)

    print("\n[SCALE]")
    print(f"  Total meme reply records     : {N:,}")
    print(f"  Unique root threads          : {s['unique_root_threads']:,}")

    print("\n[THREAD STRUCTURE]")
    print(f"  First-level replies          : {s['first_level_replies']:,}  ({s['first_level_pct']}%)")
    print(f"  Re-replies (nested)          : {s['re_replies']:,}  ({s['re_reply_pct']}%)")

    print("\n[MULTIMODAL CONTEXT]")
    print(f"  Root post has image          : {s['root_post_has_image']:,}  ({s['root_post_has_image_pct']}%)")
    print(f"  Parent reply has image       : {s['parent_reply_has_image']:,}  ({s['parent_reply_has_image_pct']}%)")
    print(f"  Quoted/embedded post present : {s['quoted_post_available']:,}  ({s['quoted_post_pct']}%)")
    print(f"  Quoted post has image        : {s['quoted_post_has_image']:,}  ({s['quoted_post_has_image_pct']}%)")

    print("\n[ENGAGEMENT COMPARISON]")
    print(f"  Comparison reply available   : {s['comparison_reply_available']:,}  ({s['comparison_reply_pct']}%)")
    print(f"  Meme > nearby text reply     : {s['meme_gt_nearby_count']:,} / {s['meme_gt_nearby_N']:,}  ({s['meme_gt_nearby_pct']}%)")
    print(f"  Meme > best reply            : {s['meme_gt_best_count']:,} / {s['meme_gt_best_N']:,}  ({s['meme_gt_best_pct']}%)")

    print("\n[MEME RELAY & CLUSTER]")
    print(f"  Direct meme-to-meme relay    : {s['direct_meme_to_meme_relay']:,}  ({s['direct_meme_relay_pct']}%)")
    print(f"  Thread-level meme cluster    : {s['thread_level_meme_cluster']:,}  ({s['thread_cluster_pct']}%)")
    print(f"  Same-parent meme cluster     : {s['same_parent_meme_cluster']:,}  ({s['same_parent_cluster_pct']}%)")
    print(f"  Image-to-image reply cand.   : {s['image_to_image_reply_candidate']:,}  ({s['image_to_image_pct']}%)")

    if s["stance_labeled"] > 0:
        print(f"\n[ANNOTATION  n={s['stance_labeled']:,}]")
        print(f"  Sarcastic                    : {s['sarcastic_count']:,}  ({s['sarcastic_pct']}%)")
        print(f"  Humorous                     : {s['humorous_count']:,}  ({s['humorous_pct']}%)")
        print(f"  Offensive                    : {s['offensive_count']:,}  ({s['offensive_pct']}%)")
        print(f"  Has visual description       : {s['has_visual_description']:,}  ({s['visual_description_pct']}%)")

    print(f"\n[TEMPORAL COVERAGE  {s['n_months_covered']} months: Sept 2023 - Aug 2025]")
    mc     = s["month_counts"]
    months = [(m, mc[m]) for m in VALID_MONTHS]
    max_c  = max(c for _, c in months) if months else 1
    for m, c in months:
        bar = "#" * min(40, int(c / max(1, max_c) * 40))
        print(f"  {m}  {c:5,}  {bar}")

    print("\n" + "=" * 62)


def main():
    parser = argparse.ArgumentParser(description="MEMECONV dataset statistics extractor")
    parser.add_argument("--jsonl",    default=DEFAULT_JSONL)
    parser.add_argument("--records",  default=DEFAULT_RECORDS)
    parser.add_argument("--stats",    default=DEFAULT_STATS)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    records      = []
    source_label = ""
    jsonl_path   = Path(args.jsonl)
    records_path = Path(args.records)

    if jsonl_path.exists():
        print(f"Loading JSONL: {jsonl_path} ...")
        records      = load_jsonl(jsonl_path)
        source_label = str(jsonl_path)
        print(f"  -> {len(records):,} records loaded")
    elif records_path.exists():
        print(f"Loading records dir: {records_path} ...")
        records      = load_records_dir(records_path)
        source_label = str(records_path)
        print(f"  -> {len(records):,} records loaded")
    else:
        print(f"[ERROR] '{jsonl_path}' 또는 '{records_path}' 를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    collection_stats = None
    stats_path = Path(args.stats)
    if stats_path.exists():
        try:
            collection_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            print(f"Collection stats loaded: {stats_path}")
        except Exception as e:
            print(f"  [warn] stats.json: {e}", file=sys.stderr)

    s = compute_stats(records, source_label)
    if not s:
        sys.exit(1)

    print_stats(s, collection_stats)

    if args.out_json:
        out     = Path(args.out_json)
        payload = {"computed_stats": s}
        if collection_stats:
            payload["collection_stats_summary"] = {
                k: v for k, v in collection_stats.items() if k != "per_day"
            }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON saved: {out}")


if __name__ == "__main__":
    main()
