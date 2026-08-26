#!/usr/bin/env python3
"""
Inspect a context-augmented dataset file against the original JSONL.

This is a lightweight diagnostic tool for checking whether
labeled_memes_with_context.jsonl is complete, partial, prefix-aligned with the
original file, and which context fields were actually added.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    bad += 1
            except Exception:
                bad += 1
                print(f"[WARN] bad JSON at {path}:{line_no}")
    return rows, bad


def uid(record: dict[str, Any]) -> str:
    return str(record.get("uid") or record.get("uri") or "")


def uri(record: dict[str, Any]) -> str:
    return str(record.get("uri") or (record.get("meme_reply") or {}).get("uri") or "")


def has_dict(record: dict[str, Any], key: str) -> bool:
    return isinstance(record.get(key), dict) and bool(record.get(key))


def has_list(record: dict[str, Any], key: str) -> bool:
    return isinstance(record.get(key), list) and bool(record.get(key))


def nested_has(record: dict[str, Any], *keys: str) -> bool:
    cur: Any = record
    for key in keys:
        if not isinstance(cur, dict):
            return False
        cur = cur.get(key)
    return bool(cur)


def img_count(post: Any) -> int:
    if not isinstance(post, dict):
        return 0
    images = post.get("images")
    return len(images) if isinstance(images, list) else 0


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_mb": round(stat.st_size / 1024 / 1024, 2),
        "mtime": stat.st_mtime,
    }


def coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    cov = Counter()
    for r in rows:
        ancestor = r.get("ancestor_chain")
        quoted = r.get("quoted_post")
        parent = r.get("parent_reply")
        original = r.get("original_post")
        meme = r.get("meme_reply")
        comparison = r.get("comparison_reply")

        cov["records"] += 1
        cov["has_parent_reply"] += int(has_dict(r, "parent_reply"))
        cov["has_ancestor_chain"] += int(has_list(r, "ancestor_chain"))
        cov["ancestor_chain_nodes"] += len(ancestor) if isinstance(ancestor, list) else 0
        cov["ancestor_chain_has_images"] += int(
            isinstance(ancestor, list) and any(img_count(node) for node in ancestor if isinstance(node, dict))
        )
        cov["ancestor_chain_has_quoted_post"] += int(
            isinstance(ancestor, list)
            and any(isinstance(node, dict) and isinstance(node.get("quoted_post"), dict) for node in ancestor)
        )
        cov["has_quoted_post"] += int(has_dict(r, "quoted_post"))
        cov["quoted_post_has_text"] += int(nested_has(r, "quoted_post", "text"))
        cov["quoted_post_has_images"] += int(img_count(quoted) > 0)
        cov["original_post_has_images"] += int(img_count(original) > 0)
        cov["parent_reply_has_images"] += int(img_count(parent) > 0)
        cov["meme_reply_has_images"] += int(img_count(meme) > 0)
        cov["has_best_reply_before_meme"] += int(has_dict(r, "best_reply_before_meme"))
        cov["has_closest_text_reply"] += int(has_dict(r, "closest_text_reply"))
        cov["has_closest_sibling_text_reply"] += int(has_dict(r, "closest_sibling_text_reply"))
        cov["has_comparison_reply"] += int(isinstance(comparison, dict) and bool(comparison))
        cov["comparison_selected_by"] += int(nested_has(r, "comparison_reply", "selected_by"))
        cov["has_visual_description_top_level"] += int(bool(r.get("visual_description")))
        cov["has_visual_description_meme_reply"] += int(nested_has(r, "meme_reply", "visual_description"))
        cov["has_visual_description_legacy"] += int(
            nested_has(r, "discourse_labels", "meme_reply", "visual", "visual_description")
        )
    return dict(cov)


def print_coverage(title: str, rows: list[dict[str, Any]]) -> None:
    cov = coverage(rows)
    total = cov.get("records", 0)
    print(f"\n[{title}] field coverage")
    for key, value in cov.items():
        if key == "records" or key == "ancestor_chain_nodes":
            print(f"  {key}: {value:,}")
        else:
            pct = value / total * 100 if total else 0
            print(f"  {key}: {value:,} ({pct:.2f}%)")


def record_brief(record: dict[str, Any]) -> str:
    return (
        f"uid={uid(record)} "
        f"uri={uri(record)} "
        f"process_date={record.get('process_date')} "
        f"depth={(record.get('thread_structure') or {}).get('depth')} "
        f"ancestor_len={len(record.get('ancestor_chain') or []) if isinstance(record.get('ancestor_chain'), list) else 0} "
        f"quoted={has_dict(record, 'quoted_post')} "
        f"parent={has_dict(record, 'parent_reply')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect original vs context-augmented meme JSONL files.")
    parser.add_argument("--base", required=True, help="Original labeled_memes.jsonl")
    parser.add_argument("--context", required=True, help="Context-augmented JSONL to inspect")
    parser.add_argument("--sample", type=int, default=5, help="Number of boundary records to print")
    args = parser.parse_args()

    base_path = Path(args.base)
    context_path = Path(args.context)

    print("[FILES]")
    for p in [base_path, context_path]:
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
        info = file_info(p)
        print(f"  {info['path']} ({info['size_mb']} MB)")

    base, base_bad = load_jsonl(base_path)
    context, context_bad = load_jsonl(context_path)

    base_uids = [uid(r) for r in base]
    context_uids = [uid(r) for r in context]
    base_set = set(base_uids)
    context_set = set(context_uids)

    print("\n[COUNTS]")
    print(f"  base lines/records: {len(base):,} bad_json={base_bad:,} unique_uid={len(base_set):,}")
    print(f"  context lines/records: {len(context):,} bad_json={context_bad:,} unique_uid={len(context_set):,}")
    print(f"  context/base ratio: {len(context) / len(base) * 100 if base else 0:.2f}%")
    print(f"  context uids missing from base: {len(context_set - base_set):,}")
    print(f"  base uids missing from context: {len(base_set - context_set):,}")

    is_prefix = context_uids == base_uids[: len(context_uids)]
    print("\n[ORDER]")
    print(f"  context is exact prefix of base by uid order: {is_prefix}")
    if not is_prefix:
        first_mismatch = None
        for i, (a, b) in enumerate(zip(base_uids, context_uids), start=1):
            if a != b:
                first_mismatch = i
                break
        print(f"  first uid-order mismatch line: {first_mismatch}")
    if context:
        last_uid = context_uids[-1]
        try:
            base_pos = base_uids.index(last_uid) + 1
        except ValueError:
            base_pos = None
        print(f"  last context uid appears at base line: {base_pos}")
        if base_pos and base_pos < len(base):
            print(f"  next base uid after last context record: {base_uids[base_pos]}")

    print_coverage("base", base)
    print_coverage("context", context)

    print("\n[TOP-LEVEL KEY DIFF ON SHARED SAMPLE]")
    shared = [r for r in context if uid(r) in base_set][: min(1000, len(context))]
    base_by_uid = {uid(r): r for r in base}
    added_keys = Counter()
    removed_keys = Counter()
    changed_keys = Counter()
    for r in shared:
        b = base_by_uid.get(uid(r), {})
        b_keys = set(b.keys())
        r_keys = set(r.keys())
        added_keys.update(r_keys - b_keys)
        removed_keys.update(b_keys - r_keys)
        for key in b_keys & r_keys:
            if b.get(key) != r.get(key):
                changed_keys[key] += 1
    print(f"  shared sample size: {len(shared):,}")
    print(f"  added keys: {dict(added_keys.most_common(20))}")
    print(f"  removed keys: {dict(removed_keys.most_common(20))}")
    print(f"  changed keys: {dict(changed_keys.most_common(20))}")

    n = max(0, args.sample)
    if n:
        print(f"\n[FIRST {n} CONTEXT RECORDS]")
        for r in context[:n]:
            print("  " + record_brief(r))

        print(f"\n[LAST {n} CONTEXT RECORDS]")
        for r in context[-n:]:
            print("  " + record_brief(r))

        if context:
            last_uid = context_uids[-1]
            try:
                base_idx = base_uids.index(last_uid)
            except ValueError:
                base_idx = None
            if base_idx is not None:
                print(f"\n[NEXT {n} BASE RECORDS AFTER CONTEXT CUTOFF]")
                for r in base[base_idx + 1 : base_idx + 1 + n]:
                    print("  " + record_brief(r))


if __name__ == "__main__":
    main()
