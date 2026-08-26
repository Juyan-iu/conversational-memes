#!/usr/bin/env python3
"""
Export a public labeled-release file from the internal labeled JSONL.

The public release must not include Bluesky-owned text, image metadata, author
metadata, or local image paths. It keeps only:

- UID and AT-URI pointers needed for public hydration
- labels and scores produced by this project
- optional benchmark/split metadata keyed by UID

Sample:

  python export_public_labeled_release.py \
    --input labeled_final/labeled_memes.jsonl \
    --out ../download/data/labeled_release.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


URI_POINTER_FIELDS = (
    "uid",
    "uri",
    "meme_reply_uri",
    "root_post_uri",
    "reply_parent_uri",
    "parent_reply_uri",
    "quoted_post_uri",
    "best_reply_before_meme_uri",
    "closest_text_reply_uri",
    "closest_sibling_text_reply_uri",
    "comparison_reply_uri",
    "thread_depth",
    "thread_label",
)

DEFAULT_LABEL_FIELDS = (
    "meme_prob",
    "threshold",
    "is_meme",
    "meme_validation",
    "stance_labels",
    "visual_description",
    "labeled_at",
    "label_version",
    "annotation_version",
    "split",
    "benchmark_split",
    "benchmark_id",
    "selected_by",
)

DEFAULT_EXCLUDE_FIELDS = {
    "ancestor_chain",
    "best_reply_before_meme",
    "closest_sibling_text_reply",
    "closest_text_reply",
    "comparison_reply",
    "discourse_labels",
    "downloaded_images",
    "images",
    "meme_reply",
    "original_post",
    "parent_reply",
    "quoted_post",
}
SENSITIVE_NESTED_KEYS = {
    "alt",
    "author",
    "created_at",
    "did",
    "display_name",
    "handle",
    "image",
    "images",
    "indexed_at",
    "local_path",
    "source_local_path",
    "source_url",
    "text",
    "thumb_url",
    "url",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_uri(value: Any) -> str | None:
    return value if isinstance(value, str) and value.startswith("at://") else None


def post_uri(value: Any) -> str | None:
    if isinstance(value, dict):
        return clean_uri(value.get("uri"))
    return None


def post_field(value: Any, field: str) -> Any:
    return value.get(field) if isinstance(value, dict) else None


def get_nested(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = record
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def infer_thread_depth(record: dict[str, Any], root_uri: str | None, reply_parent_uri: str | None) -> int | None:
    depth = record.get("thread_depth")
    if isinstance(depth, int):
        return depth
    structure = record.get("thread_structure")
    if isinstance(structure, dict):
        value = structure.get("depth")
        if isinstance(value, int):
            return value
    if reply_parent_uri and root_uri:
        return 1 if reply_parent_uri == root_uri else 2
    return None


def infer_thread_label(record: dict[str, Any], depth: int | None) -> str | None:
    label = record.get("thread_label")
    if isinstance(label, str) and label:
        return label
    structure = record.get("thread_structure")
    if isinstance(structure, dict) and isinstance(structure.get("label"), str):
        return structure["label"]
    if depth is None:
        return None
    return "re-reply" if depth >= 2 else "reply"


def extract_uri_pointers(record: dict[str, Any]) -> dict[str, Any]:
    meme_reply = record.get("meme_reply")
    original_post = record.get("original_post")
    parent_reply = record.get("parent_reply")

    meme_uri = (
        clean_uri(record.get("meme_reply_uri"))
        or clean_uri(record.get("uri"))
        or post_uri(meme_reply)
    )
    root_uri = (
        clean_uri(record.get("root_post_uri"))
        or post_uri(original_post)
        or clean_uri(post_field(meme_reply, "root_uri"))
    )
    reply_parent_uri = (
        clean_uri(record.get("reply_parent_uri"))
        or clean_uri(post_field(meme_reply, "parent_uri"))
    )
    parent_reply_uri = (
        clean_uri(record.get("parent_reply_uri"))
        or post_uri(parent_reply)
    )
    if parent_reply_uri and root_uri and parent_reply_uri == root_uri:
        parent_reply_uri = None

    depth = infer_thread_depth(record, root_uri, reply_parent_uri)
    label = infer_thread_label(record, depth)

    out = {
        "uid": record.get("uid"),
        "uri": meme_uri,
        "meme_reply_uri": meme_uri,
        "root_post_uri": root_uri,
        "reply_parent_uri": reply_parent_uri,
        "parent_reply_uri": parent_reply_uri,
        "quoted_post_uri": clean_uri(record.get("quoted_post_uri")) or post_uri(record.get("quoted_post")),
        "best_reply_before_meme_uri": clean_uri(record.get("best_reply_before_meme_uri")) or post_uri(record.get("best_reply_before_meme")),
        "closest_text_reply_uri": clean_uri(record.get("closest_text_reply_uri")) or post_uri(record.get("closest_text_reply")),
        "closest_sibling_text_reply_uri": clean_uri(record.get("closest_sibling_text_reply_uri")) or post_uri(record.get("closest_sibling_text_reply")),
        "comparison_reply_uri": clean_uri(record.get("comparison_reply_uri")) or post_uri(record.get("comparison_reply")),
        "comparison_reply_selected_by": record.get("comparison_reply_selected_by") or post_field(record.get("comparison_reply"), "selected_by"),
        "thread_depth": depth,
        "thread_label": label,
    }
    return {key: value for key, value in out.items() if value is not None}


def extract_visual_description(record: dict[str, Any]) -> str | None:
    candidates = [
        record.get("visual_description"),
        get_nested(record, ("visual", "visual_description")),
        get_nested(record, ("discourse_labels", "meme_reply", "visual", "visual_description")),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_stance_labels(record: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        record.get("stance_labels"),
        record.get("stance"),
        get_nested(record, ("discourse_labels", "meme_reply", "stance")),
    ]
    for value in candidates:
        if isinstance(value, dict):
            return {
                "sarcastic": bool(value.get("sarcastic")),
                "humorous": bool(value.get("humorous")),
                "offensive": bool(value.get("offensive")),
            }
    return None


def safe_extra_fields(record: dict[str, Any], include_fields: list[str], exclude_fields: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in DEFAULT_LABEL_FIELDS:
        if field in exclude_fields:
            continue
        if field in record and record[field] is not None:
            if field == "meme_validation":
                out[field] = sanitize_meme_validation(record[field])
            else:
                out[field] = sanitize_public_value(record[field])

    stance = extract_stance_labels(record)
    if stance is not None:
        out["stance_labels"] = stance

    visual = extract_visual_description(record)
    if visual is not None:
        out["visual_description"] = visual

    for field in include_fields:
        if field in exclude_fields:
            continue
        if field in record and record[field] is not None:
            if field == "meme_validation":
                out[field] = sanitize_meme_validation(record[field])
            else:
                out[field] = sanitize_public_value(record[field])
    return out


def sanitize_meme_validation(value: Any) -> Any:
    sanitized = sanitize_public_value(value)
    if isinstance(sanitized, dict):
        sanitized.pop("reason", None)
    return sanitized


def sanitize_public_value(value: Any) -> Any:
    """Remove nested Bluesky/content fields from project-created metadata."""
    if isinstance(value, dict):
        return {
            key: sanitize_public_value(inner)
            for key, inner in value.items()
            if key not in SENSITIVE_NESTED_KEYS
        }
    if isinstance(value, list):
        return [sanitize_public_value(item) for item in value]
    return value


def release_row(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    exclude_fields = set(args.exclude_field or [])
    if not args.include_discourse_labels:
        exclude_fields.add("discourse_labels")
    exclude_fields.update(DEFAULT_EXCLUDE_FIELDS)
    for field in args.include_field or []:
        exclude_fields.discard(field)

    out = extract_uri_pointers(record)
    out.update(safe_extra_fields(record, args.include_field or [], exclude_fields))
    out["release_metadata"] = {
        "exported_at": utc_now(),
        "source": "internal_labeled_jsonl",
        "contains_bluesky_content": False,
    }

    if not out.get("uid"):
        return None
    if args.require_uri and not out.get("meme_reply_uri"):
        return None
    return out


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export public labeled release JSONL from internal labeled records.")
    parser.add_argument("--input", default="labeled_final/labeled_memes.jsonl")
    parser.add_argument("--out", default="../download/data/labeled_release.jsonl")
    parser.add_argument("--report", default=None, help="Optional report path. Defaults to <out>.report.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-uri", action="store_true", help="Skip rows without meme_reply_uri/uri.")
    parser.add_argument("--include-field", action="append", default=[], help="Carry this additional top-level field. Can repeat.")
    parser.add_argument("--exclude-field", action="append", default=[], help="Exclude this field from export. Can repeat.")
    parser.add_argument("--include-discourse-labels", action="store_true", help="Internal analysis only; not paper-facing by default.")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    field_non_null = Counter()

    with out_path.open("w", encoding="utf-8") as out_handle:
        for line_no, record in iter_jsonl(input_path):
            if args.limit is not None and stats["scanned"] >= args.limit:
                break
            stats["scanned"] += 1
            row = release_row(record, args)
            if row is None:
                stats["skipped"] += 1
                continue
            out_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stats["written"] += 1
            for field in row:
                if row.get(field) is not None:
                    field_non_null[field] += 1
            if args.progress_every and stats["scanned"] % args.progress_every == 0:
                print(f"[PROGRESS] scanned={stats['scanned']} written={stats['written']} skipped={stats['skipped']}", flush=True)

    report = {
        "input": str(input_path),
        "output": str(out_path),
        "stats": dict(stats),
        "field_non_null": dict(sorted(field_non_null.items())),
        "excluded_by_default": sorted(DEFAULT_EXCLUDE_FIELDS),
        "sensitive_nested_keys_removed": sorted(SENSITIVE_NESTED_KEYS),
    }
    report_path = Path(args.report) if args.report else out_path.with_suffix(out_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if stats["written"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
