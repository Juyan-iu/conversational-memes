#!/usr/bin/env python3
"""
Export a UID/URI manifest from collected meme reply records.

Run from 01_collection:

  python export_uid_manifest.py \
    --records-dir meme_dataset_24_06/records \
    --records-dir meme_dataset_25_02/records \
    --out ../download/data/meme_reply_uid_manifest.jsonl

The downloader needs full at:// URIs because project UIDs keep only the last
14 characters of the DID and are not reversible by themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RECORD_DIRS = (
    "meme_dataset_24_06/records",
    "meme_dataset_25_02/records",
    "meme_dataset/records",
)


def uri_to_uid(uri: str) -> str:
    try:
        parts = uri.replace("at://", "").split("/")
        did_suffix = parts[0].split(":")[-1][-14:]
        rkey = parts[-1]
        return f"bsky_{did_suffix}_{rkey}"
    except Exception:
        return "bsky_" + hashlib.sha256(uri.encode("utf-8")).hexdigest()[:22]


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def clean_uri(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("at://"):
        return value
    return None


def load_json_file(path: Path) -> Iterable[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
        return []

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def load_jsonl_file(path: Path) -> Iterable[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                print(f"[WARN] {path}:{line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def iter_input_records(paths: list[Path]) -> Iterable[tuple[dict[str, Any], str]]:
    for path in paths:
        if path.is_dir():
            files = sorted(path.glob("*.json"))
            print(f"[SCAN] {path}: {len(files):,} JSON files", flush=True)
            for fp in files:
                for record in load_json_file(fp):
                    yield record, str(fp)
        elif path.suffix.lower() == ".jsonl":
            print(f"[SCAN] {path}: JSONL input", flush=True)
            for record in load_jsonl_file(path):
                yield record, str(path)
        elif path.suffix.lower() == ".json":
            print(f"[SCAN] {path}: JSON input", flush=True)
            for record in load_json_file(path):
                yield record, str(path)
        else:
            print(f"[WARN] Unsupported input path: {path}", file=sys.stderr)


def infer_thread_label(depth: int | None, parent_uri: str | None, root_uri: str | None) -> str | None:
    if depth == 1:
        return "reply"
    if depth and depth >= 2:
        return "re-reply"
    if parent_uri and root_uri and parent_uri != root_uri:
        return "re-reply"
    if parent_uri or root_uri:
        return "reply"
    return None


def normalize_record(
    record: dict[str, Any],
    source: str,
    include_source: bool = False,
) -> dict[str, Any] | None:
    meme = as_dict(record.get("meme_reply"))
    original = as_dict(record.get("original_post"))
    parent = as_dict(record.get("parent_reply"))
    quoted = as_dict(record.get("quoted_post"))
    comparison = as_dict(record.get("comparison_reply"))
    best_reply = as_dict(record.get("best_reply_before_meme"))
    closest_text = as_dict(record.get("closest_text_reply"))
    closest_sibling = as_dict(record.get("closest_sibling_text_reply"))

    uri = (
        clean_uri(record.get("uri"))
        or clean_uri(meme.get("uri"))
        or clean_uri(get_path(record, "post", "uri"))
    )
    if not uri:
        return None

    uid = str(record.get("uid") or meme.get("uid") or uri_to_uid(uri))
    root_uri = clean_uri(original.get("uri")) or clean_uri(meme.get("root_uri"))
    reply_parent_uri = clean_uri(meme.get("parent_uri")) or clean_uri(parent.get("uri"))

    parent_reply_uri: str | None = None
    if reply_parent_uri and root_uri and reply_parent_uri != root_uri:
        parent_reply_uri = reply_parent_uri
    elif clean_uri(parent.get("uri")):
        parent_reply_uri = clean_uri(parent.get("uri"))

    quoted_post_uri = clean_uri(quoted.get("uri"))
    comparison_reply_uri = clean_uri(comparison.get("uri"))
    best_reply_before_meme_uri = clean_uri(best_reply.get("uri"))
    closest_text_reply_uri = clean_uri(closest_text.get("uri"))
    closest_sibling_text_reply_uri = clean_uri(closest_sibling.get("uri"))

    depth_raw = get_path(record, "thread_structure", "depth")
    try:
        thread_depth = int(depth_raw) if depth_raw is not None else None
    except Exception:
        thread_depth = None

    thread_label = (
        get_path(record, "thread_structure", "label")
        or infer_thread_label(thread_depth, reply_parent_uri, root_uri)
    )

    row = {
        "uid": uid,
        "uri": uri,
        "meme_reply_uri": uri,
        "root_post_uri": root_uri,
        "reply_parent_uri": reply_parent_uri,
        "parent_reply_uri": parent_reply_uri,
        "quoted_post_uri": quoted_post_uri,
        "best_reply_before_meme_uri": best_reply_before_meme_uri,
        "closest_text_reply_uri": closest_text_reply_uri,
        "closest_sibling_text_reply_uri": closest_sibling_text_reply_uri,
        "comparison_reply_uri": comparison_reply_uri,
        "thread_depth": thread_depth,
        "thread_label": thread_label,
    }
    if include_source:
        row["source_record"] = source
    return row


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_uid_txt(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row["uid"]) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export UID/URI manifest from meme reply records."
    )
    parser.add_argument(
        "--records-dir",
        action="append",
        default=[],
        help="Directory containing <uid>.json records. Can be passed multiple times.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Additional JSON, JSONL, or directory input. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out",
        default="uid_manifest.jsonl",
        help="Output path. Default: uid_manifest.jsonl",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "uid-txt"),
        default="jsonl",
        help="Output JSONL manifest or plain UID list. Default: jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Write only the first N deduplicated records for a sample manifest.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Keep duplicate UIDs instead of writing the first occurrence only.",
    )
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include local source file paths in the output manifest.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after every N scanned records. Default: 1000.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = [Path(p) for p in args.records_dir + args.input]
    if not input_paths:
        input_paths = [Path(p) for p in DEFAULT_RECORD_DIRS]

    existing_paths = [p for p in input_paths if p.exists()]
    missing_paths = [p for p in input_paths if not p.exists()]
    print(
        json.dumps(
            {
                "input_paths": [str(p) for p in input_paths],
                "existing_paths": [str(p) for p in existing_paths],
                "missing_paths": [str(p) for p in missing_paths],
            },
            indent=2,
        ),
        flush=True,
    )
    for path in missing_paths:
        print(f"[WARN] Missing input path: {path}", file=sys.stderr)

    if not existing_paths:
        print("[ERROR] No input records found.", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0
    skipped = 0

    for record, source in iter_input_records(existing_paths):
        scanned += 1
        if args.progress_every and scanned % args.progress_every == 0:
            print(
                f"[PROGRESS] scanned={scanned:,} written={len(rows):,} skipped_without_uri={skipped:,}",
                flush=True,
            )
        row = normalize_record(record, source, include_source=args.include_source)
        if row is None:
            skipped += 1
            continue
        if not args.allow_duplicates and row["uid"] in seen:
            continue
        seen.add(row["uid"])
        rows.append(row)
        if args.limit is not None and len(rows) >= args.limit:
            break

    out_path = Path(args.out)
    if args.format == "jsonl":
        written = write_jsonl(rows, out_path)
    else:
        written = write_uid_txt(rows, out_path)

    if scanned == 0:
        print(
            "[WARN] No input JSON records were scanned. Pass --records-dir "
            "/path/to/records if your collected records live elsewhere.",
            file=sys.stderr,
            flush=True,
        )
    elif written == 0:
        print(
            "[WARN] Records were scanned, but none had a usable at:// URI.",
            file=sys.stderr,
            flush=True,
        )

    print(
        json.dumps(
            {
                "scanned": scanned,
                "written": written,
                "skipped_without_uri": skipped,
                "output": str(out_path),
                "format": args.format,
            },
            indent=2,
        )
    )
    return 0 if written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
