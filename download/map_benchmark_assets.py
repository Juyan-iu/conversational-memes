#!/usr/bin/env python3
"""
Map labeled records to benchmark membership and provided distractor assets by UID.

The script is intentionally format-tolerant: JSONL, JSON, CSV, and TSV files
are accepted for benchmark maps and distractor maps as long as they contain a
UID column.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


POST_IMAGE_FIELDS = (
    "meme_reply",
    "original_post",
    "parent_reply",
    "quoted_post",
    "best_reply_before_meme",
    "closest_text_reply",
    "closest_sibling_text_reply",
    "comparison_reply",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RecordItem:
    record: dict[str, Any]
    source_path: Path | None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def iter_records(input_path: Path) -> tuple[Iterable[RecordItem], Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".jsonl":
            return (RecordItem(row, None) for row in iter_jsonl(input_path)), input_path.parent
        return [RecordItem(read_json(input_path), input_path)], input_path.parent

    if (input_path / "records").exists():
        records_dir = input_path / "records"
        source_root = input_path
    else:
        records_dir = input_path
        source_root = input_path.parent if input_path.name == "records" else input_path
    return (RecordItem(read_json(path), path) for path in sorted(records_dir.glob("*.json"))), source_root


def table_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return list(iter_jsonl(path))
    if suffix == ".json":
        data = read_json(path)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            for key in ("items", "records", "data", "rows"):
                if isinstance(data.get(key), list):
                    return [row for row in data[key] if isinstance(row, dict)]
            if all(isinstance(value, dict) for value in data.values()):
                return [{"uid": key, **value} for key, value in data.items()]
            return [data]
        return []
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported table format: {path}")


def group_by_uid(rows: list[dict[str, Any]], uid_field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        uid = row.get(uid_field) or row.get("uid")
        if not uid:
            continue
        grouped.setdefault(str(uid), []).append(row)
    return grouped


def selected_items(items: Iterable[RecordItem], args: argparse.Namespace) -> Iterable[RecordItem]:
    wanted = set(args.uid or [])
    seen = 0
    yielded = 0
    for item in items:
        uid = str(item.record.get("uid", ""))
        if wanted and uid not in wanted:
            continue
        if seen < args.offset:
            seen += 1
            continue
        if args.limit is not None and yielded >= args.limit:
            break
        seen += 1
        yielded += 1
        yield item


def resolve_local_path(local_path: str | None, root: Path) -> Path | None:
    if not local_path:
        return None
    path = Path(local_path)
    return path if path.is_absolute() else root / path


def relative_image_destination(source: Path, role: str, uid: str) -> Path:
    parts = list(source.parts)
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[idx:])
    return Path("images") / role / uid / source.name


def copy_record_images(record: dict[str, Any], source_root: Path, out_root: Path) -> tuple[int, int]:
    copied = 0
    missing = 0
    uid = str(record.get("uid", "unknown"))
    for role in POST_IMAGE_FIELDS:
        post = record.get(role)
        if not isinstance(post, dict):
            continue
        for image in post.get("images") or []:
            local_path = image.get("local_path")
            source = resolve_local_path(local_path, source_root)
            if not source or not source.exists():
                if local_path:
                    image["source_local_path"] = local_path
                    image["copy_error"] = "local_image_missing"
                    missing += 1
                continue
            rel_dest = relative_image_destination(source, role, uid)
            dest = out_root / rel_dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            image["local_path"] = str(rel_dest)
            copied += 1
    return copied, missing


def write_record(record: dict[str, Any], out_dir: Path) -> Path:
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    path = records_dir / f"{record['uid']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map labeled records to benchmark and distractor assets by UID.")
    parser.add_argument("--input", default="labeled_records", help="Labeled output directory, records directory, JSON, or JSONL.")
    parser.add_argument("--out", default="benchmark_mapped")
    parser.add_argument("--benchmark-map", default=None, help="JSONL/JSON/CSV/TSV with benchmark rows and a uid column.")
    parser.add_argument("--distractors", default=None, help="JSONL/JSON/CSV/TSV with distractor rows and a uid column.")
    parser.add_argument("--benchmark-uid-field", default="uid")
    parser.add_argument("--distractor-uid-field", default="uid")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--uid", action="append", default=[], help="Map only this UID. Can repeat.")
    parser.add_argument("--require-benchmark-match", action="store_true")
    parser.add_argument("--require-distractors", action="store_true")
    parser.add_argument("--no-copy-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        return 1

    benchmark_path = Path(args.benchmark_map) if args.benchmark_map else None
    distractor_path = Path(args.distractors) if args.distractors else None

    try:
        benchmark_rows = table_rows(benchmark_path)
        distractor_rows = table_rows(distractor_path)
    except Exception as exc:
        print(f"[ERROR] Could not load mapping tables: {exc}")
        return 1

    benchmark_by_uid = group_by_uid(benchmark_rows, args.benchmark_uid_field)
    distractors_by_uid = group_by_uid(distractor_rows, args.distractor_uid_field)

    raw_items, source_root = iter_records(input_path)
    items = selected_items(raw_items, args)

    stats = {
        "selected": 0,
        "written": 0,
        "benchmark_matched": 0,
        "distractor_matched": 0,
        "skipped_no_benchmark": 0,
        "skipped_no_distractors": 0,
        "images_copied": 0,
        "images_missing_on_copy": 0,
        "benchmark_rows_loaded": len(benchmark_rows),
        "distractor_rows_loaded": len(distractor_rows),
        "started_at": utc_now(),
        "finished_at": None,
    }

    jsonl_path = out_dir / "benchmark_mapped_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in items:
            stats["selected"] += 1
            uid = str(item.record.get("uid", ""))
            benchmark_entries = benchmark_by_uid.get(uid, [])
            distractor_entries = distractors_by_uid.get(uid, [])

            if args.require_benchmark_match and not benchmark_entries:
                stats["skipped_no_benchmark"] += 1
                continue
            if args.require_distractors and not distractor_entries:
                stats["skipped_no_distractors"] += 1
                continue

            if benchmark_entries:
                stats["benchmark_matched"] += 1
            if distractor_entries:
                stats["distractor_matched"] += 1

            out_record = copy.deepcopy(item.record)
            out_record["benchmark_membership"] = benchmark_entries
            out_record["distractor_assets"] = distractor_entries
            out_record["mapping_metadata"] = {
                "mapped_at": utc_now(),
                "source_input": str(input_path),
                "benchmark_map": str(benchmark_path) if benchmark_path else None,
                "distractors": str(distractor_path) if distractor_path else None,
                "join_key": "uid",
            }

            if not args.no_copy_images:
                copied, missing = copy_record_images(out_record, source_root, out_dir)
                stats["images_copied"] += copied
                stats["images_missing_on_copy"] += missing

            write_record(out_record, out_dir)
            handle.write(json.dumps(out_record, ensure_ascii=False, sort_keys=True) + "\n")
            stats["written"] += 1

    stats["finished_at"] = utc_now()
    (out_dir / "mapping_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))
    if not benchmark_rows and not distractor_rows:
        print("[NOTE] No benchmark/distractor maps were provided; records were copied with empty mapping fields.")
    return 0 if stats["written"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
