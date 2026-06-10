#!/usr/bin/env python3
"""
Hydrate benchmark records from a benchmark UID list.

This is the UID-only entry point for the 5,000-item benchmark release. A
benchmark UID list does not contain enough information to query the Bluesky API
directly because project UIDs keep only a truncated DID suffix. This script
therefore joins benchmark UIDs against the released collection UID/URI manifest
and hydrates only those matched rows.

Example:

  python hydrate_benchmark_from_uids.py \
    --benchmark-uids ../04_benchmark/benchmark_data \
    --manifest data/collection_pool_uid_manifest.jsonl \
    --out benchmark_hydrated \
    --download-images context
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from hydrate_from_uid_manifest import (
    BskyClient,
    HYDRATED_FIELDS,
    build_record,
    download_record_images,
    read_manifest,
    utc_now,
    write_record,
)


UID_KEYS = (
    "uid",
    "id",
    "meme_uid",
    "meme_reply_uid",
    "benchmark_uid",
    "item_uid",
)


def normalize_uid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("bsky_"):
        return text
    return None


def extract_uid(row: Any) -> str | None:
    if isinstance(row, str):
        token = row.strip().split()[0] if row.strip() else ""
        return normalize_uid(token.strip(","))

    if isinstance(row, (list, tuple)):
        for value in row:
            uid = normalize_uid(value)
            if uid:
                return uid

    if isinstance(row, dict):
        for key in UID_KEYS:
            uid = normalize_uid(row.get(key))
            if uid:
                return uid
        for value in row.values():
            uid = normalize_uid(value)
            if uid:
                return uid

    return None


def rows_from_json(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("uids", "items", "records", "data", "benchmark"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return list(data.values())
    raise ValueError(f"Unsupported JSON benchmark UID file: {path}")


def rows_from_jsonl_or_text(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
            else:
                rows.append(line)
    return rows


def rows_from_table(path: Path, delimiter: str) -> list[Any]:
    with path.open(newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample else False
        if has_header:
            return list(csv.DictReader(f, delimiter=delimiter))
        return [row for row in csv.reader(f, delimiter=delimiter) if row]


def rows_from_benchmark_dir(path: Path) -> list[Any]:
    summary_path = path / "benchmark_summary.jsonl"
    if summary_path.exists():
        return rows_from_jsonl_or_text(summary_path)

    rows: list[Any] = []
    for meta_path in sorted(path.glob("*/meta.json")):
        try:
            rows.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{meta_path}: {exc}") from exc

    if rows:
        return rows

    return [child.name for child in sorted(path.iterdir()) if child.is_dir() and child.name.startswith("bsky_")]


def read_benchmark_uid_rows(path: Path) -> tuple[list[str], dict[str, list[Any]], list[Any]]:
    if path.is_dir():
        rows = rows_from_benchmark_dir(path)
    else:
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows = rows_from_json(path)
        elif suffix == ".csv":
            rows = rows_from_table(path, ",")
        elif suffix == ".tsv":
            rows = rows_from_table(path, "\t")
        else:
            rows = rows_from_jsonl_or_text(path)

    ordered_uids: list[str] = []
    by_uid: dict[str, list[Any]] = {}
    rejected: list[Any] = []

    for row in rows:
        uid = extract_uid(row)
        if not uid:
            rejected.append(row)
            continue
        if uid not in by_uid:
            ordered_uids.append(uid)
        by_uid.setdefault(uid, []).append(row)

    return ordered_uids, by_uid, rejected


def build_manifest_index(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    for entry in entries:
        uid = normalize_uid(entry.get("uid"))
        if uid and uid not in by_uid:
            by_uid[uid] = entry
    return by_uid


def select_benchmark_entries(
    ordered_uids: list[str],
    benchmark_rows: dict[str, list[Any]],
    manifest_by_uid: dict[str, dict[str, Any]],
    wanted_uids: set[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    missing: list[str] = []

    for uid in ordered_uids:
        if wanted_uids and uid not in wanted_uids:
            continue
        manifest_entry = manifest_by_uid.get(uid)
        if not manifest_entry:
            missing.append(uid)
            continue
        entry = dict(manifest_entry)
        entry["benchmark_uid"] = uid
        entry["benchmark_source_rows"] = benchmark_rows.get(uid, [])
        selected.append(entry)

    return selected, missing


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def apply_slice(rows: list[dict[str, Any]], offset: int, limit: int | None) -> list[dict[str, Any]]:
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hydrate only benchmark UIDs by joining them to a UID/URI manifest."
    )
    parser.add_argument(
        "--benchmark-uids",
        default="data/benchmark_uids.jsonl",
        help="Benchmark UID source: benchmark_data directory, txt, jsonl, json, csv, tsv, or benchmark_summary.jsonl.",
    )
    parser.add_argument("--manifest", default="data/collection_pool_uid_manifest.jsonl")
    parser.add_argument("--out", default="benchmark_hydrated")
    parser.add_argument(
        "--selected-manifest",
        default=None,
        help="Where to write the matched benchmark manifest. Default: <out>/benchmark_uid_manifest.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Hydrate only the first N selected rows.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N selected rows.")
    parser.add_argument("--uid", action="append", default=[], help="Hydrate only this benchmark UID. Can repeat.")
    parser.add_argument(
        "--download-images",
        choices=("none", "meme", "context", "all"),
        default="meme",
        help="Image download scope. Default: meme.",
    )
    parser.add_argument("--parent-height", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay after each API request.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any benchmark UID is absent from the manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Join and preview rows without API calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_path = Path(args.benchmark_uids)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    selected_manifest_path = (
        Path(args.selected_manifest)
        if args.selected_manifest
        else out_dir / "benchmark_uid_manifest.jsonl"
    )
    report_path = out_dir / "benchmark_hydration_report.json"

    if not benchmark_path.exists():
        print(f"[ERROR] Benchmark UID file not found: {benchmark_path}")
        return 1
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("        A benchmark UID alone is not enough; pass the full UID/URI manifest.")
        return 1

    ordered_uids, benchmark_rows, rejected_rows = read_benchmark_uid_rows(benchmark_path)
    manifest_entries = read_manifest(manifest_path)
    manifest_by_uid = build_manifest_index(manifest_entries)
    wanted_uids = set(args.uid) if args.uid else None

    selected, missing = select_benchmark_entries(
        ordered_uids=ordered_uids,
        benchmark_rows=benchmark_rows,
        manifest_by_uid=manifest_by_uid,
        wanted_uids=wanted_uids,
    )
    selected = apply_slice(selected, offset=args.offset, limit=args.limit)
    missing_uri = [
        entry.get("uid")
        for entry in selected
        if not (entry.get("meme_reply_uri") or entry.get("uri"))
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(selected_manifest_path, selected)

    stats: dict[str, Any] = {
        "benchmark_uid_source": str(benchmark_path),
        "manifest": str(manifest_path),
        "selected_manifest": str(selected_manifest_path),
        "manifest_rows": len(manifest_entries),
        "benchmark_rows": sum(len(v) for v in benchmark_rows.values()) + len(rejected_rows),
        "benchmark_uids": len(ordered_uids),
        "rejected_benchmark_rows": len(rejected_rows),
        "matched_uids": len(selected),
        "missing_uids": len(missing),
        "missing_uri_rows": len(missing_uri),
        "download_images": args.download_images,
        "dry_run": args.dry_run,
        "skipped_existing": 0,
        "hydrated": 0,
        "failed": 0,
        "image_downloaded": 0,
        "image_failed": 0,
        "hydrated_field_present": {field: 0 for field in HYDRATED_FIELDS},
        "hydrated_field_non_null": {field: 0 for field in HYDRATED_FIELDS},
        "started_at": utc_now(),
        "finished_at": None,
    }
    failures: list[dict[str, Any]] = []

    if missing and args.strict:
        failures.append({"error": "missing benchmark UIDs in manifest", "uids": missing})

    print(
        json.dumps(
            {
                "benchmark_uids": len(ordered_uids),
                "selected": len(selected),
                "missing": len(missing),
                "missing_uri_rows": len(missing_uri),
                "out": str(out_dir),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.dry_run or failures:
        stats["finished_at"] = utc_now()
        report = {
            "stats": stats,
            "missing_uids": missing,
            "missing_uri_uids": missing_uri,
            "failures": failures,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if failures:
            print(f"[ERROR] Wrote report: {report_path}")
            return 1
        print(f"[DRY-RUN] Wrote selected manifest: {selected_manifest_path}")
        print(f"[DRY-RUN] Wrote report: {report_path}")
        return 0 if selected else 1

    client = BskyClient(timeout=args.timeout, retries=args.retries, sleep=args.sleep)

    for index, entry in enumerate(selected, start=1):
        uid = entry.get("uid") or entry.get("benchmark_uid")
        record_path = out_dir / "records" / f"{uid}.json"
        if record_path.exists() and not args.overwrite:
            print(f"[{index}/{len(selected)}] {uid} [SKIP existing]")
            stats["skipped_existing"] += 1
            continue

        print(f"[{index}/{len(selected)}] {uid}")
        try:
            record = build_record(entry, client, parent_height=args.parent_height)
            record["benchmark_metadata"] = {
                "benchmark_uid_source": str(benchmark_path),
                "source_rows": entry.get("benchmark_source_rows") or [],
            }
            ok_images, failed_images = download_record_images(
                record=record,
                mode=args.download_images,
                out_dir=out_dir,
                client=client,
                overwrite=args.overwrite,
            )
            write_record(record, out_dir, overwrite=args.overwrite)
            stats["hydrated"] += 1
            stats["image_downloaded"] += ok_images
            stats["image_failed"] += failed_images
            for field in HYDRATED_FIELDS:
                if field in record:
                    stats["hydrated_field_present"][field] += 1
                if record.get(field) is not None:
                    stats["hydrated_field_non_null"][field] += 1
        except Exception as exc:
            stats["failed"] += 1
            failures.append(
                {
                    "uid": uid,
                    "uri": entry.get("meme_reply_uri") or entry.get("uri"),
                    "error": str(exc),
                }
            )
            print(f"  [FAIL] {exc}")

    stats["finished_at"] = utc_now()
    report = {
        "stats": stats,
        "missing_uids": missing,
        "missing_uri_uids": missing_uri,
        "failures": failures,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats["hydrated"] or stats["skipped_existing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
