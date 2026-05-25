#!/usr/bin/env python3
"""
Analyze observable characteristics of naturally occurring meme replies.

This script intentionally does NOT use discourse_function or stance labels.
It summarizes collection, context, relay, comparison-reply, engagement,
validation, and visual-description fields that are directly observable in the
dataset.

Example:
  python analyze_meme_reply_characteristics.py \
    --input ../03_filter_and_label/labeled_final/labeled_memes_with_context.jsonl \
    --out ./analysis_out

Inputs can be:
  - JSONL files
  - JSON files containing one record or a list of records
  - directories containing records/*.json
  - directories containing *.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"[WARN] skip malformed JSONL line {path}:{line_no}: {exc}")
                continue
            if isinstance(obj, dict):
                yield obj


def iter_json(path: Path) -> Iterable[dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] skip malformed JSON file {path}: {exc}")
        return
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        yield obj


def iter_records_from_path(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_file():
        if path.suffix == ".jsonl":
            yield from iter_jsonl(path)
        elif path.suffix == ".json":
            yield from iter_json(path)
        return

    if not path.is_dir():
        print(f"[WARN] input path not found: {path}")
        return

    records_dir = path / "records"
    if records_dir.exists():
        for fp in sorted(records_dir.glob("*.json")):
            yield from iter_json(fp)
        return

    for fp in sorted(path.glob("*.jsonl")):
        yield from iter_jsonl(fp)
    for fp in sorted(path.glob("*.json")):
        yield from iter_json(fp)


def load_records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        for record in iter_records_from_path(path):
            uid = str(record.get("uid") or record.get("uri") or "")
            if uid and uid in seen:
                continue
            if uid:
                seen.add(uid)
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Safe accessors
# ---------------------------------------------------------------------------

def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def get_path(obj: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def text_of(post: Any) -> str:
    return str(as_dict(post).get("text") or "")


def uri_of(post: Any) -> str:
    return str(as_dict(post).get("uri") or "")


def bool_has_images(post: Any) -> bool:
    p = as_dict(post)
    images = as_list(p.get("images"))
    return bool(images) or bool(p.get("has_image"))


def image_count(post: Any) -> int:
    return len(as_list(as_dict(post).get("images")))


def num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def int_or_zero(value: Any) -> int:
    x = num(value)
    return int(x) if x is not None else 0


def parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        pass
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def month_of_dt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m") if dt else "unknown"


def created_dt(record: dict[str, Any]) -> datetime | None:
    return (
        parse_dt(get_path(record, "meme_reply", "created_at"))
        or parse_dt(record.get("created_at"))
        or parse_dt(record.get("process_date"))
    )


def created_month(record: dict[str, Any]) -> str:
    return month_of_dt(created_dt(record))


def get_visual_description(record: dict[str, Any]) -> str:
    candidates = [
        record.get("visual_description"),
        get_path(record, "meme_reply", "visual_description"),
        get_path(record, "meme_reply", "visual", "visual_description"),
        get_path(record, "discourse_labels", "meme_reply", "visual", "visual_description"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_template_names(record: dict[str, Any]) -> list[str]:
    names = []
    validation = as_dict(record.get("meme_validation"))
    for item in as_list(validation.get("validations")):
        item = as_dict(item)
        name = item.get("template_name")
        if isinstance(name, str) and name.strip() and name.strip().lower() not in {"none", "null"}:
            names.append(name.strip())
    return names


def get_validation_confidences(record: dict[str, Any]) -> list[float]:
    vals = []
    validation = as_dict(record.get("meme_validation"))
    for item in as_list(validation.get("validations")):
        conf = num(as_dict(item).get("confidence"))
        if conf is not None:
            vals.append(conf)
    return vals


def get_thread_depth(record: dict[str, Any]) -> int:
    depth = num(get_path(record, "thread_structure", "depth"))
    if depth is not None:
        return int(depth)
    if record.get("parent_reply"):
        return 2
    meme = as_dict(record.get("meme_reply"))
    return 2 if meme.get("is_re_reply") else 1


def get_structure_label(record: dict[str, Any]) -> str:
    label = get_path(record, "thread_structure", "label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return "re-reply" if get_thread_depth(record) >= 2 else "reply"


def get_root_uri(record: dict[str, Any]) -> str:
    return (
        uri_of(record.get("original_post"))
        or str(get_path(record, "meme_reply", "root_uri") or "")
        or uri_of(record.get("meme_reply"))
    )


def get_parent_uri(record: dict[str, Any]) -> str:
    return (
        str(get_path(record, "meme_reply", "parent_uri") or "")
        or uri_of(record.get("parent_reply"))
        or get_root_uri(record)
    )


def get_meme_uri(record: dict[str, Any]) -> str:
    return str(record.get("uri") or uri_of(record.get("meme_reply")) or record.get("uid") or "")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def pct(part: int | float, total: int | float) -> float:
    return round((part / total * 100), 2) if total else 0.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def describe(values: Iterable[Any]) -> dict[str, Any]:
    vals = [float(v) for v in values if num(v) is not None]
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p10": round(percentile(vals, 0.10), 4),
        "p90": round(percentile(vals, 0.90), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def counter_rows(counter: Counter, total: int, name_col: str = "value") -> list[dict[str, Any]]:
    return [
        {name_col: key, "count": count, "percent": pct(count, total)}
        for key, count in counter.most_common()
    ]


def desc_rows(groups: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows = []
    for metric, values in groups.items():
        row = {"metric": metric}
        row.update(describe(values))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_feature_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_feature_rows_with_options(records, ignore_ancestor_chain=False)


def build_feature_rows_with_options(
    records: list[dict[str, Any]],
    ignore_ancestor_chain: bool = False,
) -> list[dict[str, Any]]:
    meme_uris = {get_meme_uri(r) for r in records if get_meme_uri(r)}
    by_meme_uri = {get_meme_uri(r): r for r in records if get_meme_uri(r)}

    root_groups: dict[str, list[int]] = defaultdict(list)
    parent_groups: dict[str, list[int]] = defaultdict(list)
    dt_by_idx: dict[int, datetime | None] = {}

    for i, record in enumerate(records):
        root_groups[get_root_uri(record)].append(i)
        parent_groups[get_parent_uri(record)].append(i)
        dt_by_idx[i] = created_dt(record)

    thread_rank = {}
    thread_prior_count = {}
    thread_prev_gap = {}
    same_parent_rank = {}
    same_parent_prior_count = {}
    same_parent_prev_gap = {}

    def sort_key(idx: int) -> tuple[int, datetime, int]:
        dt = dt_by_idx.get(idx)
        fallback = datetime.min.replace(tzinfo=timezone.utc)
        return (0 if dt else 1, dt or fallback, idx)

    for _, indices in root_groups.items():
        sorted_indices = sorted(indices, key=sort_key)
        prev_dt = None
        for rank, idx in enumerate(sorted_indices, start=1):
            dt = dt_by_idx.get(idx)
            thread_rank[idx] = rank
            thread_prior_count[idx] = rank - 1
            thread_prev_gap[idx] = (
                abs((dt - prev_dt).total_seconds())
                if dt and prev_dt
                else None
            )
            if dt:
                prev_dt = dt

    for _, indices in parent_groups.items():
        sorted_indices = sorted(indices, key=sort_key)
        prev_dt = None
        for rank, idx in enumerate(sorted_indices, start=1):
            dt = dt_by_idx.get(idx)
            same_parent_rank[idx] = rank
            same_parent_prior_count[idx] = rank - 1
            same_parent_prev_gap[idx] = (
                abs((dt - prev_dt).total_seconds())
                if dt and prev_dt
                else None
            )
            if dt:
                prev_dt = dt

    def consecutive_meme_parent_depth(record: dict[str, Any]) -> int:
        depth = 0
        seen = set()
        parent_uri = get_parent_uri(record)
        while parent_uri and parent_uri in by_meme_uri and parent_uri not in seen:
            seen.add(parent_uri)
            depth += 1
            parent_record = by_meme_uri[parent_uri]
            parent_uri = get_parent_uri(parent_record)
        return depth

    rows: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        original = as_dict(record.get("original_post"))
        parent = as_dict(record.get("parent_reply"))
        meme = as_dict(record.get("meme_reply"))
        quoted = as_dict(record.get("quoted_post"))
        best = as_dict(record.get("best_reply_before_meme"))
        closest_text = as_dict(record.get("closest_text_reply"))
        closest_sibling = as_dict(record.get("closest_sibling_text_reply"))
        comparison = as_dict(record.get("comparison_reply"))
        validation = as_dict(record.get("meme_validation"))
        ancestor_chain = [] if ignore_ancestor_chain else as_list(record.get("ancestor_chain"))
        visual_desc = get_visual_description(record)
        template_names = get_template_names(record)
        confidences = get_validation_confidences(record)

        root_uri = get_root_uri(record)
        parent_uri = get_parent_uri(record)
        meme_uri = get_meme_uri(record)

        ancestor_dataset_meme_count = sum(1 for node in ancestor_chain if uri_of(node) in meme_uris)
        ancestor_image_count = sum(image_count(node) for node in ancestor_chain)
        ancestor_quote_count = sum(1 for node in ancestor_chain if as_dict(node).get("quoted_post"))
        ancestor_quote_image_count = sum(
            image_count(as_dict(node).get("quoted_post"))
            for node in ancestor_chain
            if as_dict(node).get("quoted_post")
        )

        root_has_image = bool_has_images(original)
        parent_has_image = bool_has_images(parent)
        quoted_has_image = bool_has_images(quoted)
        meme_image_count = image_count(meme)
        context_image_count = (
            image_count(original)
            + image_count(parent)
            + image_count(quoted)
            + ancestor_image_count
            + ancestor_quote_image_count
        )

        meme_likes = int_or_zero(meme.get("like_count"))
        meme_replies = int_or_zero(meme.get("reply_count"))
        comp_likes = num(comparison.get("like_count"))
        best_likes = num(best.get("like_count"))

        parent_is_dataset_meme = parent_uri in meme_uris
        direct_meme_relay_depth = consecutive_meme_parent_depth(record)
        same_thread_meme_count = len(root_groups.get(root_uri, []))
        same_parent_meme_count = len(parent_groups.get(parent_uri, []))

        row = {
            "uid": record.get("uid") or "",
            "uri": meme_uri,
            "root_uri": root_uri,
            "parent_uri": parent_uri,
            "created_month": created_month(record),
            "process_date": record.get("process_date") or "",
            "process_month": str(record.get("process_date") or "")[:7] or "unknown",

            # classifier / validation
            "meme_prob": record.get("meme_prob"),
            "threshold": record.get("threshold"),
            "is_meme": bool(record.get("is_meme", True)),
            "validation_passed": validation.get("passed"),
            "valid_ratio": validation.get("valid_ratio"),
            "validation_max_confidence": max(confidences) if confidences else None,
            "template_names": "; ".join(template_names),
            "template_count": len(template_names),

            # structure
            "thread_depth": get_thread_depth(record),
            "structure_label": get_structure_label(record),
            "is_re_reply": bool(parent),
            "ancestor_chain_length": len(ancestor_chain),
            "thread_meme_count": same_thread_meme_count,
            "thread_meme_rank": thread_rank.get(i),
            "thread_prior_meme_count": thread_prior_count.get(i),
            "thread_gap_from_previous_meme_seconds": thread_prev_gap.get(i),
            "same_parent_meme_count": same_parent_meme_count,
            "same_parent_meme_rank": same_parent_rank.get(i),
            "same_parent_prior_meme_count": same_parent_prior_count.get(i),
            "same_parent_gap_from_previous_meme_seconds": same_parent_prev_gap.get(i),

            # context availability
            "has_original_post": bool(original),
            "has_parent_reply": bool(parent),
            "has_quoted_post": bool(quoted),
            "root_has_image": root_has_image,
            "parent_has_image": parent_has_image,
            "quoted_has_image": quoted_has_image,
            "meme_image_count": meme_image_count,
            "context_image_count": context_image_count,
            "ancestor_image_count": ancestor_image_count,
            "ancestor_quote_count": ancestor_quote_count,
            "ancestor_quote_image_count": ancestor_quote_image_count,

            # relay / chain indicators
            "parent_is_dataset_meme_reply": parent_is_dataset_meme,
            "ancestor_dataset_meme_count": ancestor_dataset_meme_count,
            "direct_meme_relay_depth": direct_meme_relay_depth,
            "has_direct_meme_relay": direct_meme_relay_depth > 0,
            "has_thread_meme_cluster": same_thread_meme_count >= 2,
            "has_same_parent_meme_cluster": same_parent_meme_count >= 2,
            "has_prior_meme_in_thread": int(thread_prior_count.get(i) or 0) > 0,
            "has_prior_meme_same_parent": int(same_parent_prior_count.get(i) or 0) > 0,
            "image_to_image_reply_candidate": parent_has_image and meme_image_count > 0,

            # text / visual description
            "meme_text_len": len(text_of(meme)),
            "meme_has_text": bool(text_of(meme).strip()),
            "original_text_len": len(text_of(original)),
            "parent_text_len": len(text_of(parent)),
            "comparison_text_len": len(text_of(comparison)),
            "visual_description_available": bool(visual_desc),
            "visual_description_len": len(visual_desc),
            "visual_description_word_count": len(visual_desc.split()) if visual_desc else 0,

            # engagement
            "meme_like_count": meme_likes,
            "meme_reply_count": meme_replies,
            "original_like_count": int_or_zero(original.get("like_count")),
            "original_reply_count": int_or_zero(original.get("reply_count")),
            "parent_like_count": int_or_zero(parent.get("like_count")),
            "parent_reply_count": int_or_zero(parent.get("reply_count")),

            # comparison replies
            "has_best_reply_before_meme": bool(best),
            "best_reply_like_count": best_likes,
            "best_reply_has_image": bool_has_images(best),
            "has_closest_text_reply": bool(closest_text),
            "closest_text_time_delta_seconds": closest_text.get("time_delta_seconds"),
            "has_closest_sibling_text_reply": bool(closest_sibling),
            "closest_sibling_time_delta_seconds": closest_sibling.get("time_delta_seconds"),
            "has_comparison_reply": bool(comparison),
            "comparison_selected_by": comparison.get("selected_by") or "",
            "comparison_like_count": comp_likes,
            "comparison_reply_count": comparison.get("reply_count"),
            "comparison_time_delta_seconds": comparison.get("time_delta_seconds"),
            "meme_more_liked_than_comparison": (
                meme_likes > comp_likes if comp_likes is not None else None
            ),
            "meme_more_liked_than_best_before": (
                meme_likes > best_likes if best_likes is not None else None
            ),
            "meme_to_comparison_like_ratio": (
                round((meme_likes + 1) / (comp_likes + 1), 4)
                if comp_likes is not None else None
            ),
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def count_bool(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def summarize(
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    out_dir: Path,
    ignore_ancestor_chain: bool = False,
) -> None:
    total = len(rows)

    # Full feature table
    write_csv(out_dir / "record_features.csv", rows)

    # Basic counts
    unique_roots = len({row["root_uri"] for row in rows if row.get("root_uri")})
    unique_parents = len({row["parent_uri"] for row in rows if row.get("parent_uri")})
    basic_rows = [
        {"metric": "records", "value": total, "percent": 100.0},
        {"metric": "unique_root_threads", "value": unique_roots, "percent": ""},
        {"metric": "unique_parent_nodes", "value": unique_parents, "percent": ""},
        {"metric": "re_replies", "value": count_bool(rows, "is_re_reply"), "percent": pct(count_bool(rows, "is_re_reply"), total)},
        {"metric": "records_with_parent_reply", "value": count_bool(rows, "has_parent_reply"), "percent": pct(count_bool(rows, "has_parent_reply"), total)},
        {"metric": "records_with_quoted_post", "value": count_bool(rows, "has_quoted_post"), "percent": pct(count_bool(rows, "has_quoted_post"), total)},
        {"metric": "records_with_visual_description", "value": count_bool(rows, "visual_description_available"), "percent": pct(count_bool(rows, "visual_description_available"), total)},
        {"metric": "records_with_comparison_reply", "value": count_bool(rows, "has_comparison_reply"), "percent": pct(count_bool(rows, "has_comparison_reply"), total)},
    ]
    write_csv(out_dir / "table_basic_counts.csv", basic_rows)

    # Monthly counts
    write_csv(
        out_dir / "table_monthly_counts.csv",
        counter_rows(Counter(row["created_month"] for row in rows), total, "month"),
    )

    # Structure
    write_csv(
        out_dir / "table_thread_structure.csv",
        counter_rows(Counter(row["structure_label"] for row in rows), total, "structure_label"),
    )
    write_csv(
        out_dir / "table_thread_depth.csv",
        counter_rows(Counter(row["thread_depth"] for row in rows), total, "thread_depth"),
    )

    # Context availability
    context_rows = []
    for key in [
        "has_original_post",
        "has_parent_reply",
        "has_quoted_post",
        "root_has_image",
        "parent_has_image",
        "quoted_has_image",
        "image_to_image_reply_candidate",
    ]:
        count = count_bool(rows, key)
        context_rows.append({"field": key, "count": count, "percent": pct(count, total)})
    write_csv(out_dir / "table_context_availability.csv", context_rows)

    # Relay / chain statistics
    relay_rows = []
    for key in [
        "has_direct_meme_relay",
        "parent_is_dataset_meme_reply",
        "has_thread_meme_cluster",
        "has_same_parent_meme_cluster",
        "has_prior_meme_in_thread",
        "has_prior_meme_same_parent",
        "image_to_image_reply_candidate",
    ]:
        count = count_bool(rows, key)
        relay_rows.append({"feature": key, "count": count, "percent": pct(count, total)})
    write_csv(out_dir / "table_meme_relay_features.csv", relay_rows)

    write_csv(
        out_dir / "table_direct_meme_relay_depth.csv",
        counter_rows(Counter(row["direct_meme_relay_depth"] for row in rows), total, "direct_meme_relay_depth"),
    )

    # Thread/parent clusters
    thread_counts = Counter(row["root_uri"] for row in rows if row.get("root_uri"))
    parent_counts = Counter(row["parent_uri"] for row in rows if row.get("parent_uri"))
    top_threads = [
        {"root_uri": uri, "meme_reply_count": count}
        for uri, count in thread_counts.most_common(50)
    ]
    top_parents = [
        {"parent_uri": uri, "meme_reply_count": count}
        for uri, count in parent_counts.most_common(50)
    ]
    write_csv(out_dir / "top_threads_by_meme_count.csv", top_threads)
    write_csv(out_dir / "top_parents_by_meme_count.csv", top_parents)

    # Comparison reply selected_by
    selected = Counter(row["comparison_selected_by"] or "none" for row in rows)
    write_csv(
        out_dir / "table_comparison_selected_by.csv",
        counter_rows(selected, total, "selected_by"),
    )

    # Numeric descriptions
    numeric_groups = {
        "meme_prob": [row["meme_prob"] for row in rows],
        "valid_ratio": [row["valid_ratio"] for row in rows],
        "validation_max_confidence": [row["validation_max_confidence"] for row in rows],
        "meme_like_count": [row["meme_like_count"] for row in rows],
        "meme_reply_count": [row["meme_reply_count"] for row in rows],
        "parent_like_count": [row["parent_like_count"] for row in rows],
        "thread_meme_count": [row["thread_meme_count"] for row in rows],
        "same_parent_meme_count": [row["same_parent_meme_count"] for row in rows],
        "thread_gap_from_previous_meme_seconds": [row["thread_gap_from_previous_meme_seconds"] for row in rows],
        "same_parent_gap_from_previous_meme_seconds": [row["same_parent_gap_from_previous_meme_seconds"] for row in rows],
        "comparison_time_delta_seconds": [row["comparison_time_delta_seconds"] for row in rows],
        "closest_text_time_delta_seconds": [row["closest_text_time_delta_seconds"] for row in rows],
        "closest_sibling_time_delta_seconds": [row["closest_sibling_time_delta_seconds"] for row in rows],
        "meme_to_comparison_like_ratio": [row["meme_to_comparison_like_ratio"] for row in rows],
        "meme_text_len": [row["meme_text_len"] for row in rows],
        "original_text_len": [row["original_text_len"] for row in rows],
        "parent_text_len": [row["parent_text_len"] for row in rows],
        "visual_description_word_count": [row["visual_description_word_count"] for row in rows],
        "context_image_count": [row["context_image_count"] for row in rows],
    }
    write_csv(out_dir / "table_numeric_summary.csv", desc_rows(numeric_groups))

    # Comparison win rates
    comp_rows = []
    for key in ["meme_more_liked_than_comparison", "meme_more_liked_than_best_before"]:
        comparable = [row for row in rows if row.get(key) is not None]
        wins = sum(1 for row in comparable if row.get(key) is True)
        comp_rows.append({
            "metric": key,
            "comparable_n": len(comparable),
            "meme_higher_count": wins,
            "meme_higher_percent": pct(wins, len(comparable)),
        })
    write_csv(out_dir / "table_engagement_comparison.csv", comp_rows)

    # Validation templates
    template_counter = Counter()
    for row in rows:
        for name in str(row.get("template_names") or "").split("; "):
            if name:
                template_counter[name] += 1
    write_csv(
        out_dir / "table_template_names.csv",
        counter_rows(template_counter, sum(template_counter.values()), "template_name"),
    )

    # Candidate examples for manual inspection.
    relay_candidates = []
    for row, record in zip(rows, records):
        if (
            row["has_direct_meme_relay"]
            or row["has_prior_meme_same_parent"]
            or row["has_prior_meme_in_thread"]
            or row["image_to_image_reply_candidate"]
        ):
            relay_candidates.append({
                "uid": row["uid"],
                "uri": row["uri"],
                "root_uri": row["root_uri"],
                "parent_uri": row["parent_uri"],
                "feature_flags": {
                    "has_direct_meme_relay": row["has_direct_meme_relay"],
                    "has_prior_meme_same_parent": row["has_prior_meme_same_parent"],
                    "has_prior_meme_in_thread": row["has_prior_meme_in_thread"],
                    "image_to_image_reply_candidate": row["image_to_image_reply_candidate"],
                },
                "meme_text_preview": text_of(record.get("meme_reply"))[:200],
                "parent_text_preview": text_of(record.get("parent_reply"))[:200],
                "original_text_preview": text_of(record.get("original_post"))[:200],
            })
    with (out_dir / "meme_relay_candidates.jsonl").open("w", encoding="utf-8") as f:
        for item in relay_candidates[:1000]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_summary_md(out_dir, rows, basic_rows, relay_rows, comp_rows, ignore_ancestor_chain)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep] + body)


def write_summary_md(
    out_dir: Path,
    rows: list[dict[str, Any]],
    basic_rows: list[dict[str, Any]],
    relay_rows: list[dict[str, Any]],
    comp_rows: list[dict[str, Any]],
    ignore_ancestor_chain: bool = False,
) -> None:
    total = len(rows)

    def first_value(metric: str) -> Any:
        for row in basic_rows:
            if row["metric"] == metric:
                return row["value"]
        return ""

    numeric = {row["metric"]: row for row in desc_rows({
        "meme_like_count": [r["meme_like_count"] for r in rows],
        "meme_reply_count": [r["meme_reply_count"] for r in rows],
        "thread_meme_count": [r["thread_meme_count"] for r in rows],
        "same_parent_meme_count": [r["same_parent_meme_count"] for r in rows],
        "comparison_time_delta_seconds": [r["comparison_time_delta_seconds"] for r in rows],
    })}

    research_map_rows = [
        {
            "Paper angle": "Memes as conversational replies",
            "Observable evidence in this dataset": (
                "reply/re-reply rate, depth, and parent availability"
                if ignore_ancestor_chain
                else "reply/re-reply rate, depth, parent availability, ancestor-chain length"
            ),
            "Output tables": "table_thread_structure.csv, table_thread_depth.csv, table_context_availability.csv",
        },
        {
            "Paper angle": "Memes as participatory circulation or relay",
            "Observable evidence in this dataset": "direct meme-to-meme parent chains, multiple meme replies in the same thread/parent, temporal gaps between meme replies",
            "Output tables": "table_meme_relay_features.csv, top_threads_by_meme_count.csv, top_parents_by_meme_count.csv",
        },
        {
            "Paper angle": "Memes as context-dependent multimodal artifacts",
            "Observable evidence in this dataset": "quoted posts, context images, parent/root images, visual descriptions, meme templates",
            "Output tables": "table_context_availability.csv, table_template_names.csv, table_numeric_summary.csv",
        },
        {
            "Paper angle": "Memes as engagement-bearing replies",
            "Observable evidence in this dataset": "meme likes/replies, comparison with nearby text-only replies and best prior sibling replies",
            "Output tables": "table_engagement_comparison.csv, table_comparison_selected_by.csv",
        },
    ]

    lines = [
        "# Meme Reply Characteristics Summary",
        "",
        f"- Records analyzed: **{total:,}**",
        f"- Unique root threads: **{first_value('unique_root_threads')}**",
        f"- Unique parent nodes: **{first_value('unique_parent_nodes')}**",
        "",
        "## Basic Counts",
        "",
        md_table(basic_rows, ["metric", "value", "percent"]),
        "",
        "## Meme Relay / Cluster Features",
        "",
        md_table(relay_rows, ["feature", "count", "percent"]),
        "",
        "## Engagement Comparison",
        "",
        md_table(comp_rows, ["metric", "comparable_n", "meme_higher_count", "meme_higher_percent"]),
        "",
        "## Selected Numeric Summaries",
        "",
        md_table(
            [
                {"metric": key, **value}
                for key, value in numeric.items()
            ],
            ["metric", "n", "mean", "median", "p10", "p90", "min", "max"],
        ),
        "",
        "## How These Descriptives Map to Prior Meme-Research Claims",
        "",
        "These are observable, non-discourse-label features. They do not claim to infer conversational acts directly.",
        "",
        md_table(
            research_map_rows,
            ["Paper angle", "Observable evidence in this dataset", "Output tables"],
        ),
        "",
        "## Important Caveats",
        "",
        "- `has_direct_meme_relay` is strict: the meme reply's direct parent must also appear as a meme reply in this dataset.",
        "- `image_to_image_reply_candidate` is broader: the parent has an image, but that parent image has not necessarily passed meme validation.",
        "- Thread and parent meme clusters indicate co-occurrence of meme replies, not necessarily direct conversational uptake.",
        "- Comparison-reply statistics depend on the auxiliary reply selection rules in the collection pipeline.",
        "- No `discourse_function` or stance attributes are used in any table generated by this script.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze meme reply characteristics without discourse labels.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="JSONL/JSON file(s), records directory, or dataset directory containing records/*.json",
    )
    parser.add_argument("--out", default="./analysis_out", help="Output directory for CSV/MD files.")
    parser.add_argument(
        "--ignore-ancestor-chain",
        action="store_true",
        help="Ignore ancestor_chain fields even if they are present in the input JSONL.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input)
    print(f"[LOAD] {len(records):,} records")
    if not records:
        raise SystemExit("No records loaded.")

    rows = build_feature_rows_with_options(records, ignore_ancestor_chain=args.ignore_ancestor_chain)
    summarize(records, rows, out_dir, ignore_ancestor_chain=args.ignore_ancestor_chain)
    print(f"[DONE] wrote analysis files to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
