#!/usr/bin/env python3
"""
Add reconstructed conversation-context fields to existing meme dataset records.

This script updates previously collected JSON/JSONL records with:
  1. ancestor_chain: reply ancestors between the root post and the meme reply
  2. quoted_post: hydrated quoted/embedded post content, when available

It is intentionally lightweight and does not import the original meme detector,
because that file loads GPU/OCR dependencies. The parsing helpers below mirror
the collection pipeline's JSON schema.

Examples:
  # Default: update the labeled JSONL used by the benchmark pipeline
  python update_dataset_context_fields.py

  # Update a JSONL file
  python update_dataset_context_fields.py \
    --input ./labeled_memes.jsonl \
    --archive-base /path/to/firehose_archives \
    --output ./labeled_memes_with_context.jsonl

By default, ancestor_chain excludes the direct parent reply because the
benchmark pipeline stores parent_reply separately. Use
--include-parent-in-ancestor-chain if you want a single chain that includes
the direct parent as well.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import time
import re
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


_TID_CHARS = "234567abcdefghijklmnopqrstuvwxyz"
_TID_MAP = {c: i for i, c in enumerate(_TID_CHARS)}

BASE_POST_FIELDS = (
    "uid",
    "uri",
    "did",
    "rkey",
    "seq",
    "text",
    "langs",
    "post_url",
    "created_at",
    "is_reply",
    "is_re_reply",
    "root_uri",
    "parent_uri",
    "like_count",
    "reply_count",
    "has_image",
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Defaults mirrored from the existing collection/benchmark pipeline configs.
DEFAULT_ARCHIVE_BASE = Path("/home/exouser/slate_project/bluesky/firehose_archives")
DEFAULT_INPUT = REPO_ROOT / "03_filter_and_label" / "labeled_final" / "labeled_memes.jsonl"
DEFAULT_COLLECTION_DATASETS = [
    REPO_ROOT / "01_collection" / "meme_dataset_24_06",
    REPO_ROOT / "01_collection" / "meme_dataset_25_02",
    REPO_ROOT / "01_collection" / "meme_dataset",
]


def rkey_to_date(rkey: str) -> date | None:
    """Approximate AT Protocol TID rkey date."""
    if not rkey or len(rkey) < 13:
        return None
    try:
        n = 0
        for c in rkey[:13]:
            n = n * 32 + _TID_MAP[c]
        ts_us = n >> 10
        dt = datetime.fromtimestamp(ts_us / 1_000_000, tz=timezone.utc)
        if 2020 <= dt.year <= 2030:
            return dt.date()
    except Exception:
        return None
    return None


def uri_to_uid(uri: str) -> str:
    try:
        parts = uri.replace("at://", "").split("/")
        did_suffix = parts[0].split(":")[-1][-14:]
        rkey = parts[-1]
        return f"bsky_{did_suffix}_{rkey}"
    except Exception:
        return "bsky_" + hashlib.sha256((uri or "").encode()).hexdigest()[:22]


def uri_to_date(uri: str) -> date | None:
    if not uri:
        return None
    return rkey_to_date(uri.split("/")[-1])


def parse_datetime_safe(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)


def date_from_record(record: dict[str, Any]) -> date:
    if record.get("process_date"):
        try:
            return date.fromisoformat(str(record["process_date"])[:10])
        except Exception:
            pass

    for container in (record.get("meme_reply"), record):
        if isinstance(container, dict):
            dt = parse_datetime_safe(container.get("created_at"))
            if dt != datetime.min.replace(tzinfo=timezone.utc):
                return dt.date()

    meme_uri = record.get("uri") or (record.get("meme_reply") or {}).get("uri")
    return uri_to_date(meme_uri) or date.today()


def archive_path(base: str | Path, d: date) -> Path:
    return Path(base) / f"{d.year}-{d.month:02d}" / f"{d.year}-{d.month:02d}-{d.day:02d}.json.gz"


def safe_uri(value: Any) -> str | None:
    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, str):
        return value or None
    return None


def normalize_embed(embed: Any) -> Any:
    if isinstance(embed, str) and embed.strip().startswith("{"):
        try:
            parsed = ast.literal_eval(embed)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return embed
    return embed


def is_image_embed(embed: Any) -> bool:
    embed = normalize_embed(embed)
    if isinstance(embed, dict):
        embed_type = embed.get("$type", "")
        return embed_type in {
            "app.bsky.embed.images",
            "app.bsky.embed.images#view",
            "app.bsky.embed.recordWithMedia",
            "app.bsky.embed.recordWithMedia#view",
        }
    if isinstance(embed, str):
        return ("Image(" in embed and "app.bsky.embed.images" in embed) or (
            "app.bsky.embed.images" in embed and "images" in embed
        )
    return False


def parse_embed_images(embed: Any, did: str) -> list[dict[str, Any]]:
    embed = normalize_embed(embed)
    if not is_image_embed(embed):
        return []

    if isinstance(embed, dict):
        embed_type = embed.get("$type", "")
        if embed_type in {"app.bsky.embed.recordWithMedia", "app.bsky.embed.recordWithMedia#view"}:
            media = embed.get("media")
            embed = media if isinstance(media, dict) else {}
        raw_images = embed.get("images", []) if isinstance(embed, dict) else []
        images = []
        for img in raw_images:
            alt = img.get("alt", "")
            blob = img.get("image") or img.get("fullsize") or {}
            if isinstance(blob, str):
                cid = blob.rsplit("/", 1)[-1].replace("@jpeg", "")
            else:
                ref = blob.get("ref", "")
                if isinstance(ref, dict):
                    cid = ref.get("$link", "")
                elif ref:
                    cid = str(ref)
                else:
                    cid = blob.get("cid", "")
            if cid:
                images.append(
                    {
                        "cid": cid,
                        "alt": alt,
                        "url": f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg",
                    }
                )
        return images

    images = []
    for chunk in str(embed).split("Image(")[1:]:
        alt = ""
        match = re.match(r"alt='((?:[^'\\]|\\.)*)'", chunk) or re.match(
            r'alt="((?:[^"\\]|\\.)*)"', chunk
        )
        if match:
            alt = match.group(1)
        match = re.search(r"'\$link':\s*'([^']+)'", chunk)
        if match:
            cid = match.group(1)
            images.append(
                {
                    "cid": cid,
                    "alt": alt,
                    "url": f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg",
                }
            )
    return images


def extract_external(embed: Any) -> dict[str, Any]:
    embed = normalize_embed(embed)
    if not embed:
        return {}

    if isinstance(embed, dict):
        embed_type = embed.get("$type", "")
        if embed_type in {"app.bsky.embed.external", "app.bsky.embed.external#view"}:
            external = embed.get("external", {}) or {}
            return {
                "external_title": external.get("title"),
                "external_url": external.get("uri"),
                "external_description": external.get("description"),
            }
        if embed_type in {"app.bsky.embed.recordWithMedia", "app.bsky.embed.recordWithMedia#view"}:
            media = embed.get("media", {}) or {}
            if isinstance(media, dict) and "external" in media:
                external = media["external"] or {}
                return {
                    "external_title": external.get("title"),
                    "external_url": external.get("uri"),
                    "external_description": external.get("description"),
                }

    if isinstance(embed, str) and "External(" in embed:
        title_match = re.search(r"title='((?:[^'\\]|\\.)*)'", embed)
        uri_match = re.search(r"uri='((?:[^'\\]|\\.)*)'", embed)
        title = title_match.group(1) if title_match else None
        url = uri_match.group(1) if uri_match else None
        if title or url:
            return {"external_title": title, "external_url": url, "external_description": None}

    return {}


def find_quoted_uri(embed: Any) -> str | None:
    """Return the quoted/embedded record URI from a Bluesky embed object."""
    embed = normalize_embed(embed)
    if not embed:
        return None

    if isinstance(embed, dict):
        embed_type = embed.get("$type", "")
        if embed_type in {"app.bsky.embed.record", "app.bsky.embed.record#view"}:
            record = embed.get("record")
            if isinstance(record, dict):
                return safe_uri(record.get("uri")) or find_quoted_uri(record)
            return safe_uri(record)

        if embed_type in {"app.bsky.embed.recordWithMedia", "app.bsky.embed.recordWithMedia#view"}:
            record = embed.get("record")
            if isinstance(record, dict):
                return safe_uri(record.get("uri")) or find_quoted_uri(record)
            return safe_uri(record)

        # Some archive/view variants nest the record more deeply.
        for key in ("record", "subject"):
            value = embed.get(key)
            if isinstance(value, dict):
                found = safe_uri(value.get("uri")) or find_quoted_uri(value)
                if found:
                    return found

    if isinstance(embed, str):
        match = re.search(r"at://[A-Za-z0-9:._%-]+/app\.bsky\.feed\.post/[A-Za-z0-9._%-]+", embed)
        if match:
            return match.group(0)

    return None


def obj_to_post(obj: dict[str, Any]) -> dict[str, Any] | None:
    if obj.get("type") != "app.bsky.feed.post" or obj.get("action") == "delete":
        return None

    uri = obj.get("uri", "")
    did = obj.get("author", "")
    record = obj.get("record") if isinstance(obj.get("record"), dict) else {}
    created = obj.get("create_time") or obj.get("createdAt") or record.get("createdAt") or obj.get("commit_time", "")
    text = obj.get("text")
    if text is None:
        text = record.get("text", "")
    langs = obj.get("langs")
    if langs is None:
        langs = record.get("langs") or []

    embed = obj.get("embed") or record.get("embed")
    images = parse_embed_images(embed, did) if embed else []

    reply = obj.get("reply") if isinstance(obj.get("reply"), dict) else record.get("reply")
    reply_ref = reply if isinstance(reply, dict) else {}
    parent = reply_ref.get("parent") or obj.get("parent")
    root = reply_ref.get("root") or obj.get("root")
    parent_uri = safe_uri(parent.get("uri")) if isinstance(parent, dict) else None
    root_uri = safe_uri(root.get("uri")) if isinstance(root, dict) else None
    quoted_uri = find_quoted_uri(embed)

    return {
        "uid": uri_to_uid(uri),
        "uri": uri,
        "did": did,
        "rkey": uri.split("/")[-1] if uri else None,
        "seq": obj.get("seq"),
        "text": text or "",
        "langs": langs or [],
        "post_url": obj.get("url", ""),
        "created_at": created,
        "is_reply": bool(parent_uri),
        "is_re_reply": bool(parent_uri and root_uri and parent_uri != root_uri),
        "root_uri": root_uri,
        "parent_uri": parent_uri,
        "images": images,
        "has_image": len(images) > 0,
        "like_count": 0,
        "reply_count": 0,
        "quoted_post_uri": quoted_uri,
        "embed_type": embed.get("$type") if isinstance(embed, dict) else None,
        "embed": embed,
        "_dt": parse_datetime_safe(created),
    }


def load_day(
    d: date,
    archive_base: str | Path,
    verbose: bool = False,
    include_likes: bool = False,
) -> dict[str, Any] | None:
    path = archive_path(archive_base, d)
    if not path.exists():
        if verbose:
            print(f"[skip] archive not found: {path}")
        return None

    by_uri: dict[str, dict[str, Any]] = {}
    by_parent: dict[str, list[str]] = defaultdict(list)
    like_counts: dict[str, int] = defaultdict(int)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Context reconstruction only needs posts. Skipping JSON parsing for
            # likes/reposts makes large firehose archives much faster to scan.
            if not include_likes and "app.bsky.feed.post" not in line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_type = obj.get("type", "")
            if record_type == "app.bsky.feed.post":
                post = obj_to_post(obj)
                if not post:
                    continue
                by_uri[post["uri"]] = post
                if post.get("parent_uri"):
                    by_parent[post["parent_uri"]].append(post["uri"])
            elif record_type == "app.bsky.feed.like":
                if not include_likes:
                    continue
                subject = obj.get("subject") or obj.get("root") or {}
                liked_uri = safe_uri(subject.get("uri")) if isinstance(subject, dict) else None
                if liked_uri:
                    like_counts[liked_uri] += 1 if obj.get("action") == "create" else -1

    for uri, post in by_uri.items():
        post["like_count"] = max(0, like_counts.get(uri, 0))
        post["reply_count"] = len(by_parent.get(uri, []))

    return {"by_uri": by_uri, "date": d}


class ArchiveManager:
    def __init__(
        self,
        archive_base: str | Path,
        cache_size: int = 14,
        max_days_back: int = 0,
        log_loads: bool = True,
        allow_full_lookup: bool = True,
    ):
        self.archive_base = archive_base
        self.cache_size = cache_size
        self.max_days_back = max_days_back
        self.log_loads = log_loads
        self.allow_full_lookup = allow_full_lookup
        self._cache: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
        self._idx_cache: OrderedDict[str, dict[str, Any] | None] = OrderedDict()
        self._post_cache: dict[str, dict[str, Any] | None] = {}
        self._missing_archive_dates: set[date] = set()

    def _compact(self, by_uri: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        keep = set(BASE_POST_FIELDS) | {
            "images",
            "quoted_post_uri",
            "embed_type",
            "embed",
            "_dt",
        }
        return {uri: {k: v for k, v in post.items() if k in keep} for uri, post in by_uri.items()}

    def _load_date(self, date_str: str) -> None:
        if date_str in self._cache:
            self._cache.move_to_end(date_str)
            return
        try:
            if date_str in self._idx_cache:
                idx = self._idx_cache[date_str]
                self._idx_cache.move_to_end(date_str)
            else:
                if self.log_loads:
                    print(f"[archive-load] {date_str} start", flush=True)
                start = time.time()
                idx = load_day(date.fromisoformat(date_str), self.archive_base, verbose=False)
                if self.log_loads:
                    n_posts = len((idx or {}).get("by_uri", {}))
                    print(f"[archive-load] {date_str} done: {n_posts:,} posts in {time.time() - start:.1f}s", flush=True)
            compact = self._compact(idx["by_uri"]) if idx else {}
        except Exception:
            compact = {}
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[date_str] = compact

    def get_day_index(self, d: date) -> dict[str, Any] | None:
        """Load one archive day once and reuse it across records from that day."""
        date_str = d.isoformat()
        if date_str in self._idx_cache:
            self._idx_cache.move_to_end(date_str)
            return self._idx_cache[date_str]

        if self.log_loads:
            print(f"[archive-load] {date_str} start", flush=True)
        start = time.time()
        idx = load_day(d, self.archive_base, verbose=False)
        if self.log_loads:
            n_posts = len((idx or {}).get("by_uri", {}))
            print(f"[archive-load] {date_str} done: {n_posts:,} posts in {time.time() - start:.1f}s", flush=True)
        if idx:
            idx = {"by_uri": self._compact(idx["by_uri"]), "date": d}

        if len(self._idx_cache) >= self.cache_size:
            self._idx_cache.popitem(last=False)
        self._idx_cache[date_str] = idx
        return idx

    def lookup(
        self,
        uri: str | None,
        current_date: date,
        current_idx: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not uri:
            return None
        if uri in self._post_cache:
            return self._post_cache[uri]

        if current_idx and uri in current_idx.get("by_uri", {}):
            return current_idx["by_uri"][uri]
        if not self.allow_full_lookup:
            return None

        rkey_date = uri_to_date(uri)
        dates_to_try: list[str] = []
        if rkey_date:
            dates_to_try.append(rkey_date.isoformat())

        for delta in range(0, self.max_days_back + 1):
            d_str = (current_date - timedelta(days=delta)).isoformat()
            if d_str not in dates_to_try:
                dates_to_try.append(d_str)

        for d_str in dates_to_try:
            self._load_date(d_str)
            post = self._cache.get(d_str, {}).get(uri)
            if post:
                self._post_cache[uri] = post
                return post
        self._post_cache[uri] = None
        return None

    def scan_date_for_uris(self, d: date, uris: set[str]) -> dict[str, dict[str, Any]]:
        """Scan one archive day and parse only post lines whose top-level URI is needed."""
        missing = {uri for uri in uris if uri and uri not in self._post_cache}
        if not missing:
            return {}

        path = archive_path(self.archive_base, d)
        if not path.exists():
            if d not in self._missing_archive_dates and self.log_loads:
                print(f"[target-scan] {d.isoformat()} missing archive: {path}", flush=True)
            self._missing_archive_dates.add(d)
            return {}

        found: dict[str, dict[str, Any]] = {}
        uri_re = re.compile(r'"uri"\s*:\s*"(at://[^"]+)"')
        start = time.time()
        if self.log_loads:
            print(f"[target-scan] {d.isoformat()} start: {len(missing):,} target URIs", flush=True)

        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "app.bsky.feed.post" not in line:
                    continue
                match = uri_re.search(line)
                if not match:
                    continue
                uri = match.group(1)
                if uri not in missing:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                post = obj_to_post(obj)
                if not post:
                    continue
                self._post_cache[uri] = post
                found[uri] = post
                missing.remove(uri)
                if not missing:
                    break

        if self.log_loads:
            print(
                f"[target-scan] {d.isoformat()} done: found={len(found):,} "
                f"missing={len(missing):,} in {time.time() - start:.1f}s",
                flush=True,
            )
        return found

    def targeted_lookup(self, uri: str | None, fallback_date: date) -> dict[str, Any] | None:
        """Lookup one URI without building a full-day index."""
        if not uri:
            return None
        if uri in self._post_cache and self._post_cache[uri] is not None:
            return self._post_cache[uri]

        dates: list[date] = []
        inferred = uri_to_date(uri)
        if inferred:
            dates.append(inferred)
        if fallback_date not in dates:
            dates.append(fallback_date)

        for d in dates:
            self.scan_date_for_uris(d, {uri})
            post = self._post_cache.get(uri)
            if post:
                return post
        return None

    def prefetch_records(self, records: list[dict[str, Any]], max_hops: int) -> None:
        """Batch-prefetch only the URIs needed to enrich these records."""
        wanted_by_date: dict[date, set[str]] = defaultdict(set)

        def add_uri(uri: str | None, fallback_date: date) -> None:
            if not uri or uri in self._post_cache:
                return
            inferred_date = uri_to_date(uri)
            if inferred_date:
                wanted_by_date[inferred_date].add(uri)
            if not inferred_date or inferred_date != fallback_date:
                wanted_by_date[fallback_date].add(uri)

        for record in records:
            fallback_date = date_from_record(record)
            meme = record.get("meme_reply") or {}
            original = record.get("original_post") or {}
            parent = record.get("parent_reply") or {}
            add_uri(record.get("uri") or meme.get("uri"), fallback_date)
            add_uri(meme.get("uri"), fallback_date)
            add_uri(meme.get("root_uri") or original.get("uri"), fallback_date)
            add_uri(meme.get("parent_uri") or parent.get("uri"), fallback_date)
            add_uri(original.get("uri"), fallback_date)
            add_uri(parent.get("uri"), fallback_date)
            for node in record.get("ancestor_chain") or []:
                add_uri((node or {}).get("uri"), fallback_date)
            quoted = record.get("quoted_post") or {}
            add_uri(quoted.get("uri"), fallback_date)

        for _ in range(max_hops + 3):
            pending = {
                d: {uri for uri in uris if uri not in self._post_cache}
                for d, uris in wanted_by_date.items()
            }
            pending = {d: uris for d, uris in pending.items() if uris}
            if not pending:
                break

            newly_found: list[tuple[date, dict[str, Any]]] = []
            for d in sorted(pending):
                found = self.scan_date_for_uris(d, pending[d])
                newly_found.extend((d, post) for post in found.values())

            if not newly_found:
                break

            added = False
            for fallback_date, post in newly_found:
                for uri in (post.get("parent_uri"), post.get("root_uri"), post.get("quoted_post_uri")):
                    if uri and uri not in self._post_cache:
                        wanted_by_date[uri_to_date(uri) or fallback_date].add(uri)
                        added = True
            if not added:
                break

    def prefetch_benchmark_quotes(self, records: list[dict[str, Any]]) -> None:
        """Prefetch only quote targets consumed by benchmark_pipeline.py."""
        wanted_by_date: dict[date, set[str]] = defaultdict(set)

        def fallback_date_for(post: dict[str, Any] | None, record: dict[str, Any]) -> date:
            if isinstance(post, dict):
                dt = parse_datetime_safe(post.get("created_at")).date()
                if dt != datetime.min.replace(tzinfo=timezone.utc).date():
                    return dt
            return date_from_record(record)

        def add_quote(post: dict[str, Any] | None, record: dict[str, Any]) -> None:
            uri = quote_uri_from_post(post)
            if not uri or uri in self._post_cache:
                return
            fallback = fallback_date_for(post, record)
            inferred = uri_to_date(uri)
            if inferred:
                wanted_by_date[inferred].add(uri)
            if not inferred or inferred != fallback:
                wanted_by_date[fallback].add(uri)

        for record in records:
            add_quote(record.get("original_post"), record)
            for node in record.get("ancestor_chain") or []:
                if isinstance(node, dict):
                    add_quote(node, record)

        for d in sorted(wanted_by_date):
            pending = {uri for uri in wanted_by_date[d] if uri not in self._post_cache}
            if pending:
                self.scan_date_for_uris(d, pending)


def post_stub(uri: str | None) -> dict[str, Any] | None:
    if not uri:
        return None
    return {
        "uri": uri,
        "uid": uri_to_uid(uri),
        "in_archive": False,
        "did": None,
        "rkey": uri.split("/")[-1] if uri else None,
        "text": None,
        "created_at": None,
        "like_count": None,
        "reply_count": None,
        "has_image": None,
        "images": [],
    }


def post_to_dict(post: dict[str, Any] | None, fallback: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if post is None:
        return fallback
    result = {k: post.get(k) for k in BASE_POST_FIELDS if k in post}
    result["in_archive"] = bool(post.get("in_archive", True))
    result["images"] = [
        {
            "cid": img.get("cid"),
            "alt": img.get("alt", ""),
            "source_url": img.get("source_url") or img.get("url"),
        }
        for img in (post.get("images") or [])
    ]
    if post.get("quoted_post_uri"):
        result["quoted_post_uri"] = post.get("quoted_post_uri")
    result.update(extract_external(post.get("embed")))
    return result


def hydrate_quoted_post(
    source_post: dict[str, Any] | None,
    manager: ArchiveManager,
    current_date: date,
    current_idx: dict[str, Any] | None,
    source_field: str,
) -> dict[str, Any] | None:
    """Hydrate a post quoted/embedded by source_post, if present."""
    if not source_post:
        return None

    quoted_uri = source_post.get("quoted_post_uri") or find_quoted_uri(source_post.get("embed"))
    if not quoted_uri:
        return None

    source_date = parse_datetime_safe(source_post.get("created_at")).date()
    if source_date == datetime.min.replace(tzinfo=timezone.utc).date():
        source_date = current_date

    quoted = context_lookup(manager, quoted_uri, source_date, current_idx=current_idx)
    quoted_dict = post_to_dict(quoted) if quoted else post_stub(quoted_uri)
    if not quoted_dict:
        return None
    quoted_dict["quoted_from_uri"] = source_post.get("uri")
    quoted_dict["quoted_from_field"] = source_field
    return quoted_dict


def post_to_context_dict(
    post: dict[str, Any] | None,
    manager: ArchiveManager,
    current_date: date,
    current_idx: dict[str, Any] | None,
    source_field: str,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Convert a post to dataset schema and attach nested quoted_post if any."""
    result = post_to_dict(post, fallback=fallback)
    if not result:
        return result
    quoted_post = hydrate_quoted_post(post, manager, current_date, current_idx, source_field)
    if quoted_post:
        result["quoted_post"] = quoted_post
    elif "quoted_post" in result and not result["quoted_post"]:
        result.pop("quoted_post", None)
    return result


def context_lookup(
    manager: ArchiveManager,
    uri: str | None,
    current_date: date,
    current_idx: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Lookup a post using either full-day or targeted mode."""
    post = manager.lookup(uri, current_date, current_idx=current_idx)
    if post or manager.allow_full_lookup:
        return post
    return manager.targeted_lookup(uri, current_date)


def same_uri(a: dict[str, Any] | None, uri: str | None) -> bool:
    return bool(a and uri and a.get("uri") == uri)


def build_ancestor_chain(
    meme_post: dict[str, Any],
    record: dict[str, Any],
    manager: ArchiveManager,
    current_date: date,
    current_idx: dict[str, Any] | None,
    include_parent: bool,
    max_hops: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    root_uri = meme_post.get("root_uri") or (record.get("original_post") or {}).get("uri")
    parent_uri = meme_post.get("parent_uri") or (record.get("parent_reply") or {}).get("uri")
    if not parent_uri or parent_uri == root_uri:
        return [], None

    reverse_chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor = parent_uri

    for _ in range(max_hops):
        if not cursor or cursor == root_uri or cursor in seen:
            break
        seen.add(cursor)
        post = context_lookup(manager, cursor, current_date, current_idx=current_idx)
        if post:
            reverse_chain.append(post)
            cursor = post.get("parent_uri")
        else:
            stub = post_stub(cursor)
            if stub:
                reverse_chain.append(stub)
            break

    chain_posts = list(reversed(reverse_chain))
    direct_parent = reverse_chain[0] if reverse_chain else None

    chain_dicts = [
        post_to_context_dict(
            post,
            manager=manager,
            current_date=current_date,
            current_idx=current_idx,
            source_field=f"ancestor_chain[{idx}]",
        )
        for idx, post in enumerate(chain_posts)
    ]
    chain_dicts = [post for post in chain_dicts if post]

    if not include_parent and parent_uri:
        chain_dicts = [post for post in chain_dicts if post.get("uri") != parent_uri]

    parent_dict = (
        post_to_context_dict(
            direct_parent,
            manager=manager,
            current_date=current_date,
            current_idx=current_idx,
            source_field="parent_reply",
        )
        if direct_parent
        else None
    )
    return chain_dicts, parent_dict


def find_first_quoted_post(
    source_posts: list[tuple[str, dict[str, Any] | None]],
    manager: ArchiveManager,
    current_date: date,
    current_idx: dict[str, Any] | None,
) -> dict[str, Any] | None:
    seen: set[str] = set()
    for source_field, source in source_posts:
        if not source:
            continue
        quoted_uri = source.get("quoted_post_uri")
        if not quoted_uri or quoted_uri in seen:
            continue
        seen.add(quoted_uri)
        source_date = parse_datetime_safe(source.get("created_at")).date()
        if source_date == datetime.min.replace(tzinfo=timezone.utc).date():
            source_date = current_date
        quoted = context_lookup(manager, quoted_uri, source_date, current_idx=current_idx)
        quoted_dict = post_to_dict(quoted) if quoted else post_stub(quoted_uri)
        if quoted_dict:
            quoted_dict["quoted_from_uri"] = source.get("uri")
            quoted_dict["quoted_from_field"] = source_field
            return quoted_dict
    return None


def update_record(
    record: dict[str, Any],
    manager: ArchiveManager,
    include_parent_in_chain: bool,
    max_hops: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_date = date_from_record(record)
    current_idx = None

    meme_uri = record.get("uri") or (record.get("meme_reply") or {}).get("uri")
    meme_post = context_lookup(manager, meme_uri, current_date, current_idx=current_idx)
    if not meme_post:
        meme_post = {
            **(record.get("meme_reply") or {}),
            "uri": meme_uri,
            "root_uri": (record.get("meme_reply") or {}).get("root_uri")
            or (record.get("original_post") or {}).get("uri"),
            "parent_uri": (record.get("meme_reply") or {}).get("parent_uri")
            or (record.get("parent_reply") or {}).get("uri"),
        }

    root_uri = meme_post.get("root_uri") or (record.get("original_post") or {}).get("uri")
    root_post = context_lookup(manager, root_uri, current_date, current_idx=current_idx)
    parent_uri = meme_post.get("parent_uri") or (record.get("parent_reply") or {}).get("uri")
    parent_post = context_lookup(manager, parent_uri, current_date, current_idx=current_idx) if parent_uri else None

    ancestor_chain, reconstructed_parent = build_ancestor_chain(
        meme_post=meme_post,
        record=record,
        manager=manager,
        current_date=current_date,
        current_idx=current_idx,
        include_parent=include_parent_in_chain,
        max_hops=max_hops,
    )

    if parent_post:
        parent_reply = post_to_context_dict(
            parent_post,
            manager=manager,
            current_date=current_date,
            current_idx=current_idx,
            source_field="parent_reply",
        )
    elif reconstructed_parent:
        parent_reply = reconstructed_parent
    else:
        parent_reply = record.get("parent_reply")

    # Prefer archive-hydrated posts, but keep existing values if lookup fails.
    updated = dict(record)
    original_post = post_to_context_dict(
        root_post,
        manager=manager,
        current_date=current_date,
        current_idx=current_idx,
        source_field="original_post",
        fallback=record.get("original_post"),
    )
    updated["original_post"] = original_post
    updated["parent_reply"] = parent_reply
    updated["ancestor_chain"] = ancestor_chain

    meme_reply = post_to_context_dict(
        meme_post,
        manager=manager,
        current_date=current_date,
        current_idx=current_idx,
        source_field="meme_reply",
        fallback=record.get("meme_reply"),
    )
    if meme_reply:
        updated["meme_reply"] = meme_reply
        updated["uri"] = updated.get("uri") or meme_reply.get("uri")

    full_chain_len = len(ancestor_chain)
    if parent_reply and not any(same_uri(item, parent_reply.get("uri")) for item in ancestor_chain):
        full_chain_len += 1
    depth = 1 + full_chain_len
    updated["thread_structure"] = {
        **(record.get("thread_structure") or {}),
        "depth": depth,
        "label": {1: "reply", 2: "re-reply", 3: "re-re-reply"}.get(depth, f"depth-{depth}"),
    }

    # Benchmark format: top-level quoted_post represents a quote attached to
    # the original post. Quotes attached to ancestors are nested inside each
    # ancestor_chain node.
    quoted_post = (original_post or {}).get("quoted_post")
    if quoted_post:
        updated["quoted_post"] = quoted_post
    elif "quoted_post" not in updated:
        updated["quoted_post"] = None

    stats = {
        "uid": updated.get("uid"),
        "has_ancestor_chain": bool(updated.get("ancestor_chain")),
        "ancestor_count": len(updated.get("ancestor_chain") or []),
        "has_quoted_post": bool(updated.get("quoted_post")),
    }
    return updated, stats


def load_records_from_file(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records, "jsonl"

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, "json-array"
    if isinstance(data, dict):
        return [data], "json-object"
    raise ValueError(f"Unsupported JSON structure: {path}")


def write_records_to_file(records: list[dict[str, Any]], path: Path, file_kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_kind == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    elif file_kind == "json-object" and len(records) == 1:
        path.write_text(json.dumps(records[0], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    else:
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def resolve_records_dir(path: Path) -> Path:
    if (path / "records").is_dir():
        return path / "records"
    return path


def resolve_output_records_dir(input_path: Path, output_path: Path) -> Path:
    """Mirror dataset-root/records layout when the input is a dataset root."""
    if (input_path / "records").is_dir() and output_path.name != "records":
        return output_path / "records"
    return output_path


def default_output_for(input_path: Path) -> Path:
    """Create a safe context-enriched copy path without overwriting input."""
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}_with_context{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}_with_context")


POST_OBJECT_FIELDS = (
    "original_post",
    "parent_reply",
    "meme_reply",
    "quoted_post",
    "best_reply_before_meme",
    "closest_text_reply",
    "closest_sibling_text_reply",
    "comparison_reply",
)


def normalize_existing_post(post: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize an already-collected post object without archive hydration."""
    if not isinstance(post, dict) or not post.get("uri"):
        return None
    out = dict(post)
    out["uid"] = out.get("uid") or uri_to_uid(out["uri"])
    out["rkey"] = out.get("rkey") or str(out["uri"]).split("/")[-1]
    out["images"] = [
        {
            "cid": img.get("cid"),
            "alt": img.get("alt", ""),
            "source_url": img.get("source_url") or img.get("url"),
            **({"local_path": img.get("local_path")} if img.get("local_path") else {}),
        }
        for img in (out.get("images") or [])
        if isinstance(img, dict)
    ]
    if "in_archive" not in out:
        out["in_archive"] = True
    return out


def post_quality(post: dict[str, Any] | None) -> int:
    if not post:
        return -1
    score = 0
    score += 10 if post.get("text") else 0
    score += 5 if post.get("created_at") else 0
    score += 3 if post.get("images") else 0
    score += 2 if post.get("parent_uri") else 0
    score += 2 if post.get("root_uri") else 0
    score += 1 if post.get("in_archive") is not False else 0
    score += 1 if post.get("quoted_post") else 0
    return score


def merge_existing_posts(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep the richer already-collected post object and fill missing fields."""
    a = normalize_existing_post(a)
    b = normalize_existing_post(b)
    if not a:
        return b
    if not b:
        return a
    rich, other = (a, b) if post_quality(a) >= post_quality(b) else (b, a)
    merged = dict(rich)
    for key, value in other.items():
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
    return normalize_existing_post(merged)


def iter_existing_posts(record: dict[str, Any]):
    """Yield all post-like objects already present in a record."""
    for field in POST_OBJECT_FIELDS:
        post = record.get(field)
        if isinstance(post, dict):
            yield post
            nested_quote = post.get("quoted_post")
            if isinstance(nested_quote, dict):
                yield nested_quote
    for node in record.get("ancestor_chain") or []:
        if isinstance(node, dict):
            yield node
            nested_quote = node.get("quoted_post")
            if isinstance(nested_quote, dict):
                yield nested_quote


def add_record_to_post_cache(record: dict[str, Any], post_cache: dict[str, dict[str, Any]]) -> None:
    for post in iter_existing_posts(record):
        norm = normalize_existing_post(post)
        if not norm:
            continue
        uri = norm["uri"]
        post_cache[uri] = merge_existing_posts(post_cache.get(uri), norm) or norm


def load_json_records_iter(path: Path):
    """Stream JSONL or yield records from JSON/records directory."""
    if path.is_dir():
        records_dir = resolve_records_dir(path)
        for fp in sorted(records_dir.glob("*.json")):
            try:
                yield json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
        return

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield data


def build_existing_post_cache(input_path: Path, context_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Build a URI->post cache from already collected/labeled dataset files."""
    cache: dict[str, dict[str, Any]] = {}
    paths = [input_path] + [p for p in context_dirs if p.exists()]
    for path in paths:
        before = len(cache)
        for record in load_json_records_iter(path):
            add_record_to_post_cache(record, cache)
        print(f"[existing-cache] {path}: +{len(cache) - before:,} posts (total={len(cache):,})", flush=True)
    return cache


def existing_lookup(
    uri: str | None,
    post_cache: dict[str, dict[str, Any]],
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if uri and uri in post_cache:
        return merge_existing_posts(post_cache.get(uri), fallback)
    return normalize_existing_post(fallback)


def attach_existing_quote(post: dict[str, Any] | None, post_cache: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    post = normalize_existing_post(post)
    if not post:
        return None
    if isinstance(post.get("quoted_post"), dict):
        post["quoted_post"] = normalize_existing_post(post["quoted_post"])
        return post
    quoted_uri = post.get("quoted_post_uri")
    if quoted_uri and quoted_uri in post_cache:
        quoted = normalize_existing_post(post_cache[quoted_uri])
        if quoted:
            quoted["quoted_from_uri"] = post.get("uri")
            post["quoted_post"] = quoted
    return post


def build_existing_ancestor_chain(
    meme_post: dict[str, Any],
    record: dict[str, Any],
    post_cache: dict[str, dict[str, Any]],
    include_parent: bool,
    max_hops: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if record.get("ancestor_chain"):
        chain = [
            attach_existing_quote(existing_lookup((node or {}).get("uri"), post_cache, node), post_cache)
            for node in (record.get("ancestor_chain") or [])
            if isinstance(node, dict)
        ]
        chain = [node for node in chain if node]
        parent = existing_lookup((record.get("parent_reply") or {}).get("uri"), post_cache, record.get("parent_reply"))
        return chain, attach_existing_quote(parent, post_cache)

    root_uri = meme_post.get("root_uri") or (record.get("original_post") or {}).get("uri")
    parent_uri = meme_post.get("parent_uri") or (record.get("parent_reply") or {}).get("uri")
    if not parent_uri or parent_uri == root_uri:
        return [], None

    reverse_chain = []
    seen = set()
    cursor = parent_uri
    for _ in range(max_hops):
        if not cursor or cursor == root_uri or cursor in seen:
            break
        seen.add(cursor)
        post = existing_lookup(cursor, post_cache, record.get("parent_reply") if cursor == parent_uri else None)
        if not post:
            break
        reverse_chain.append(post)
        cursor = post.get("parent_uri")

    direct_parent = reverse_chain[0] if reverse_chain else existing_lookup(parent_uri, post_cache, record.get("parent_reply"))
    chain = list(reversed(reverse_chain))
    if not include_parent:
        chain = [node for node in chain if node.get("uri") != parent_uri]
    chain = [attach_existing_quote(node, post_cache) for node in chain]
    return [node for node in chain if node], attach_existing_quote(direct_parent, post_cache)


def update_record_existing_only(
    record: dict[str, Any],
    post_cache: dict[str, dict[str, Any]],
    include_parent_in_chain: bool,
    max_hops: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = dict(record)
    meme_fallback = record.get("meme_reply") or {}
    meme_uri = record.get("uri") or meme_fallback.get("uri")
    meme_post = existing_lookup(meme_uri, post_cache, meme_fallback) or meme_fallback

    original_uri = meme_post.get("root_uri") or (record.get("original_post") or {}).get("uri")
    parent_uri = meme_post.get("parent_uri") or (record.get("parent_reply") or {}).get("uri")

    original_post = attach_existing_quote(
        existing_lookup(original_uri, post_cache, record.get("original_post")),
        post_cache,
    )
    parent_reply = attach_existing_quote(
        existing_lookup(parent_uri, post_cache, record.get("parent_reply")),
        post_cache,
    )
    meme_reply = attach_existing_quote(meme_post, post_cache)
    ancestor_chain, reconstructed_parent = build_existing_ancestor_chain(
        meme_post=meme_reply or meme_post,
        record=record,
        post_cache=post_cache,
        include_parent=include_parent_in_chain,
        max_hops=max_hops,
    )
    if not parent_reply and reconstructed_parent:
        parent_reply = reconstructed_parent

    updated["original_post"] = original_post
    updated["parent_reply"] = parent_reply
    updated["meme_reply"] = meme_reply
    updated["ancestor_chain"] = ancestor_chain
    if meme_reply:
        updated["uri"] = updated.get("uri") or meme_reply.get("uri")

    quoted_post = (original_post or {}).get("quoted_post") or record.get("quoted_post")
    updated["quoted_post"] = normalize_existing_post(quoted_post) if isinstance(quoted_post, dict) else None

    full_chain_len = len(ancestor_chain)
    if parent_reply and not any(same_uri(item, parent_reply.get("uri")) for item in ancestor_chain):
        full_chain_len += 1
    depth = 1 + full_chain_len
    updated["thread_structure"] = {
        **(record.get("thread_structure") or {}),
        "depth": depth,
        "label": {1: "reply", 2: "re-reply", 3: "re-re-reply"}.get(depth, f"depth-{depth}"),
    }

    stats = {
        "uid": updated.get("uid"),
        "has_ancestor_chain": bool(updated.get("ancestor_chain")),
        "ancestor_count": len(updated.get("ancestor_chain") or []),
        "has_quoted_post": bool(updated.get("quoted_post")),
    }
    return updated, stats


def has_hydrated_quote(post: dict[str, Any] | None) -> bool:
    quoted = (post or {}).get("quoted_post")
    return isinstance(quoted, dict) and bool(quoted.get("text") or quoted.get("images") or quoted.get("external_title"))


def has_quote_signal(post: dict[str, Any] | None) -> bool:
    if not isinstance(post, dict):
        return False
    return bool(quote_uri_from_post(post) or post.get("quoted_post"))


def quote_uri_from_post(post: dict[str, Any] | None) -> str | None:
    if not isinstance(post, dict):
        return None
    if post.get("quoted_post_uri"):
        return post.get("quoted_post_uri")
    quoted = post.get("quoted_post")
    if isinstance(quoted, dict) and quoted.get("uri"):
        return quoted.get("uri")
    return find_quoted_uri(post.get("embed"))


def record_needs_quote_hydration(record: dict[str, Any]) -> bool:
    # Match benchmark_pipeline.py: it reads top-level quoted_post for
    # original_post quotes and node["quoted_post"] for ancestor_chain quotes.
    relevant_posts = [record.get("original_post")]
    relevant_posts.extend(node for node in (record.get("ancestor_chain") or []) if isinstance(node, dict))
    for post in relevant_posts:
        if has_quote_signal(post) and not has_hydrated_quote(post):
            return True
    return False


def record_needs_parent_chain(record: dict[str, Any]) -> bool:
    """Return True only when existing fields indicate a missing upper reply chain."""
    meme = record.get("meme_reply") or {}
    original = record.get("original_post") or {}
    parent = record.get("parent_reply") or {}
    root_uri = meme.get("root_uri") or original.get("uri")
    parent_uri = meme.get("parent_uri") or parent.get("uri")

    if not parent_uri or (root_uri and parent_uri == root_uri):
        return False

    # If the direct parent itself is missing, archive lookup is useful.
    if not parent:
        return True

    # Benchmark keeps the direct parent separately. We only need ancestor_chain
    # when there is an upper reply between root and direct parent.
    parent_parent_uri = parent.get("parent_uri")
    if parent_parent_uri and root_uri and parent_parent_uri != root_uri:
        return not bool(record.get("ancestor_chain"))

    return False


def record_context_reasons(record: dict[str, Any]) -> list[str]:
    reasons = []
    if record_needs_parent_chain(record):
        reasons.append("parent_chain")
    if record_needs_quote_hydration(record):
        reasons.append("quoted_post")
    return reasons


def update_record_auto(
    record: dict[str, Any],
    manager: ArchiveManager,
    include_parent_in_chain: bool,
    max_hops: int,
    selective_archive: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use archive only for records that actually need missing context."""
    reasons = record_context_reasons(record) if selective_archive else ["all"]
    if "parent_chain" in reasons or not selective_archive:
        updated, stats = update_record(record, manager, include_parent_in_chain, max_hops)
        stats["archive_used"] = True
        stats["archive_reasons"] = reasons
        return updated, stats

    local_cache: dict[str, dict[str, Any]] = {}
    add_record_to_post_cache(record, local_cache)
    updated, stats = update_record_existing_only(record, local_cache, include_parent_in_chain, max_hops)
    if "quoted_post" in reasons:
        updated = hydrate_benchmark_quotes_only(updated, manager)
        stats["has_quoted_post"] = bool(updated.get("quoted_post"))
        stats["archive_used"] = True
        stats["archive_reasons"] = reasons
    else:
        stats["archive_used"] = False
        stats["archive_reasons"] = []
    return updated, stats


def hydrate_quote_for_post(
    post: dict[str, Any] | None,
    manager: ArchiveManager,
    source_field: str,
) -> dict[str, Any] | None:
    post = normalize_existing_post(post)
    if not post or has_hydrated_quote(post):
        return post
    quoted_uri = quote_uri_from_post(post)
    if not quoted_uri:
        return post
    fallback_date = parse_datetime_safe(post.get("created_at")).date()
    if fallback_date == datetime.min.replace(tzinfo=timezone.utc).date():
        fallback_date = uri_to_date(post.get("uri")) or date.today()
    quoted = context_lookup(manager, quoted_uri, fallback_date, current_idx=None)
    quoted_dict = post_to_dict(quoted) if quoted else post_stub(quoted_uri)
    if quoted_dict:
        quoted_dict["quoted_from_uri"] = post.get("uri")
        quoted_dict["quoted_from_field"] = source_field
        post["quoted_post"] = quoted_dict
    return post


def hydrate_benchmark_quotes_only(record: dict[str, Any], manager: ArchiveManager) -> dict[str, Any]:
    """Hydrate only the quote fields actually consumed by benchmark_pipeline.py."""
    updated = dict(record)
    original_post = hydrate_quote_for_post(updated.get("original_post"), manager, "original_post")
    updated["original_post"] = original_post
    if original_post and original_post.get("quoted_post"):
        updated["quoted_post"] = original_post["quoted_post"]

    new_chain = []
    for i, node in enumerate(updated.get("ancestor_chain") or []):
        if isinstance(node, dict):
            new_chain.append(hydrate_quote_for_post(node, manager, f"ancestor_chain[{i}]") or node)
    updated["ancestor_chain"] = new_chain
    return updated


def update_directory(
    input_dir: Path,
    output_dir: Path,
    manager: ArchiveManager,
    include_parent_in_chain: bool,
    max_hops: int,
    dry_run: bool,
    progress_every: int,
    selective_archive: bool,
) -> dict[str, int]:
    records_dir = resolve_records_dir(input_dir)
    files = sorted(records_dir.glob("*.json"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"total": 0, "ancestor": 0, "quoted": 0, "errors": 0, "archive_used": 0, "archive_skipped": 0}
    start = time.time()
    for i, path in enumerate(files, start=1):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            updated, stats = update_record_auto(record, manager, include_parent_in_chain, max_hops, selective_archive)
            summary["total"] += 1
            summary["ancestor"] += int(stats["has_ancestor_chain"])
            summary["quoted"] += int(stats["has_quoted_post"])
            summary["archive_used"] += int(stats.get("archive_used", False))
            summary["archive_skipped"] += int(not stats.get("archive_used", False))
            if not dry_run:
                (output_dir / path.name).write_text(
                    json.dumps(updated, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
        except Exception as exc:
            summary["errors"] += 1
            print(f"[ERROR] {path}: {exc}")

        if progress_every and i % progress_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            rate = i / elapsed
            print(f"[progress] {i}/{len(files)} records ({rate:.2f} rec/s)", flush=True)

    return summary


def update_file(
    input_file: Path,
    output_file: Path,
    manager: ArchiveManager,
    include_parent_in_chain: bool,
    max_hops: int,
    dry_run: bool,
    progress_every: int,
    flush_every: int,
    batch_size: int,
    fast_targeted: bool,
    selective_archive: bool,
) -> dict[str, int]:
    summary = {"total": 0, "ancestor": 0, "quoted": 0, "errors": 0, "archive_used": 0, "archive_skipped": 0}
    start = time.time()

    if input_file.suffix == ".jsonl":
        output_handle = None
        try:
            if not dry_run:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_handle = output_file.open("w", encoding="utf-8")
                output_handle.flush()
                print(f"[stream] writing JSONL incrementally to {output_file}", flush=True)

            def process_batch(batch: list[tuple[int, dict[str, Any]]]) -> None:
                if not batch:
                    return
                if fast_targeted:
                    if selective_archive:
                        parent_chain_records = []
                        quote_records = []
                        for _, record in batch:
                            reasons = record_context_reasons(record)
                            if "parent_chain" in reasons:
                                parent_chain_records.append(record)
                            if "quoted_post" in reasons:
                                quote_records.append(record)
                        if parent_chain_records:
                            manager.prefetch_records(parent_chain_records, max_hops=max_hops)
                        if quote_records:
                            manager.prefetch_benchmark_quotes(quote_records)
                    else:
                        manager.prefetch_records([record for _, record in batch], max_hops=max_hops)
                for i, record in batch:
                    nonlocal summary
                    try:
                        updated, stats = update_record_auto(
                            record, manager, include_parent_in_chain, max_hops, selective_archive
                        )
                        summary["total"] += 1
                        summary["ancestor"] += int(stats["has_ancestor_chain"])
                        summary["quoted"] += int(stats["has_quoted_post"])
                        summary["archive_used"] += int(stats.get("archive_used", False))
                        summary["archive_skipped"] += int(not stats.get("archive_used", False))
                    except Exception as exc:
                        summary["errors"] += 1
                        updated = record
                        print(f"[ERROR] record {i}: {exc}", flush=True)

                    if output_handle:
                        output_handle.write(json.dumps(updated, ensure_ascii=False, default=str) + "\n")
                        if flush_every and i % flush_every == 0:
                            output_handle.flush()

                    if progress_every and i % progress_every == 0:
                        elapsed = max(time.time() - start, 1e-6)
                        rate = i / elapsed
                        print(
                            f"[progress] {i} lines | processed={summary['total']} "
                            f"ancestors={summary['ancestor']} quoted={summary['quoted']} "
                            f"archive_used={summary['archive_used']} skipped={summary['archive_skipped']} "
                            f"errors={summary['errors']} ({rate:.2f} lines/s)",
                            flush=True,
                        )

            batch: list[tuple[int, dict[str, Any]]] = []
            with input_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception as exc:
                        summary["errors"] += 1
                        print(f"[ERROR] JSON parse failed at line {i}: {exc}", flush=True)
                        continue

                    batch.append((i, record))
                    if len(batch) >= batch_size:
                        process_batch(batch)
                        batch = []
                process_batch(batch)
        finally:
            if output_handle:
                output_handle.flush()
                output_handle.close()
        return summary

    records, file_kind = load_records_from_file(input_file)
    updated_records = []

    for i, record in enumerate(records, start=1):
        try:
            updated, stats = update_record_auto(record, manager, include_parent_in_chain, max_hops, selective_archive)
            updated_records.append(updated)
            summary["total"] += 1
            summary["ancestor"] += int(stats["has_ancestor_chain"])
            summary["quoted"] += int(stats["has_quoted_post"])
            summary["archive_used"] += int(stats.get("archive_used", False))
            summary["archive_skipped"] += int(not stats.get("archive_used", False))
        except Exception as exc:
            summary["errors"] += 1
            updated_records.append(record)
            print(f"[ERROR] record {i}: {exc}")

        if progress_every and i % progress_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            rate = i / elapsed
            print(f"[progress] {i}/{len(records)} records ({rate:.2f} rec/s)", flush=True)

    if not dry_run:
        write_records_to_file(updated_records, output_file, file_kind)

    return summary


def update_file_existing_only(
    input_file: Path,
    output_file: Path,
    post_cache: dict[str, dict[str, Any]],
    include_parent_in_chain: bool,
    max_hops: int,
    dry_run: bool,
    progress_every: int,
    flush_every: int,
) -> dict[str, int]:
    """Update JSON/JSONL using only already-collected post objects."""
    summary = {"total": 0, "ancestor": 0, "quoted": 0, "errors": 0}
    start = time.time()

    if input_file.suffix == ".jsonl":
        output_handle = None
        try:
            if not dry_run:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_handle = output_file.open("w", encoding="utf-8")
                output_handle.flush()
                print(f"[stream] writing JSONL incrementally to {output_file}", flush=True)

            with input_file.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        updated, stats = update_record_existing_only(
                            record, post_cache, include_parent_in_chain, max_hops
                        )
                        summary["total"] += 1
                        summary["ancestor"] += int(stats["has_ancestor_chain"])
                        summary["quoted"] += int(stats["has_quoted_post"])
                    except Exception as exc:
                        summary["errors"] += 1
                        updated = {}
                        print(f"[ERROR] record {i}: {exc}", flush=True)

                    if output_handle:
                        output_handle.write(json.dumps(updated, ensure_ascii=False, default=str) + "\n")
                        if flush_every and i % flush_every == 0:
                            output_handle.flush()

                    if progress_every and i % progress_every == 0:
                        elapsed = max(time.time() - start, 1e-6)
                        print(
                            f"[progress] {i} lines | processed={summary['total']} "
                            f"ancestors={summary['ancestor']} quoted={summary['quoted']} "
                            f"errors={summary['errors']} ({i / elapsed:.2f} lines/s)",
                            flush=True,
                        )
        finally:
            if output_handle:
                output_handle.flush()
                output_handle.close()
        return summary

    records, file_kind = load_records_from_file(input_file)
    updated_records = []
    for i, record in enumerate(records, start=1):
        try:
            updated, stats = update_record_existing_only(record, post_cache, include_parent_in_chain, max_hops)
            summary["total"] += 1
            summary["ancestor"] += int(stats["has_ancestor_chain"])
            summary["quoted"] += int(stats["has_quoted_post"])
        except Exception as exc:
            summary["errors"] += 1
            updated = record
            print(f"[ERROR] record {i}: {exc}", flush=True)
        updated_records.append(updated)
        if progress_every and i % progress_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(f"[progress] {i}/{len(records)} records ({i / elapsed:.2f} rec/s)", flush=True)

    if not dry_run:
        write_records_to_file(updated_records, output_file, file_kind)
    return summary


def update_directory_existing_only(
    input_dir: Path,
    output_dir: Path,
    post_cache: dict[str, dict[str, Any]],
    include_parent_in_chain: bool,
    max_hops: int,
    dry_run: bool,
    progress_every: int,
) -> dict[str, int]:
    records_dir = resolve_records_dir(input_dir)
    files = sorted(records_dir.glob("*.json"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"total": 0, "ancestor": 0, "quoted": 0, "errors": 0}
    start = time.time()

    for i, path in enumerate(files, start=1):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            updated, stats = update_record_existing_only(record, post_cache, include_parent_in_chain, max_hops)
            summary["total"] += 1
            summary["ancestor"] += int(stats["has_ancestor_chain"])
            summary["quoted"] += int(stats["has_quoted_post"])
            if not dry_run:
                (output_dir / path.name).write_text(
                    json.dumps(updated, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
        except Exception as exc:
            summary["errors"] += 1
            print(f"[ERROR] {path}: {exc}", flush=True)

        if progress_every and i % progress_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(f"[progress] {i}/{len(files)} records ({i / elapsed:.2f} rec/s)", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add ancestor_chain and quoted_post to meme dataset records.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Input records directory, JSONL file, or JSON file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--archive-base",
        default=str(DEFAULT_ARCHIVE_BASE),
        help=f"Root directory of Bluesky firehose archives. Default: {DEFAULT_ARCHIVE_BASE}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the context-enriched copy. Default: input path with _with_context suffix.",
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Do not scan firehose archives; use only posts already present in labeled/collected dataset records.",
    )
    parser.add_argument(
        "--context-dir",
        action="append",
        default=None,
        help="Additional collected dataset directory/file to use as an existing post cache. Can be repeated.",
    )
    parser.add_argument(
        "--print-defaults",
        action="store_true",
        help="Print default paths mirrored from the existing pipelines and exit.",
    )
    parser.add_argument("--cache-size", type=int, default=30, help="Number of archive days to keep in memory.")
    parser.add_argument("--max-days-back", type=int, default=0, help="Fallback backward search window for lookup.")
    parser.add_argument("--max-hops", type=int, default=20, help="Maximum parent-chain hops to follow.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N input records/lines. Set 0 to disable.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="For JSONL output, flush to disk every N input lines. Set 1 for maximum visibility.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="For JSONL input, prefetch archive context in batches of N records.",
    )
    parser.add_argument(
        "--full-day-index",
        action="store_true",
        help="Use the old full-day archive index mode instead of targeted URI scans.",
    )
    parser.add_argument(
        "--all-records-archive",
        action="store_true",
        help="Use archive lookup for every record. By default, archive is used only for records missing parent-chain or quote context.",
    )
    parser.add_argument(
        "--include-parent-in-ancestor-chain",
        dest="include_parent_in_ancestor_chain",
        action="store_true",
        default=False,
        help="Include the direct parent_reply in ancestor_chain. By default, parent_reply is stored separately.",
    )
    parser.add_argument(
        "--exclude-parent-from-ancestor-chain",
        dest="include_parent_in_ancestor_chain",
        action="store_false",
        help="Exclude the direct parent_reply from ancestor_chain. This is the default.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run lookup and print summary without writing files.")
    parser.add_argument(
        "--quiet-archive-loads",
        action="store_true",
        help="Do not print archive-day loading messages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_defaults:
        print("[defaults]")
        print(f"  input:        {DEFAULT_INPUT}")
        print(f"  output:       {default_output_for(DEFAULT_INPUT)}")
        print(f"  archive_base: {DEFAULT_ARCHIVE_BASE}")
        print("  existing_only command:")
        print("    python update_dataset_context_fields.py --existing-only")
        print("  collection datasets:")
        for path in DEFAULT_COLLECTION_DATASETS:
            print(f"    - {path}")
        return

    input_path = Path(args.input)
    archive_base = Path(args.archive_base)

    if not args.existing_only and not archive_base.exists():
        raise FileNotFoundError(f"Archive base does not exist: {archive_base}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    output_path = Path(args.output) if args.output else default_output_for(input_path)
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path must differ from input path; this script never overwrites the input dataset.")
    print(f"[input] {input_path}", flush=True)
    print(f"[output] {output_path if not args.dry_run else '(dry run)'}", flush=True)
    print(f"[ancestor_chain] include_parent={args.include_parent_in_ancestor_chain}", flush=True)

    if args.existing_only:
        context_dirs = [Path(p) for p in (args.context_dir or [])] or DEFAULT_COLLECTION_DATASETS
        print("[mode] existing-only (no firehose archive scan)", flush=True)
        print("[context-cache] sources:", flush=True)
        for path in [input_path] + context_dirs:
            print(f"  - {path}", flush=True)
        post_cache = build_existing_post_cache(input_path, context_dirs)
        if input_path.is_dir():
            summary = update_directory_existing_only(
                input_dir=input_path,
                output_dir=resolve_output_records_dir(input_path, output_path),
                post_cache=post_cache,
                include_parent_in_chain=args.include_parent_in_ancestor_chain,
                max_hops=args.max_hops,
                dry_run=args.dry_run,
                progress_every=args.progress_every,
            )
        else:
            summary = update_file_existing_only(
                input_file=input_path,
                output_file=output_path,
                post_cache=post_cache,
                include_parent_in_chain=args.include_parent_in_ancestor_chain,
                max_hops=args.max_hops,
                dry_run=args.dry_run,
                progress_every=args.progress_every,
                flush_every=args.flush_every,
            )
        print("\n[summary]", flush=True)
        print(f"  records processed:       {summary['total']}", flush=True)
        print(f"  records with ancestors:  {summary['ancestor']}", flush=True)
        print(f"  records with quote:      {summary['quoted']}", flush=True)
        print(f"  archive used:            {summary.get('archive_used', 0)}", flush=True)
        print(f"  archive skipped:         {summary.get('archive_skipped', 0)}", flush=True)
        print(f"  errors:                  {summary['errors']}", flush=True)
        return

    manager = ArchiveManager(
        archive_base,
        cache_size=args.cache_size,
        max_days_back=args.max_days_back,
        log_loads=not args.quiet_archive_loads,
        allow_full_lookup=args.full_day_index,
    )

    print(f"[archive] {archive_base}", flush=True)
    print(
        f"[speed] cache_size={args.cache_size} max_days_back={args.max_days_back} "
        f"post_only_archive_scan=True targeted_uri_scan={not args.full_day_index} "
        f"batch_size={args.batch_size} selective_archive={not args.all_records_archive}",
        flush=True,
    )

    if input_path.is_dir():
        summary = update_directory(
            input_dir=input_path,
            output_dir=resolve_output_records_dir(input_path, output_path),
            manager=manager,
            include_parent_in_chain=args.include_parent_in_ancestor_chain,
            max_hops=args.max_hops,
            dry_run=args.dry_run,
            progress_every=args.progress_every,
            selective_archive=not args.all_records_archive,
        )
    else:
        summary = update_file(
            input_file=input_path,
            output_file=output_path,
            manager=manager,
            include_parent_in_chain=args.include_parent_in_ancestor_chain,
            max_hops=args.max_hops,
            dry_run=args.dry_run,
            progress_every=args.progress_every,
            flush_every=args.flush_every,
            batch_size=args.batch_size,
            fast_targeted=not args.full_day_index,
            selective_archive=not args.all_records_archive,
        )

    print("\n[summary]", flush=True)
    print(f"  records processed:       {summary['total']}", flush=True)
    print(f"  records with ancestors:  {summary['ancestor']}", flush=True)
    print(f"  records with quote:      {summary['quoted']}", flush=True)
    print(f"  archive used:            {summary.get('archive_used', 0)}", flush=True)
    print(f"  archive skipped:         {summary.get('archive_skipped', 0)}", flush=True)
    print(f"  errors:                  {summary['errors']}", flush=True)


if __name__ == "__main__":
    main()
