#!/usr/bin/env python3
"""
Hydrate conversational meme records from a UID/URI manifest.

This script is the public data-collection step for users who should reproduce
the released dataset by UID instead of re-running the classifier. It reads the
manifest exported by 01_collection/export_uid_manifest.py, calls the public
Bluesky getPostThread API, and writes one hydrated JSON record per UID.

Sample dry run:

  python hydrate_from_uid_manifest.py \
    --manifest sample_uid_manifest.jsonl \
    --out sample_hydrated \
    --dry-run

Small real run:

  python hydrate_from_uid_manifest.py \
    --manifest data/collection_pool_uid_manifest.jsonl \
    --out hydrated_records \
    --limit 10 \
    --download-images meme
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread"
USER_AGENT = "ConversationalMemeHydrator/1.0 (academic reproducibility)"
HYDRATED_FIELDS = (
    "uid",
    "uri",
    "original_post",
    "parent_reply",
    "quoted_post",
    "meme_reply",
    "thread_structure",
    "best_reply_before_meme",
    "closest_text_reply",
    "closest_sibling_text_reply",
    "comparison_reply",
)
MANIFEST_HYDRATION_FIELDS = {
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
}
DEFAULT_CARRY_EXCLUDE_FIELDS = {"ancestor_chain", "discourse_labels"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def uri_to_uid(uri: str) -> str:
    try:
        parts = uri.replace("at://", "").split("/")
        did_suffix = parts[0].split(":")[-1][-14:]
        rkey = parts[-1]
        return f"bsky_{did_suffix}_{rkey}"
    except Exception:
        return "bsky_" + hashlib.sha256(uri.encode("utf-8")).hexdigest()[:22]


def did_from_uri(uri: str | None) -> str | None:
    if not uri or not uri.startswith("at://"):
        return None
    try:
        return uri.replace("at://", "").split("/")[0]
    except Exception:
        return None


def clean_uri(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("at://"):
        return value
    return None


def ref_uri(ref: Any) -> str | None:
    if isinstance(ref, dict):
        return clean_uri(ref.get("uri"))
    return clean_uri(ref)


def nested_post_uri(entry: dict[str, Any], field: str) -> str | None:
    value = entry.get(field)
    if isinstance(value, dict):
        return clean_uri(value.get("uri"))
    return None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def count_value(post: dict[str, Any], key: str) -> int | None:
    value = post.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def cid_from_url(url: str | None) -> str | None:
    if not url:
        return None
    clean = url.split("?", 1)[0].rstrip("/")
    match = re.search(r"/([^/@?#]+)@(?:jpeg|jpg|png|webp)(?:$|[?#])?", clean)
    if match:
        return match.group(1)
    if "cdn.bsky.app/img/" in clean:
        tail = clean.rsplit("/", 1)[-1]
        if tail and tail not in {"jpeg", "jpg", "png", "webp"}:
            return tail.split("@", 1)[0]
    return None


def normalize_image_url(url: str | None) -> str | None:
    if not url:
        return None
    clean = url.split("?", 1)[0].replace("/feed_thumbnail/", "/feed_fullsize/")
    cid = cid_from_url(clean)
    if cid and "cdn.bsky.app/img/" in clean and "@" not in clean.rsplit("/", 1)[-1]:
        return f"{clean.rstrip('/')}@jpeg"
    return clean


def image_dedupe_key(image: dict[str, Any]) -> tuple[str, str] | None:
    cid = image.get("cid") or cid_from_url(image.get("source_url")) or cid_from_url(image.get("thumb_url"))
    if cid:
        return ("cid", str(cid))
    url = normalize_image_url(image.get("source_url") or image.get("thumb_url"))
    if url:
        return ("url", url)
    return None


def image_url_from_cid(did: str | None, cid: str | None) -> str | None:
    if did and cid:
        return f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg"
    return None


def merge_images(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        for img in group:
            key = image_dedupe_key(img)
            if key is None:
                out.append(img)
                continue
            if key in seen:
                existing = seen[key]
                for field in ("cid", "alt", "source_url", "thumb_url"):
                    if not existing.get(field) and img.get(field):
                        existing[field] = img[field]
                continue
            seen[key] = img
            out.append(img)
    return out


def extract_images_from_embed(embed: Any, did: str | None) -> list[dict[str, Any]]:
    if not isinstance(embed, dict):
        return []

    embed_type = embed.get("$type", "")
    if "recordWithMedia" in embed_type:
        return extract_images_from_embed(embed.get("media"), did)

    if "images" not in embed_type and "images" not in embed:
        return []

    images = []
    for item in embed.get("images") or []:
        if not isinstance(item, dict):
            continue

        alt = item.get("alt", "") or ""
        source_url = item.get("fullsize") or item.get("thumb")
        thumb_url = item.get("thumb")
        cid = cid_from_url(source_url) or cid_from_url(thumb_url)

        blob = item.get("image")
        if isinstance(blob, dict):
            ref = blob.get("ref")
            if isinstance(ref, dict):
                cid = cid or ref.get("$link")
            elif isinstance(ref, str):
                cid = cid or ref
            cid = cid or blob.get("cid")

        source_url = normalize_image_url(source_url) or image_url_from_cid(did, cid)
        thumb_url = normalize_image_url(thumb_url)
        if cid or source_url:
            images.append(
                {
                    "cid": cid,
                    "alt": alt,
                    "source_url": source_url,
                    "thumb_url": thumb_url,
                }
            )
    return images


def extract_external_from_embed(embed: Any) -> dict[str, Any]:
    if not isinstance(embed, dict):
        return {}

    embed_type = embed.get("$type", "")
    if "recordWithMedia" in embed_type:
        return extract_external_from_embed(embed.get("media"))

    if "external" not in embed_type and "external" not in embed:
        return {}

    external = as_dict(embed.get("external"))
    if not external:
        return {}

    return {
        "external_title": external.get("title"),
        "external_url": external.get("uri"),
        "external_description": external.get("description"),
    }


def stub_post(uri: str | None, reason: str = "unavailable") -> dict[str, Any] | None:
    if not uri:
        return None
    return {
        "uid": uri_to_uid(uri),
        "uri": uri,
        "did": did_from_uri(uri),
        "rkey": uri.split("/")[-1],
        "text": None,
        "langs": [],
        "created_at": None,
        "indexed_at": None,
        "like_count": None,
        "reply_count": None,
        "repost_count": None,
        "quote_count": None,
        "is_reply": None,
        "is_re_reply": None,
        "parent_uri": None,
        "root_uri": None,
        "images": [],
        "has_image": None,
        "in_archive": False,
        "unavailable_reason": reason,
    }


def quoted_record_to_dict(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None

    uri = clean_uri(record.get("uri"))
    if not uri:
        return None

    record_type = record.get("$type", "")
    if "notFoundPost" in record_type:
        return stub_post(uri, "quoted_post_not_found")
    if "blockedPost" in record_type:
        return stub_post(uri, "quoted_post_blocked")

    value = as_dict(record.get("value")) or as_dict(record.get("record"))
    author = as_dict(record.get("author"))
    did = author.get("did") or did_from_uri(uri)

    view_embed = None
    embeds = record.get("embeds")
    if isinstance(embeds, list) and embeds:
        view_embed = embeds[0]

    raw_embed = value.get("embed")
    images = merge_images(
        extract_images_from_embed(raw_embed, did),
        extract_images_from_embed(view_embed, did),
    )

    out = {
        "uid": uri_to_uid(uri),
        "uri": uri,
        "did": did,
        "handle": author.get("handle"),
        "display_name": author.get("displayName"),
        "rkey": uri.split("/")[-1],
        "text": value.get("text", ""),
        "langs": value.get("langs") or [],
        "created_at": value.get("createdAt"),
        "images": images,
        "has_image": bool(images),
        "in_archive": True,
    }
    out.update(extract_external_from_embed(raw_embed))
    out.update({k: v for k, v in extract_external_from_embed(view_embed).items() if v})
    return out


def extract_quoted_from_embed(embed: Any) -> dict[str, Any] | None:
    if not isinstance(embed, dict):
        return None

    embed_type = embed.get("$type", "")
    if "recordWithMedia" in embed_type:
        return extract_quoted_from_embed(embed.get("record"))

    if "record" in embed_type:
        return quoted_record_to_dict(embed.get("record"))

    return None


def post_view_to_dict(post: Any, include_quoted: bool = True) -> dict[str, Any] | None:
    if not isinstance(post, dict):
        return None

    uri = clean_uri(post.get("uri"))
    if not uri:
        return None

    record = as_dict(post.get("record"))
    author = as_dict(post.get("author"))
    did = author.get("did") or did_from_uri(uri)
    raw_embed = record.get("embed")
    view_embed = post.get("embed")
    reply_ref = as_dict(record.get("reply"))
    root_uri = ref_uri(reply_ref.get("root"))
    parent_uri = ref_uri(reply_ref.get("parent"))

    images = merge_images(
        extract_images_from_embed(raw_embed, did),
        extract_images_from_embed(view_embed, did),
    )

    out = {
        "uid": uri_to_uid(uri),
        "uri": uri,
        "did": did,
        "handle": author.get("handle"),
        "display_name": author.get("displayName"),
        "rkey": uri.split("/")[-1],
        "text": record.get("text", ""),
        "langs": record.get("langs") or [],
        "created_at": record.get("createdAt"),
        "indexed_at": post.get("indexedAt"),
        "like_count": count_value(post, "likeCount"),
        "reply_count": count_value(post, "replyCount"),
        "repost_count": count_value(post, "repostCount"),
        "quote_count": count_value(post, "quoteCount"),
        "is_reply": bool(parent_uri),
        "is_re_reply": bool(parent_uri and root_uri and parent_uri != root_uri),
        "parent_uri": parent_uri,
        "root_uri": root_uri,
        "images": images,
        "has_image": bool(images),
        "in_archive": True,
    }
    out.update(extract_external_from_embed(raw_embed))
    out.update({k: v for k, v in extract_external_from_embed(view_embed).items() if v})

    if include_quoted:
        quoted = extract_quoted_from_embed(view_embed) or extract_quoted_from_embed(raw_embed)
        if quoted:
            out["quoted_post"] = quoted

    return out


def is_thread_view(node: Any) -> bool:
    return isinstance(node, dict) and isinstance(node.get("post"), dict)


def collect_parent_views(thread: dict[str, Any]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    node = thread.get("parent")
    seen: set[str] = set()

    while is_thread_view(node):
        post = node["post"]
        uri = clean_uri(post.get("uri"))
        if uri in seen:
            break
        if uri:
            seen.add(uri)
        views.append(post)
        node = node.get("parent")

    return views


class BskyClient:
    def __init__(self, timeout: int, retries: int, sleep: float):
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.thread_cache: dict[str, dict[str, Any] | None] = {}
        self.post_cache: dict[str, dict[str, Any] | None] = {}

    def fetch_json(self, url: str) -> dict[str, Any] | None:
        last_error = None
        for attempt in range(self.retries):
            try:
                req = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
                with urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code in {400, 401, 403, 404}:
                    break
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = str(exc)
            if attempt < self.retries - 1:
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
        return {"_error": last_error or "unknown_error"}

    def get_post_thread(self, uri: str, depth: int = 0, parent_height: int = 10) -> dict[str, Any] | None:
        cache_key = f"{uri}|{depth}|{parent_height}"
        if cache_key in self.thread_cache:
            return self.thread_cache[cache_key]

        params = urlencode({"uri": uri, "depth": depth, "parentHeight": parent_height})
        data = self.fetch_json(f"{API_BASE}?{params}")
        if self.sleep:
            time.sleep(self.sleep)

        if data and data.get("_error"):
            self.thread_cache[cache_key] = data
        else:
            self.thread_cache[cache_key] = data
        return self.thread_cache[cache_key]

    def get_post_view(self, uri: str) -> dict[str, Any] | None:
        if uri in self.post_cache:
            return self.post_cache[uri]

        data = self.get_post_thread(uri, depth=0, parent_height=0)
        thread = as_dict((data or {}).get("thread"))
        post = thread.get("post") if is_thread_view(thread) else None
        self.post_cache[uri] = post
        return post

    def download_bytes(self, url: str) -> bytes | None:
        last_error = None
        for attempt in range(self.retries):
            try:
                req = Request(url, headers={"User-Agent": USER_AGENT})
                with urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code == 404:
                    break
            except (URLError, TimeoutError) as exc:
                last_error = str(exc)
            if attempt < self.retries - 1:
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
        return None


def fetch_post_dict(client: BskyClient, uri: str | None) -> dict[str, Any] | None:
    if not uri:
        return None
    post = client.get_post_view(uri)
    if not post:
        return stub_post(uri)
    return post_view_to_dict(post)


def fetch_auxiliary_reply(
    client: BskyClient,
    uri: str | None,
    selected_by: str | None = None,
) -> dict[str, Any] | None:
    reply = fetch_post_dict(client, uri)
    if reply and selected_by:
        reply["selected_by"] = selected_by
    return reply


def build_record(
    entry: dict[str, Any],
    client: BskyClient,
    parent_height: int,
) -> dict[str, Any]:
    meme_uri = clean_uri(entry.get("meme_reply_uri")) or clean_uri(entry.get("uri"))
    if not meme_uri:
        raise ValueError("manifest row is missing meme_reply_uri/uri")

    data = client.get_post_thread(meme_uri, depth=0, parent_height=parent_height)
    if not data or data.get("_error"):
        raise ValueError(f"getPostThread failed: {(data or {}).get('_error', 'empty_response')}")

    thread = as_dict(data.get("thread"))
    if not is_thread_view(thread):
        raise ValueError("getPostThread response did not contain a threadViewPost")

    meme_view = thread["post"]
    meme_reply = post_view_to_dict(meme_view)
    if not meme_reply:
        raise ValueError("could not parse meme reply post")

    parent_views = collect_parent_views(thread)
    parent_posts = [post_view_to_dict(post) for post in parent_views]
    by_uri = {post["uri"]: post for post in parent_posts if post}
    by_uri[meme_reply["uri"]] = meme_reply

    root_uri = clean_uri(entry.get("root_post_uri")) or nested_post_uri(entry, "original_post") or meme_reply.get("root_uri")
    reply_parent_uri = clean_uri(entry.get("reply_parent_uri")) or meme_reply.get("parent_uri")
    parent_reply_uri = clean_uri(entry.get("parent_reply_uri")) or nested_post_uri(entry, "parent_reply")
    if not parent_reply_uri and reply_parent_uri and root_uri and reply_parent_uri != root_uri:
        parent_reply_uri = reply_parent_uri

    original_post = by_uri.get(root_uri) if root_uri else None
    if not original_post:
        original_post = fetch_post_dict(client, root_uri) or stub_post(root_uri)

    parent_reply = by_uri.get(parent_reply_uri) if parent_reply_uri else None
    if parent_reply_uri and not parent_reply:
        parent_reply = fetch_post_dict(client, parent_reply_uri) or stub_post(parent_reply_uri)

    quoted_post = None
    if original_post:
        quoted_post = original_post.get("quoted_post")
    quoted_uri = clean_uri(entry.get("quoted_post_uri")) or nested_post_uri(entry, "quoted_post")
    if quoted_uri and (not quoted_post or quoted_post.get("uri") != quoted_uri):
        quoted_post = fetch_post_dict(client, quoted_uri) or stub_post(quoted_uri)

    best_reply_uri = clean_uri(entry.get("best_reply_before_meme_uri")) or nested_post_uri(entry, "best_reply_before_meme")
    closest_text_uri = clean_uri(entry.get("closest_text_reply_uri")) or nested_post_uri(entry, "closest_text_reply")
    closest_sibling_uri = clean_uri(entry.get("closest_sibling_text_reply_uri")) or nested_post_uri(entry, "closest_sibling_text_reply")
    comparison_uri = clean_uri(entry.get("comparison_reply_uri")) or nested_post_uri(entry, "comparison_reply")

    best_reply_before_meme = fetch_auxiliary_reply(client, best_reply_uri)
    closest_text_reply = fetch_auxiliary_reply(client, closest_text_uri)
    closest_sibling_text_reply = fetch_auxiliary_reply(client, closest_sibling_uri)
    comparison_reply = fetch_auxiliary_reply(client, comparison_uri) if comparison_uri else None

    depth = safe_int(entry.get("thread_depth"))
    if depth is None:
        depth = 2 if parent_reply_uri else 1
    label = entry.get("thread_label") or ("re-reply" if depth >= 2 else "reply")

    uid = str(entry.get("uid") or uri_to_uid(meme_uri))
    return {
        "uid": uid,
        "uri": meme_uri,
        "meme_reply": meme_reply,
        "original_post": original_post,
        "parent_reply": parent_reply,
        "quoted_post": quoted_post,
        "best_reply_before_meme": best_reply_before_meme,
        "closest_text_reply": closest_text_reply,
        "closest_sibling_text_reply": closest_sibling_text_reply,
        "comparison_reply": comparison_reply,
        "thread_structure": {
            "depth": depth,
            "label": label,
        },
        "hydration_metadata": {
            "hydrated_at": utc_now(),
            "engagement_counts_source": "public_api_at_hydration_time",
            "manifest_uid": entry.get("uid"),
            "manifest_meme_reply_uri": meme_uri,
            "root_post_uri": root_uri,
            "reply_parent_uri": reply_parent_uri,
            "parent_reply_uri": parent_reply_uri,
            "quoted_post_uri": quoted_uri,
            "best_reply_before_meme_uri": best_reply_uri,
            "closest_text_reply_uri": closest_text_uri,
            "closest_sibling_text_reply_uri": closest_sibling_uri,
            "comparison_reply_uri": comparison_uri,
        },
    }


def hydrated_field_status(record: dict[str, Any]) -> dict[str, bool]:
    return {field: field in record for field in HYDRATED_FIELDS}


def image_filename(image: dict[str, Any], index: int) -> str:
    cid = image.get("cid")
    if cid:
        return f"{cid}.jpg"
    url = image.get("source_url") or image.get("thumb_url") or f"image-{index}"
    return hashlib.sha256(str(url).encode("utf-8")).hexdigest()[:18] + ".jpg"


def download_post_images(
    post: dict[str, Any] | None,
    role: str,
    record_uid: str,
    out_dir: Path,
    client: BskyClient,
    overwrite: bool,
) -> tuple[int, int]:
    if not post:
        return 0, 0

    post["images"] = merge_images(post.get("images") or [])

    ok_count = 0
    fail_count = 0
    for index, image in enumerate(post.get("images") or [], start=1):
        url = image.get("source_url") or image.get("thumb_url")
        if not url:
            fail_count += 1
            image["download_error"] = "missing_url"
            continue

        path = out_dir / "images" / role / record_uid / image_filename(image, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            image["local_path"] = str(path.relative_to(out_dir))
            ok_count += 1
            continue

        data = client.download_bytes(url)
        if data is None:
            fail_count += 1
            image["download_error"] = "download_failed"
            continue

        path.write_bytes(data)
        image["local_path"] = str(path.relative_to(out_dir))
        ok_count += 1
    return ok_count, fail_count


def download_record_images(
    record: dict[str, Any],
    mode: str,
    out_dir: Path,
    client: BskyClient,
    overwrite: bool,
) -> tuple[int, int]:
    if mode == "none":
        return 0, 0

    roles = ["meme_reply"]
    if mode in {"context", "all"}:
        roles.extend(
            [
                "original_post",
                "parent_reply",
                "quoted_post",
                "best_reply_before_meme",
                "closest_text_reply",
                "closest_sibling_text_reply",
                "comparison_reply",
            ]
        )

    ok_total = 0
    fail_total = 0
    for role in roles:
        ok, fail = download_post_images(record.get(role), role, record["uid"], out_dir, client, overwrite)
        ok_total += ok
        fail_total += fail

    return ok_total, fail_total


def rows_from_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("records", "items", "data", "rows"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
        if all(isinstance(value, dict) for value in data.values()):
            rows = []
            for key, value in data.items():
                row = dict(value)
                row.setdefault("uid", key)
                rows.append(row)
            return rows
        return [data]
    raise ValueError(f"{path} is JSON but does not contain object rows")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return rows_from_json(path)

    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
            else:
                entry = {"uid": line}
            entries.append(entry)
    return entries


def carry_release_fields(record: dict[str, Any], entry: dict[str, Any], args: argparse.Namespace) -> list[str]:
    fields: set[str] = set(args.carry_field or [])
    excluded = set(args.exclude_carried_field or [])
    if args.carry_all_extra_fields:
        fields.update(
            key for key in entry
            if key not in MANIFEST_HYDRATION_FIELDS and key not in HYDRATED_FIELDS
            and key not in excluded
        )

    carried = []
    for field in sorted(fields):
        if field not in entry:
            continue
        if field in record and not args.overwrite_carried_fields:
            continue
        record[field] = entry[field]
        carried.append(field)

    if carried:
        record.setdefault("hydration_metadata", {})["carried_release_fields"] = carried
    return carried


def select_entries(entries: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.uid:
        wanted = set(args.uid)
        entries = [entry for entry in entries if entry.get("uid") in wanted]
    if args.offset:
        entries = entries[args.offset :]
    if args.limit is not None:
        entries = entries[: args.limit]
    return entries


def write_record(record: dict[str, Any], out_dir: Path, overwrite: bool) -> Path:
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    path = records_dir / f"{record['uid']}.json"
    if path.exists() and not overwrite:
        return path
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hydrate conversational meme records from a UID/URI manifest."
    )
    parser.add_argument("--manifest", default="data/collection_pool_uid_manifest.jsonl")
    parser.add_argument("--out", default="hydrated_dataset")
    parser.add_argument("--limit", type=int, default=None, help="Hydrate only the first N selected rows.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N manifest rows.")
    parser.add_argument("--uid", action="append", default=[], help="Hydrate only this UID. Can repeat.")
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
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview manifest rows without API calls.")
    parser.add_argument(
        "--carry-all-extra-fields",
        action="store_true",
        help="Copy non-hydration fields from each manifest/labeled-release row into the hydrated record.",
    )
    parser.add_argument(
        "--carry-field",
        action="append",
        default=[],
        help="Copy this field from each manifest/labeled-release row into the hydrated record. Can repeat.",
    )
    parser.add_argument(
        "--exclude-carried-field",
        action="append",
        default=sorted(DEFAULT_CARRY_EXCLUDE_FIELDS),
        help="Exclude this field from --carry-all-extra-fields. Can repeat. Defaults exclude discourse_labels and ancestor_chain.",
    )
    parser.add_argument(
        "--overwrite-carried-fields",
        action="store_true",
        help="Allow carried fields to overwrite fields produced by hydration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("        Run `python download_manifest.py` first, or pass --manifest PATH.")
        return 1

    entries = select_entries(read_manifest(manifest_path), args)
    if not entries:
        print("[ERROR] No manifest entries selected.")
        return 1

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "selected": len(entries),
                "out": args.out,
                "download_images": args.download_images,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    if args.dry_run:
        preview = entries[: min(5, len(entries))]
        print(json.dumps({"preview": preview}, ensure_ascii=False, indent=2))
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = BskyClient(timeout=args.timeout, retries=args.retries, sleep=args.sleep)

    stats = {
        "total": len(entries),
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

    for index, entry in enumerate(entries, start=1):
        uid = entry.get("uid") or entry.get("meme_reply_uri") or entry.get("uri")
        print(f"[{index}/{len(entries)}] {uid}")
        try:
            record = build_record(
                entry,
                client,
                parent_height=args.parent_height,
            )
            carry_release_fields(record, entry, args)
            ok_images, failed_images = download_record_images(
                record,
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
                    "uid": entry.get("uid"),
                    "uri": entry.get("meme_reply_uri") or entry.get("uri"),
                    "error": str(exc),
                }
            )
            print(f"  [FAIL] {exc}")

    stats["finished_at"] = utc_now()
    report = {"stats": stats, "failures": failures}
    (out_dir / "hydration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["stats"], indent=2))
    return 0 if stats["hydrated"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
