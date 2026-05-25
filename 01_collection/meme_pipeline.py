#!/usr/bin/env python3
"""
Bluesky Meme Reply Detector — v4 (multi-day)

Usage:
  # Preview one day
  python meme_pipeline.py --preview --date 2023-09-01

  # Run one day
  python meme_pipeline.py --run --date 2023-09-01

  # Run one full month (2023-09-01 to 2023-09-30)
  python meme_pipeline.py --run --month 2023-09

  # Run a date range
  python meme_pipeline.py --run --start 2023-09-01 --end 2023-09-30

[Root post lookup]
  rkeys use the AT Protocol TID (timestamp-based ID) format.
  The script estimates the post date from the rkey and searches that daily
  archive first. If the post is not found, it scans nearby prior days up to the
  configured lookup window. An LRU cache avoids repeatedly loading the same
  daily archive files.

[Statistics]
  stats.json stores daily image-reply counts, meme counts, meme ratios, and
  related run statistics.
"""

import os, sys, re, json, gzip, gc, time, argparse, hashlib, random, shutil
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from collections import defaultdict, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
import easyocr

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

CONFIG = {
    # Archive root containing subdirectories such as 2023-08/, 2023-09/, ...
    "archive_base": "/home/exouser/slate_project/bluesky/firehose_archives",

    "model_dir":    "/home/exouser/checkpoints",
    "output_dir":   "./meme_dataset",
    "clip_name":    "openai/clip-vit-large-patch14-336",

    "meme_threshold":   0.5,
    "download_retries": 3,
    "request_timeout":  15,
    "use_tta":          False,     # Disable TTA for faster inference.
    "single_fold":      None,     # None=5-fold ensemble, 1~5=single model

    # Cross-day lookup cache size, in days.
    "archive_cache_size": 7,
    # Maximum number of prior days to search for root posts.
    # 0 = try only the TID-estimated date, 7 = search up to 7 prior days.
    "lookup_max_days_back": 0,

    # Maximum number of image replies to process per day.
    # None processes all rows; an integer enables random sampling.
    "daily_sample_size": None,

    # Language filter: keep meme replies that contain at least one of these tags.
    # None or [] disables language filtering.
    "lang_filter": ["en"],
    # True excludes null/empty langs; False keeps them.
    "lang_filter_strict": False,
}

# ════════════════════════════════════════════════════════════════
#  TID (Timestamp Identifier) → date
# ════════════════════════════════════════════════════════════════

_TID_CHARS = '234567abcdefghijklmnopqrstuvwxyz'
_TID_MAP   = {c: i for i, c in enumerate(_TID_CHARS)}

def rkey_to_date(rkey: str) -> date | None:
    """
    AT Protocol TID rkey → approximate date.
    TID = base32(53-bit microsecond timestamp || 10-bit clock id)
    total = 63 bits, encoded as 13 base32 chars (65 bits, 2 padding bits at top)
    """
    if not rkey or len(rkey) < 13:
        return None
    try:
        n = 0
        for c in rkey[:13]:
            n = n * 32 + _TID_MAP[c]
        ts_us = n >> 10          # upper 53 bits = microseconds since Unix epoch
        dt = datetime.fromtimestamp(ts_us / 1_000_000, tz=timezone.utc)
        if 2020 <= dt.year <= 2030:   # sanity check
            return dt.date()
    except Exception:
        pass
    return None


def passes_lang_filter(post: dict) -> bool:
    """
    Return whether a post passes the configured language filter.
    If CONFIG['lang_filter'] is empty, all posts pass.
    If langs is null/empty, lang_filter_strict=False keeps the post and
    lang_filter_strict=True excludes it.
    """
    lang_filter = CONFIG.get('lang_filter') or []
    if not lang_filter:
        return True
    langs = post.get('langs') or []
    if not langs:
        return not CONFIG.get('lang_filter_strict', False)
    return any(l in lang_filter for l in langs)


def archive_path(base: str, d: date) -> Path:
    return Path(base) / f"{d.year}-{d.month:02d}" / f"{d.year}-{d.month:02d}-{d.day:02d}.json.gz"


# ════════════════════════════════════════════════════════════════
#  Embed parsing
# ════════════════════════════════════════════════════════════════

def is_image_embed(embed) -> bool:
    if isinstance(embed, dict):
        t = embed.get('$type', '')
        return t == 'app.bsky.embed.images' or t == 'app.bsky.embed.recordWithMedia'
    if isinstance(embed, str):
        return ('Image(' in embed and 'app.bsky.embed.images' in embed) or \
               ('app.bsky.embed.images' in embed and 'images' in embed)
    return False


def _parse_embed_dict(embed: dict, did: str) -> list:
    """dict embed → [{'cid', 'alt', 'url'}, ...]"""
    t = embed.get('$type', '')
    if t == 'app.bsky.embed.recordWithMedia':
        media = embed.get('media')
        embed = media if isinstance(media, dict) else {}  # Handle malformed media values.
    raw_images = embed.get('images', []) if isinstance(embed, dict) else []
    images = []
    for img in raw_images:
        alt  = img.get('alt', '')
        blob = img.get('image', {})
        # Form 1: ref -> {'$link': 'cid'}
        # Form 2: ref -> 'cid'
        # Form 3: cid -> 'cid'
        ref = blob.get('ref', '')
        if isinstance(ref, dict):
            cid = ref.get('$link', '')
        elif ref:
            cid = str(ref)
        else:
            cid = blob.get('cid', '')
        if cid:
            images.append({
                'cid': cid,
                'alt': alt,
                'url': f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg",
            })
    return images


def parse_embed_images(embed, did: str) -> list:
    """Convert embed data (dict / Python repr string / dict repr string) to image metadata."""
    if not is_image_embed(embed):
        return []

    # -- dict format -------------------------------------------------
    if isinstance(embed, dict):
        return _parse_embed_dict(embed, did)

    # -- string format: parse dict repr with ast.literal_eval --------
    if isinstance(embed, str) and embed.strip().startswith('{'):
        import ast
        try:
            parsed = ast.literal_eval(embed)
            if isinstance(parsed, dict):
                return _parse_embed_dict(parsed, did)
        except Exception:
            pass

    # -- legacy format: Python repr string Image(...) ----------------
    images = []
    for chunk in embed.split('Image(')[1:]:
        alt = ''
        m = re.match(r"alt='((?:[^'\\]|\\.)*)'", chunk)
        if m:
            alt = m.group(1)
        else:
            m = re.match(r'alt="((?:[^"\\]|\\.)*)"', chunk)
            if m:
                alt = m.group(1)
        m = re.search(r"'\$link':\s*'([^']+)'", chunk)
        if m:
            cid = m.group(1)
            images.append({
                'cid': cid,
                'alt': alt,
                'url': f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg",
            })
    return images


# ════════════════════════════════════════════════════════════════
#  Record parsing
# ════════════════════════════════════════════════════════════════

def _safe_uri(val):
    """Return a URI string, the first list item, or None for unsupported values."""
    if isinstance(val, list):
        return val[0] if val else None
    if isinstance(val, str):
        return val or None
    return None


def uri_to_uid(uri: str) -> str:
    try:
        parts = uri.replace('at://', '').split('/')
        did_suffix = parts[0].split(':')[-1][-14:]
        rkey = parts[-1]
        return f"bsky_{did_suffix}_{rkey}"
    except Exception:
        return 'bsky_' + hashlib.sha256(uri.encode()).hexdigest()[:22]


def parse_datetime_safe(s: str) -> datetime:
    if not s:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        try:
            return datetime.strptime(s[:19], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)


def obj_to_post(obj: dict) -> dict | None:
    """Convert a firehose record to a post dict; return None for non-posts/deletes.

    Handles archive format changes:
      Legacy format (around <= 2023-09-16): top-level parent/root, create_time
      Newer format (around >= 2023-09-26): nested reply.{parent,root}, createdAt
    """
    if obj.get('type') != 'app.bsky.feed.post' or obj.get('action') == 'delete':
        return None
    uri     = obj.get('uri', '')
    did     = obj.get('author', '')
    # Format change: create_time (legacy) -> createdAt (newer).
    created = obj.get('create_time') or obj.get('createdAt') or obj.get('commit_time', '')
    # Format change: embed is usually top-level, but may also appear under record.
    embed   = obj.get('embed') or (obj.get('record') or {}).get('embed')
    images  = parse_embed_images(embed, did) if embed else []

    # Format change: direct parent/root (legacy) -> nested reply.{parent,root}.
    # Some 2024+ records contain non-dict reply/parent/root values.
    # Some records after 2024-06 contain URI fields as lists.
    _reply     = obj.get('reply')
    reply_ref  = _reply if isinstance(_reply, dict) else {}
    _parent    = reply_ref.get('parent') or obj.get('parent')
    _root      = reply_ref.get('root')   or obj.get('root')
    parent_uri = _safe_uri(_parent.get('uri')) if isinstance(_parent, dict) else None
    root_uri   = _safe_uri(_root.get('uri'))   if isinstance(_root,   dict) else None

    return {
        'uid':        uri_to_uid(uri),
        'uri':        uri,
        'did':        did,
        'rkey':       uri.split('/')[-1],
        'seq':        obj.get('seq'),
        'text':       obj.get('text') or '',
        'langs':      obj.get('langs') or [],
        'post_url':   obj.get('url', ''),
        'created_at': created,
        'is_reply':   bool(parent_uri),
        'is_re_reply': bool(parent_uri and root_uri and parent_uri != root_uri),
        'parent_uri': parent_uri,
        'root_uri':   root_uri,
        'images':     images,
        'has_image':  len(images) > 0,
        'like_count':  0,
        'reply_count': 0,
        '_dt':        parse_datetime_safe(created),
    }


# ════════════════════════════════════════════════════════════════
#  Single-day archive loading
# ════════════════════════════════════════════════════════════════

def load_day(d: date, base: str, verbose: bool = True) -> dict | None:
    """
    Load one daily archive.
    Returns dict with by_uri, by_parent, by_root, image_replies
    or None if file doesn't exist.
    """
    path = archive_path(base, d)
    if not path.exists():
        if verbose:
            print(f"  [skip] {path.name} not found")
        return None

    by_uri      = {}
    by_parent   = defaultdict(list)
    by_root     = defaultdict(list)
    like_counts = defaultdict(int)
    image_replies = []

    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = obj.get('type', '')
            if rtype == 'app.bsky.feed.post':
                post = obj_to_post(obj)
                if post is None:
                    continue
                uri = post['uri']
                by_uri[uri] = post
                if post['is_reply']:
                    if post['parent_uri']:
                        by_parent[post['parent_uri']].append(uri)
                    if post['root_uri']:
                        by_root[post['root_uri']].append(uri)
                    if post['has_image'] and passes_lang_filter(post) and len(post['images']) == 1:
                        image_replies.append(uri)

            elif rtype == 'app.bsky.feed.like':
                # Legacy format: 'root'; newer format (2024-06+): 'subject'.
                root_info = obj.get('subject') or obj.get('root') or {}
                liked_uri = _safe_uri(root_info.get('uri')) if isinstance(root_info, dict) else None
                if liked_uri:
                    like_counts[liked_uri] += (1 if obj.get('action') == 'create' else -1)

    for uri, post in by_uri.items():
        post['like_count']  = max(0, like_counts.get(uri, 0))
        post['reply_count'] = len(by_parent.get(uri, []))

    return {
        'by_uri':        by_uri,
        'by_parent':     by_parent,
        'by_root':       by_root,
        'image_replies': image_replies,
        'date':          d,
    }


# ════════════════════════════════════════════════════════════════
#  ArchiveManager - cross-day post lookup
# ════════════════════════════════════════════════════════════════

class ArchiveManager:
    """
    Look up post URIs across multiple daily archives.
    Keeps up to N days of by_uri indexes in an LRU cache.
    """

    def __init__(self, base: str, cache_size: int = 14,
                 max_days_back: int = 30):
        self.base          = base
        self.cache_size    = cache_size
        self.max_days_back = max_days_back
        self._cache        = OrderedDict()   # date_str → {uri: compact_post}

    def _compact(self, by_uri: dict) -> dict:
        """Keep only the by_uri fields needed for lookup."""
        keep = ('uid','uri','did','rkey','seq','text','langs','post_url',
                'created_at','is_reply','is_re_reply','parent_uri','root_uri',
                'images','has_image','like_count','reply_count','_dt')
        return {uri: {k: p[k] for k in keep if k in p}
                for uri, p in by_uri.items()}

    def _load_date(self, date_str: str):
        if date_str in self._cache:
            self._cache.move_to_end(date_str)
            return
        try:
            d   = date.fromisoformat(date_str)
            idx = load_day(d, self.base, verbose=False)
            compact = self._compact(idx['by_uri']) if idx else {}
        except Exception:
            compact = {}
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[date_str] = compact

    def lookup(self, uri: str, current_date: date,
               current_idx: dict | None = None,
               cache_only: bool = False) -> dict | None:
        """
        Search for a URI across available archives.
        cache_only=True: search only memory (current_idx + existing cache) and
                         perform no disk I/O. Useful for latency-sensitive
                         context lookup immediately before saving.
        cache_only=False: use the TID-estimated date plus prior-day search,
                          loading archives from disk as needed.
        """
        if not uri:
            return None

        # 1) Return immediately if the post is in today's archive.
        if current_idx and uri in current_idx['by_uri']:
            return current_idx['by_uri'][uri]

        # cache_only: search only already-loaded indexes, with no disk access.
        if cache_only:
            for cached in self._cache.values():
                post = cached.get(uri)
                if post:
                    return post
            return None

        # 2) Estimate the post date by decoding the rkey TID.
        rkey = uri.split('/')[-1]
        tid_date = rkey_to_date(rkey)

        dates_to_try = []
        if tid_date and tid_date != current_date:
            dates_to_try.append(tid_date.strftime('%Y-%m-%d'))

        # 3) Search prior days relative to the current date.
        for delta in range(1, self.max_days_back + 1):
            d_str = (current_date - timedelta(days=delta)).strftime('%Y-%m-%d')
            if d_str not in dates_to_try:
                dates_to_try.append(d_str)

        for d_str in dates_to_try:
            self._load_date(d_str)
            post = self._cache.get(d_str, {}).get(uri)
            if post:
                return post

        return None


# ════════════════════════════════════════════════════════════════
#  Thread context
# ════════════════════════════════════════════════════════════════

def build_context(meme_uri: str, day_idx: dict,
                  mgr: ArchiveManager, current_date: date,
                  cache_only: bool = True) -> dict:
    by_uri    = day_idx['by_uri']
    by_parent = day_idx['by_parent']
    by_root   = day_idx['by_root']
    meme      = by_uri[meme_uri]
    meme_dt   = meme['_dt']

    root_uri   = meme['root_uri']
    parent_uri = meme['parent_uri']

    # Look up root/parent posts.
    # cache_only=True (default): search memory only, avoiding save-time disk I/O.
    root_post   = mgr.lookup(root_uri,   current_date, day_idx, cache_only=cache_only)
    parent_post = mgr.lookup(parent_uri, current_date, day_idx, cache_only=cache_only) if meme['is_re_reply'] else None

    depth       = 2 if meme['is_re_reply'] else 1
    depth_label = {1: 'reply', 2: 're-reply', 3: 're-re-reply'}.get(depth, f'depth-{depth}')

    # siblings = other replies attached to the same parent.
    siblings = [by_uri[u] for u in by_parent.get(parent_uri, [])
                if u != meme_uri and u in by_uri]

    before_meme = sorted(
        [p for p in siblings if p['_dt'] <= meme_dt],
        key=lambda p: p['_dt']
    )
    best_before = (max(before_meme, key=lambda p: p['like_count'])
                   if before_meme else None)

    # closest text reply (same root thread, current archive) - second priority: timely.
    text_only = [
        by_uri[u] for u in by_root.get(root_uri, [])
        if u in by_uri and u != meme_uri
        and not by_uri[u]['has_image']
        and by_uri[u]['text'].strip()
    ]
    closest_text = (
        min(text_only, key=lambda p: abs((p['_dt'] - meme_dt).total_seconds()))
        if text_only else None
    )

    # First priority: timely + structural = temporally closest text sibling.
    sibling_text = [p for p in siblings if not p['has_image'] and p['text'].strip()]
    closest_sibling_text = (
        min(sibling_text, key=lambda p: abs((p['_dt'] - meme_dt).total_seconds()))
        if sibling_text else None
    )

    return {
        'depth':                  depth,
        'structure_label':        depth_label,
        'root_post':              root_post,
        'parent_post':            parent_post,
        'best_reply_before_meme': best_before,
        'closest_text_reply':     closest_text,
        'closest_sibling_text':   closest_sibling_text,
        '_meme_post':             meme,
    }


# ════════════════════════════════════════════════════════════════
#  JSON serialization
# ════════════════════════════════════════════════════════════════

_BASE_FIELDS = (
    'uid','uri','did','rkey','seq','text','langs','post_url','created_at',
    'is_reply','is_re_reply','root_uri','parent_uri',
    'like_count','reply_count','has_image',
)

def _extract_external(post: dict) -> dict:
    """Extract external link title/URL metadata from embeds."""
    embed = post.get('embed')
    if not embed:
        return {}
    # dict format
    if isinstance(embed, dict):
        t = embed.get('$type', '')
        if t in ('app.bsky.embed.external', 'app.bsky.embed.external#view'):
            ext = embed.get('external', {})
            return {
                'external_title':       ext.get('title'),
                'external_url':         ext.get('uri'),
                'external_description': ext.get('description'),
            }
        # external inside recordWithMedia
        if t in ('app.bsky.embed.recordWithMedia', 'app.bsky.embed.recordWithMedia#view'):
            media = embed.get('media', {}) or {}
            if 'external' in media:
                ext = media['external']
                return {
                    'external_title':       ext.get('title'),
                    'external_url':         ext.get('uri'),
                    'external_description': ext.get('description'),
                }
    # legacy string repr format (simple regex)
    if isinstance(embed, str) and 'External(' in embed:
        m = re.search(r"title='((?:[^'\\]|\\.)*)'", embed)
        title = m.group(1) if m else None
        m = re.search(r"uri='((?:[^'\\]|\\.)*)'", embed)
        url = m.group(1) if m else None
        if title or url:
            return {'external_title': title, 'external_url': url, 'external_description': None}
    return {}


def post_to_dict(post: dict | None,
                 image_saves: list | None = None) -> dict | None:
    if post is None:
        return None
    d = {k: post.get(k) for k in _BASE_FIELDS}
    d['in_archive'] = True
    if image_saves is not None:
        d['images'] = image_saves
    else:
        d['images'] = [{'cid': img['cid'], 'alt': img.get('alt',''),
                        'source_url': img['url']}
                       for img in (post.get('images') or [])]
    d.update(_extract_external(post))
    return d


def post_stub(uri: str) -> dict:
    """Return a URI-only stub for posts missing from the archive."""
    return {
        'uri': uri, 'uid': uri_to_uid(uri),
        'in_archive': False,
        'did': None, 'rkey': uri.split('/')[-1] if uri else None,
        'text': None, 'created_at': None,
        'like_count': None, 'reply_count': None,
        'has_image': None, 'images': [],
    }


def build_record(ctx: dict, meme_prob: float,
                 meme_image_saves: list,
                 orig_image_saves: list,
                 process_date: str) -> dict:
    meme       = ctx['_meme_post']
    root_post  = ctx['root_post']
    root_uri   = meme['root_uri']

    # original_post
    if root_post:
        orig_dict = post_to_dict(root_post, image_saves=orig_image_saves or None)
    else:
        stub = post_stub(root_uri)
        stub['images'] = orig_image_saves
        orig_dict = stub

    meme_dict = post_to_dict(meme, image_saves=meme_image_saves)

    ct  = ctx['closest_text_reply']
    cs  = ctx['closest_sibling_text']
    bb  = ctx['best_reply_before_meme']
    pp  = ctx['parent_post']

    def _with_delta(p):
        if p is None: return None
        d = post_to_dict(p)
        d['time_delta_seconds'] = round(abs((p['_dt'] - meme['_dt']).total_seconds()), 1)
        return d

    ct_dict = _with_delta(ct)
    cs_dict = _with_delta(cs)

    # comparison_reply: first priority timely+structural, then timely, then structural.
    if cs:
        comp_dict = cs_dict.copy()
        comp_dict['selected_by'] = 'timely_structural'
    elif ct:
        comp_dict = ct_dict.copy()
        comp_dict['selected_by'] = 'timely'
    elif bb:
        comp_dict = post_to_dict(bb)
        comp_dict['selected_by'] = 'structural'
    else:
        comp_dict = None

    return {
        # identifiers
        'uid':  meme['uid'],
        'uri':  meme['uri'],

        # classification
        'meme_prob':  round(meme_prob, 4),
        'is_meme':    True,
        'threshold':  CONFIG['meme_threshold'],

        # [2] meme reply with image
        'meme_reply': meme_dict,

        # [1] original post
        'original_post': orig_dict,

        # [5] thread structure
        'thread_structure': {
            'depth': ctx['depth'],
            'label': ctx['structure_label'],
        },

        # [7] parent reply (depth >= 2 only)
        'parent_reply': post_to_dict(pp) if pp else None,

        # [6] best reply before meme (most likes, structural)
        'best_reply_before_meme': post_to_dict(bb) if bb else None,

        # [3] closest text reply (timely)
        'closest_text_reply': ct_dict,

        # [3a] closest sibling text reply (timely + structural)
        'closest_sibling_text_reply': cs_dict,

        # Selected comparison reply: timely+structural, then timely, then structural.
        'comparison_reply': comp_dict,

        # metadata
        'source_file':   f"{process_date}.json.gz",
        'process_date':  process_date,
        'processed_at':  datetime.now(timezone.utc).isoformat(),
    }


# ════════════════════════════════════════════════════════════════
#  Image download
# ════════════════════════════════════════════════════════════════

# Module-level Session reuses TCP connections for faster downloads.
_SESSION = requests.Session()
_SESSION.headers.update({'User-Agent': 'MemeResearchBot/1.0 (academic research)'})

def download_image(url: str, save_path: Path,
                   retries: int = 2, timeout: int = 15) -> bool:
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, timeout=(5, timeout), stream=True)
            if r.status_code == 404:
                return False
            if r.status_code == 200:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                Image.open(save_path).verify()
                return True
        except Exception:
            try:
                save_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < retries - 1:
                time.sleep(1)
    return False


def download_post_images(post: dict, folder: Path, output_dir: Path,
                         retries: int = 3, timeout: int = 15) -> list:
    saved = []
    for img in (post.get('images') or []):
        cid  = img['cid']
        url  = img['url']
        path = folder / post['uid'] / f"{cid}.jpg"

        # Verify existing files by reading them; previous interrupted runs may leave corrupt files.
        ok = False
        if path.exists():
            try:
                tmp = Image.open(path)
                tmp.load()   # Force data loading instead of relying on lazy open.
                ok = True
            except Exception:
                try: path.unlink(missing_ok=True)
                except Exception: pass

        if not ok:
            ok = download_image(url, path, retries, timeout)

        if ok and path.exists():
            saved.append({
                'local_path': str(path.relative_to(output_dir)),
                'cid':        cid,
                'alt':        img.get('alt', ''),
                'source_url': url,
            })
    return saved


# ════════════════════════════════════════════════════════════════
#  MemeTector v4
# ════════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim*2, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.Sigmoid())
        self.ln_i = nn.LayerNorm(dim)
        self.ln_t = nn.LayerNorm(dim)
    def forward(self, i, t):
        g = self.gate_net(torch.cat([i, t], dim=-1))
        return g * self.ln_i(i) + (1 - g) * self.ln_t(t), g


class MemeDetectorV4(nn.Module):
    def __init__(self, clip_name, dropout=0.4):
        super().__init__()
        self.clip   = CLIPModel.from_pretrained(clip_name)
        d           = self.clip.config.projection_dim
        self.fusion = GatedFusion(d)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d*3+1), nn.Linear(d*3+1, 768), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(768, 256), nn.GELU(),
            nn.Dropout(dropout*0.5), nn.Linear(256, 2))
    def encode_image(self, pv):
        return self.clip.visual_projection(
            self.clip.vision_model(pixel_values=pv, return_dict=True).pooler_output)
    def encode_text(self, iid, am):
        return self.clip.text_projection(
            self.clip.text_model(input_ids=iid, attention_mask=am,
                                 return_dict=True).pooler_output)
    def forward(self, pv, iid, am):
        img = F.normalize(self.encode_image(pv), dim=-1)
        txt = F.normalize(self.encode_text(iid, am), dim=-1)
        cos = (img * txt).sum(-1, keepdim=True)
        fused, _ = self.fusion(img, txt)
        return self.classifier(torch.cat([img, txt, fused, cos], dim=-1))


IMG_SIZE    = 336
DEFAULT_OCR = 'an image'
TTA_AUGS    = [
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE))]),
    T.Compose([T.Resize((int(IMG_SIZE*1.14), int(IMG_SIZE*1.14))), T.CenterCrop(IMG_SIZE)]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(p=1.0)]),
    T.Compose([T.Resize((int(IMG_SIZE*1.14), int(IMG_SIZE*1.14))),
               T.FiveCrop(IMG_SIZE), T.Lambda(lambda c: c[0])]),
    T.Compose([T.Resize((int(IMG_SIZE*1.14), int(IMG_SIZE*1.14))),
               T.FiveCrop(IMG_SIZE), T.Lambda(lambda c: c[4])]),
    T.Compose([T.Resize((int(IMG_SIZE*1.3), int(IMG_SIZE*1.3))), T.CenterCrop(IMG_SIZE)]),
]


def load_models(model_dir, clip_name, device, single_fold=None):
    folds  = [single_fold] if single_fold else [1,2,3,4,5]
    models = []
    for f in folds:
        ckpt = Path(model_dir) / f'fold{f}_best.pth'
        if not ckpt.exists():
            print(f"  ⚠  {ckpt.name} not found"); continue
        m = MemeDetectorV4(clip_name)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.to(device).eval()
        models.append(m)
        print(f"  Loaded {ckpt.name}")
    if not models:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")
    return models


def run_ocr(pil_img, reader, thresh=0.25, max_chars=200, timeout=10):
    """Run OCR and return an empty string if it does not finish within timeout seconds."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    def _ocr():
        try:
            arr  = np.array(pil_img.convert('RGB'))
            dets = reader.readtext(arr, detail=1, paragraph=False)
            txts = [t.strip() for (_, t, c) in dets if c >= thresh and len(t.strip()) >= 2]
            return ' '.join(txts)[:max_chars]
        except Exception:
            return ''
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_ocr)
    ex.shutdown(wait=False)   # Return immediately without waiting for the worker thread.
    try:
        return fut.result(timeout=timeout)
    except Exception:
        return ''


def classify(pil_img, ocr_text, models, processor, device, use_tta=True):
    """Run CLIP 5-fold ensemble classification. OCR runs on CPU when enabled."""
    augs  = TTA_AUGS if use_tta else [TTA_AUGS[0]]
    text  = ocr_text.strip() or DEFAULT_OCR
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    all_probs = []
    for model in models:
        vp = []
        for aug in augs:
            enc = processor(images=aug(pil_img.convert('RGB')), text=text,
                            return_tensors='pt', padding='max_length',
                            max_length=77, truncation=True)
            pv  = enc['pixel_values'].to(device)
            iid = enc['input_ids'].to(device)
            am  = enc['attention_mask'].to(device)
            with torch.no_grad(), autocast(device, dtype=dtype):
                p = torch.softmax(model(pv, iid, am).float(), dim=1).cpu().numpy()[0]
            vp.append(p); del pv, iid, am
        all_probs.append(np.mean(vp, axis=0))
    avg = np.mean(all_probs, axis=0)
    mp  = float(avg[1])
    return (1 if mp >= CONFIG['meme_threshold'] else 0), mp


def classify_batch(pil_imgs: list, ocr_texts: list,
                   models, processor, device) -> list:
    """Run batched GPU inference and return [(label, meme_prob), ...]."""
    texts = [t.strip() or DEFAULT_OCR for t in ocr_texts]
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    aug   = TTA_AUGS[0]   # TTA is disabled, so use the basic resize transform.

    all_probs = []
    for model in models:
        enc = processor(
            images=[aug(img.convert('RGB')) for img in pil_imgs],
            text=texts,
            return_tensors='pt',
            padding='max_length',
            max_length=77,
            truncation=True,
        )
        pv  = enc['pixel_values'].to(device)
        iid = enc['input_ids'].to(device)
        am  = enc['attention_mask'].to(device)
        with torch.no_grad(), autocast(device, dtype=dtype):
            probs = torch.softmax(model(pv, iid, am).float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        del pv, iid, am

    avg = np.mean(all_probs, axis=0)   # (N, 2)
    return [(1 if float(avg[i][1]) >= CONFIG['meme_threshold'] else 0,
             round(float(avg[i][1]), 4))
            for i in range(len(pil_imgs))]


# ════════════════════════════════════════════════════════════════
#  Date range parsing
# ════════════════════════════════════════════════════════════════

def parse_date_range(args) -> list[date]:
    if args.date:
        return [date.fromisoformat(args.date)]
    if args.month:
        yr, mo = map(int, args.month.split('-'))
        d = date(yr, mo, 1)
        dates = []
        while d.month == mo:
            dates.append(d)
            d += timedelta(days=1)
        return dates
    if args.start and args.end:
        d   = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        dates = []
        while d <= end:
            dates.append(d)
            d += timedelta(days=1)
        return dates
    raise ValueError("Date required: --date / --month / --start + --end")


# ════════════════════════════════════════════════════════════════
#  Preview
# ════════════════════════════════════════════════════════════════

def run_preview(dates: list[date], n_per_day: int = 10):
    mgr = ArchiveManager(CONFIG['archive_base'],
                         CONFIG['archive_cache_size'],
                         CONFIG['lookup_max_days_back'])
    for d in dates:
        print(f"\n{'='*65}")
        print(f"  Preview: {d}")
        day_idx = load_day(d, CONFIG['archive_base'])
        if not day_idx:
            continue
        ir = day_idx['image_replies']
        print(f"  Image replies: {len(ir)}")
        by_uri = day_idx['by_uri']
        for i, uri in enumerate(ir[:n_per_day]):
            post = by_uri[uri]
            ctx  = build_context(uri, day_idx, mgr, d)
            rp   = ctx['root_post']
            ct   = ctx['closest_text_reply']
            bb   = ctx['best_reply_before_meme']
            pp   = ctx['parent_post']

            print(f"\n  [{i+1}] uid={post['uid']}")
            print(f"    text:      {post['text'][:80]!r}")
            print(f"    structure: {ctx['structure_label']}")
            print(f"    images:    {len(post['images'])}")
            for img in post['images']:
                print(f"      {img['url']}")
            print(f"    likes={post['like_count']}  replies={post['reply_count']}")
            if rp:
                print(f"    [1] original_post: {rp['text'][:70]!r}  in_archive=True")
                print(f"        has_image={rp['has_image']}  images={len(rp['images'])}")
            else:
                print(f"    [1] original_post: NOT FOUND  root_uri={post['root_uri']}")
            if ct:
                dt = abs((ct['_dt'] - post['_dt']).total_seconds())
                print(f"    [3] closest_text: {ct['text'][:60]!r}  Δt={dt:.0f}s")
            if bb:
                print(f"    [6] best_before:  {bb['text'][:60]!r}  likes={bb['like_count']}")
            if pp:
                print(f"    [7] parent_reply: {pp['text'][:60]!r}")
        print(f"\n  Original posts found in archive: "
              f"{sum(1 for u in ir[:n_per_day] if build_context(u, day_idx, mgr, d)['root_post'])}"
              f"/{min(n_per_day, len(ir))}")


# ════════════════════════════════════════════════════════════════
#  SIGALRM-based timeout (Linux main thread only)
# ════════════════════════════════════════════════════════════════

import signal as _signal

class _CtxTimeout(Exception):
    pass

def _run_with_timeout(fn, *args, timeout: int = 10, **kwargs):
    """
    Run fn(*args, **kwargs) and return None if it exceeds timeout seconds.
    Uses SIGALRM, so it works only on the Linux main thread.
    Compared with creating a ThreadPoolExecutor each time, this avoids:
      - thread leaks
      - concurrent access to ArchiveManager._cache
    """
    def _handler(signum, frame):
        raise _CtxTimeout()

    old_handler = _signal.signal(_signal.SIGALRM, _handler)
    _signal.alarm(timeout)
    try:
        result = fn(*args, **kwargs)
        _signal.alarm(0)
        return result
    except _CtxTimeout:
        return None
    except Exception:
        raise
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old_handler)


# ════════════════════════════════════════════════════════════════
#  Full run
# ════════════════════════════════════════════════════════════════

def run_full(dates: list[date], monthly_quota: int = 6250):
    output_dir   = Path(CONFIG['output_dir'])
    meme_img_dir = output_dir / 'meme_images'
    orig_img_dir = output_dir / 'original_post_images'
    records_dir  = output_dir / 'records'
    for d in (meme_img_dir, orig_img_dir, records_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading models ...")
    processor  = CLIPProcessor.from_pretrained(CONFIG['clip_name'])
    models     = load_models(CONFIG['model_dir'], CONFIG['clip_name'],
                              device, CONFIG['single_fold'])
    ocr_reader = None  # OCR disabled for speed; use CLIP image inference only.

    mgr = ArchiveManager(CONFIG['archive_base'],
                         CONFIG['archive_cache_size'],
                         CONFIG['lookup_max_days_back'])

    # Overall statistics.
    all_stats = []
    total     = {'image_replies': 0, 'meme': 0, 'not_meme': 0,
                 'dl_fail': 0, 'error': 0, 'orig_imgs': 0}

    # Group dates by month, then randomly shuffle days within each month.
    from itertools import groupby
    months_shuffled = []
    for _, group in groupby(dates, key=lambda d: (d.year, d.month)):
        month_days = list(group)
        random.shuffle(month_days)
        months_shuffled.extend(month_days)
    dates = months_shuffled

    def _count_existing_month(year: int, month: int) -> int:
        """Count existing collected memes for a month from index_YYYY-MM-DD.jsonl files."""
        count = 0
        for idx_f in sorted(output_dir.glob(f'index_{year}-{month:02d}-*.jsonl')):
            try:
                count += sum(1 for ln in idx_f.read_text('utf-8').splitlines() if ln.strip())
            except Exception:
                pass
        return count

    # Track monthly quotas.
    current_month     = None
    month_meme_count  = 0

    for d in dates:
        d_str     = d.strftime('%Y-%m-%d')
        month_key = (d.year, d.month)

        # Reset counters and restore existing counts when the month changes.
        if month_key != current_month:
            if current_month is not None:
                print(f"\n  -- {current_month[0]}-{current_month[1]:02d} complete: "
                      f"{month_meme_count} memes collected --")
                sys.stdout.flush()
                # Merge and save monthly index files for crash recovery.
                _ym = f"{current_month[0]}-{current_month[1]:02d}"
                _month_idx = output_dir / f'meme_index_{_ym}.jsonl'
                with open(_month_idx, 'w', encoding='utf-8') as _mf:
                    for _df in sorted(output_dir.glob(f'index_{_ym}-*.jsonl')):
                        _mf.write(_df.read_text('utf-8'))
                _cnt = sum(1 for ln in _month_idx.read_text('utf-8').splitlines() if ln.strip())
                print(f"  [monthly-save] {_month_idx.name}  ({_cnt} records)")
                sys.stdout.flush()
            current_month    = month_key
            month_meme_count = _count_existing_month(month_key[0], month_key[1])
            if month_meme_count > 0:
                print(f"  [resume] {month_key[0]}-{month_key[1]:02d} restored existing count: {month_meme_count}")
                sys.stdout.flush()
            print(f"  > Starting {month_key[0]}-{month_key[1]:02d}")
            sys.stdout.flush()

        # Skip once this month's quota has already been met.
        if monthly_quota and month_meme_count >= monthly_quota:
            all_stats.append({'date': d_str, 'skipped': True, 'reason': 'monthly_quota_met'})
            continue

        print(f"\n{'─'*55}")
        print(f"  Processing {d_str} ...  "
              f"({month_meme_count}/{monthly_quota} collected this month)")

        day_idx = load_day(d, CONFIG['archive_base'])
        if not day_idx:
            all_stats.append({'date': d_str, 'skipped': True})
            continue

        ir = day_idx['image_replies']

        # Shuffle one day's data for more even sampling.
        random.shuffle(ir)

        # Limit daily processing to reduce CDN rate-limit and memory pressure.
        daily_cap = CONFIG.get('daily_sample_size')
        if daily_cap and len(ir) > daily_cap:
            print(f"  [sample] sampled {daily_cap:,}/{len(ir):,} rows (daily_sample_size)")
            ir = ir[:daily_cap]

        day_stats = {
            'date':             d_str,
            'image_reply_count': len(ir),
            'meme_count':        0,
            'not_meme_count':    0,
            'dl_fail_count':     0,
            'error_count':       0,
            'orig_img_saved':    0,
        }
        total['image_replies'] += len(ir)

        # Lightweight daily index file.
        day_index_path = output_dir / f'index_{d_str}.jsonl'
        day_index_f    = open(day_index_path, 'w', encoding='utf-8')

        quota_reached = False
        by_uri = day_idx['by_uri']

        BATCH_SIZE = 32   # GPU batch size
        PREFETCH   = 32   # Concurrent downloads

        dl_executor = ThreadPoolExecutor(max_workers=PREFETCH)

        def submit_download(u):
            return dl_executor.submit(
                download_post_images, by_uri[u], meme_img_dir, output_dir,
                CONFIG['download_retries'], CONFIG['request_timeout'])

        uri_iter = iter(ir)
        pending  = deque()

        for _ in range(PREFETCH):
            try:
                u = next(uri_iter)
                pending.append((u, submit_download(u)))
            except StopIteration:
                break

        pbar = tqdm(total=len(ir), desc=d_str, leave=False)

        while pending and not quota_reached:
            # -- collect a batch -------------------------------------
            batch_uris  = []
            batch_saves = []

            while len(batch_uris) < BATCH_SIZE and pending:
                if monthly_quota and month_meme_count + len(batch_uris) >= monthly_quota:
                    quota_reached = True   # Prevent an empty batch from causing an outer-loop spin.
                    break

                uri, future = pending.popleft()
                pbar.update(1)

                try:
                    u = next(uri_iter)
                    pending.append((u, submit_download(u)))
                except StopIteration:
                    pass

                try:
                    meme_saves = future.result(timeout=10)  # Skip if download exceeds 10 seconds.
                    if not meme_saves:
                        day_stats['dl_fail_count'] += 1
                        total['dl_fail'] += 1
                        continue
                    batch_uris.append(uri)
                    batch_saves.append(meme_saves)
                except Exception as e:
                    day_stats['error_count'] += 1
                    total['error'] += 1
                    tqdm.write(f"  ERROR download ({by_uri[uri]['uid']}): {e}")

            if not batch_uris:
                continue

            # -- load images -----------------------------------------
            loaded = []   # (uri, meme_saves, pil_img)
            for uri, meme_saves in zip(batch_uris, batch_saves):
                try:
                    pil_img = Image.open(output_dir / meme_saves[0]['local_path'])
                    loaded.append((uri, meme_saves, pil_img))
                except Exception as e:
                    tqdm.write(f"  ERROR open ({by_uri[uri]['uid']}): {e}")
                    for s in meme_saves:
                        fp = output_dir / s['local_path']
                        try: fp.unlink(missing_ok=True); shutil.rmtree(fp.parent, ignore_errors=True)
                        except Exception: pass

            if not loaded:
                continue

            # -- batched GPU classification --------------------------
            tqdm.write(f"  [batch] starting GPU classification for {len(loaded)} images")
            try:
                pil_imgs = [x[2] for x in loaded]
                results  = classify_batch(pil_imgs, [DEFAULT_OCR]*len(pil_imgs),
                                          models, processor, device)
            except Exception as e:
                import traceback
                tqdm.write(f"  ERROR classify_batch: {e}\n{traceback.format_exc()}")
                for uri, meme_saves, _ in loaded:
                    for s in meme_saves:
                        fp = output_dir / s['local_path']
                        try: fp.unlink(missing_ok=True); shutil.rmtree(fp.parent, ignore_errors=True)
                        except Exception: pass
                continue

            tqdm.write(f"  [batch] classification complete")

            # -- process results -------------------------------------
            processed_in_batch = 0
            for (uri, meme_saves, pil_img), (label, meme_prob) in zip(loaded, results):
                if monthly_quota and month_meme_count >= monthly_quota:
                    quota_reached = True
                    break

                post = by_uri[uri]
                try:
                    tqdm.write(f"  [classify] {post['uid']} label={label} prob={meme_prob:.4f}")
                    processed_in_batch += 1

                    if label == 0:
                        for s in meme_saves:
                            fp = output_dir / s['local_path']
                            try:
                                fp.unlink(missing_ok=True)
                                shutil.rmtree(fp.parent, ignore_errors=True)
                            except Exception: pass
                        day_stats['not_meme_count'] += 1
                        total['not_meme'] += 1
                        continue

                    # Meme prediction.
                    tqdm.write(f"  [MEME] {post['uid']} prob={meme_prob:.4f} saving...")
                    day_stats['meme_count'] += 1
                    total['meme'] += 1
                    month_meme_count += 1

                    # build_context(cache_only=True) returns immediately without disk I/O.
                    try:
                        ctx = build_context(uri, day_idx, mgr, d, cache_only=True)
                    except Exception as e:
                        tqdm.write(f"  [ctx] ERROR; saving without context: {e}")
                        ctx = {
                            'root_post': None, 'depth': 0,
                            'structure_label': 'unknown',
                            'closest_text_reply': None,
                            'best_reply_before_meme': None,
                            'closest_sibling_text': None,
                            'parent_post': None,
                        }
                    ctx['_meme_post'] = post

                    orig_saves = []  # Original-post images are downloaded separately.

                    record = build_record(ctx, meme_prob, meme_saves, orig_saves, d_str)

                    (records_dir / f"{post['uid']}.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2, default=str),
                        encoding='utf-8')

                    day_index_f.write(json.dumps({
                        'uid':              record['uid'],
                        'uri':              record['uri'],
                        'meme_prob':        record['meme_prob'],
                        'created_at':       post['created_at'],
                        'thread_depth':     ctx['depth'],
                        'thread_label':     ctx['structure_label'],
                        'like_count':       post['like_count'],
                        'reply_count':      post['reply_count'],
                        'text_preview':     post['text'][:100],
                        'meme_image_paths': [s['local_path'] for s in meme_saves],
                        'orig_image_paths': [],
                        'original_post_uri':        post['root_uri'],
                        'original_post_in_archive': ctx['root_post'] is not None,
                    }, ensure_ascii=False) + '\n')
                    tqdm.write(f"  [MEME] saved (monthly count {month_meme_count}/{monthly_quota})")

                except Exception as e:
                    import traceback
                    day_stats['error_count'] += 1
                    total['error'] += 1
                    tqdm.write(f"  ERROR ({post['uid']}): {e}\n{traceback.format_exc()}")
                    for s in meme_saves:
                        fp = output_dir / s['local_path']
                        try:
                            fp.unlink(missing_ok=True)
                            shutil.rmtree(fp.parent, ignore_errors=True)
                        except Exception: pass

            tqdm.write(f"  [batch] processed {processed_in_batch}/{len(loaded)} images")

            # Check quota after a natural loop exit in case the last item reached the quota.
            if monthly_quota and month_meme_count >= monthly_quota:
                quota_reached = True

        # Print daily stats before cleanup, in case cleanup hangs.
        img_cnt  = day_stats['image_reply_count']
        meme_cnt = day_stats['meme_count']
        day_stats['meme_ratio'] = round(meme_cnt / img_cnt, 4) if img_cnt else 0
        all_stats.append(day_stats)
        quota_msg = '  [monthly quota reached; skipping remaining days]' if quota_reached else ''
        tqdm.write(f"\n  {d_str}: image_replies={img_cnt}  meme={meme_cnt}  "
                   f"not_meme={day_stats['not_meme_count']}  "
                   f"ratio={day_stats['meme_ratio']:.1%}"
                   f"  (monthly total {month_meme_count}/{monthly_quota}){quota_msg}")

        # ── cleanup ────────────────────────────────────────────────
        # Cancel remaining pending futures when quota is reached.
        if quota_reached:
            for _, fut in list(pending):
                fut.cancel()
            pending.clear()

        # Use disable instead of pbar.close(); with leave=False and an incomplete
        # bar, close() was observed to block on stderr flushing.
        pbar.disable = True

        dl_executor.shutdown(wait=False, cancel_futures=True)
        day_index_f.close()

    # Combined JSON output.
    all_records = []
    for fp in sorted(records_dir.glob('*.json')):
        try:
            all_records.append(json.loads(fp.read_text('utf-8')))
        except Exception:
            pass
    (output_dir / 'all_memes.json').write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8')

    # Save the last monthly index after the loop because no month boundary follows it.
    if current_month is not None:
        _ym = f"{current_month[0]}-{current_month[1]:02d}"
        _month_idx = output_dir / f'meme_index_{_ym}.jsonl'
        with open(_month_idx, 'w', encoding='utf-8') as _mf:
            for _df in sorted(output_dir.glob(f'index_{_ym}-*.jsonl')):
                _mf.write(_df.read_text('utf-8'))
        _cnt = sum(1 for ln in _month_idx.read_text('utf-8').splitlines() if ln.strip())
        print(f"  [monthly-save] {_month_idx.name}  ({_cnt} records)")

    # Merge daily JSONL indexes into meme_index.jsonl.
    # Restrict the glob to date-formatted filenames to avoid meme_index_YYYY-MM.jsonl.
    with open(output_dir / 'meme_index.jsonl', 'w', encoding='utf-8') as out_f:
        for day_file in sorted(output_dir.glob('index_????-??-??.jsonl')):
            out_f.write(day_file.read_text('utf-8'))

    # Save statistics.
    total_img = total['image_replies']
    summary = {
        'date_range':          f"{dates[0]} ~ {dates[-1]}",
        'days_processed':      len([s for s in all_stats if not s.get('skipped')]),
        'lang_filter':         CONFIG['lang_filter'],
        'lang_filter_strict':  CONFIG['lang_filter_strict'],
        'total_image_replies': total_img,
        'total_meme':          total['meme'],
        'total_not_meme':      total['not_meme'],
        'total_dl_fail':       total['dl_fail'],
        'total_errors':        total['error'],
        'total_orig_imgs_saved': total['orig_imgs'],
        'overall_meme_ratio':  round(total['meme'] / total_img, 4) if total_img else 0,
        'per_day':             all_stats,
    }
    stats_path = output_dir / 'stats.json'
    stats_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8')

    # Final summary.
    print(f"\n{'='*60}")
    print(f"  DONE  ({dates[0]} ~ {dates[-1]})")
    print(f"  Total image replies:  {total_img:,}")
    print(f"  Total meme:           {total['meme']:,}  "
          f"({total['meme']/max(1,total_img)*100:.1f}%)")
    print(f"  Total not meme:       {total['not_meme']:,}")
    print(f"  Download failures:    {total['dl_fail']:,}")
    print(f"  Errors:               {total['error']:,}")
    print(f"  Orig post imgs saved: {total['orig_imgs']:,}")
    print(f"\n  Output: {output_dir.resolve()}")
    print(f"  ├─ meme_images/{{uid}}/{{cid}}.jpg")
    print(f"  ├─ original_post_images/{{uid}}/{{cid}}.jpg")
    print(f"  ├─ records/{{uid}}.json")
    print(f"  ├─ index_YYYY-MM-DD.jsonl  (daily)")
    print(f"  ├─ meme_index.jsonl        (merged)")
    print(f"  ├─ all_memes.json")
    print(f"  └─ stats.json              <- daily and aggregate statistics")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='Bluesky Meme Reply Detector v4')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--run',     action='store_true')

    # Date selection (choose one of these forms).
    p.add_argument('--date',  default=None, help='Single date, e.g. 2023-09-01')
    p.add_argument('--month', default=None, help='Full month, e.g. 2023-09')
    p.add_argument('--start', default=None, help='Start date, e.g. 2023-09-01')
    p.add_argument('--end',   default=None, help='End date, e.g. 2023-09-30')

    p.add_argument('--archive-base', default=None)
    p.add_argument('--model-dir',    default=None)
    p.add_argument('--output-dir',   default=None)
    p.add_argument('--single-fold',  type=int, default=None, choices=[1,2,3,4,5])
    p.add_argument('--no-tta',       action='store_true')
    p.add_argument('--threshold',    type=float, default=None)
    p.add_argument('--preview-n',    type=int, default=10)
    p.add_argument('--lang',         nargs='+', default=None,
                   help='Language filter (default: en). Examples: --lang en, --lang en ko, --lang all')
    p.add_argument('--lang-strict',  action='store_true',
                   help='Exclude posts with langs=null or empty langs.')
    p.add_argument('--monthly-quota', type=int, default=5000,
                   help='Monthly meme collection target (default: 5000). Use 0 for unlimited.')
    p.add_argument('--daily-sample', type=int, default=None,
                   help='Maximum image replies to process per day. Example: --daily-sample 3000')
    p.add_argument('--lookup-days', type=int, default=None,
                   help='Prior days to search for root posts (default: 0 = TID estimate only). Example: --lookup-days 3')
    args = p.parse_args()

    if args.archive_base: CONFIG['archive_base']    = args.archive_base
    if args.model_dir:    CONFIG['model_dir']       = args.model_dir
    if args.output_dir:   CONFIG['output_dir']      = args.output_dir
    if args.single_fold:  CONFIG['single_fold']     = args.single_fold
    if args.no_tta:       CONFIG['use_tta']         = False
    if args.threshold:    CONFIG['meme_threshold']  = args.threshold
    if args.lang:
        CONFIG['lang_filter'] = [] if args.lang == ['all'] else args.lang
    if args.lang_strict:  CONFIG['lang_filter_strict'] = True
    if args.daily_sample: CONFIG['daily_sample_size']       = args.daily_sample
    if args.lookup_days is not None: CONFIG['lookup_max_days_back'] = args.lookup_days

    lf = CONFIG['lang_filter']
    print(f"Language filter: {lf if lf else 'none (all languages)'}"
          f"  strict={CONFIG['lang_filter_strict']}")

    dates = parse_date_range(args)
    print(f"Date range: {dates[0]} ~ {dates[-1]}  ({len(dates)} days)")

    if args.preview:
        run_preview(dates, n_per_day=args.preview_n)
    else:
        run_full(dates, monthly_quota=args.monthly_quota)


if __name__ == '__main__':
    main()
