#!/usr/bin/env python3
"""
기존 라벨링 데이터 보정 스크립트

아카이브(gzip) 대신 이미 수집된 records/*.json을 인덱싱해서 사용.

Fix 1: 대댓글 상위 체인 — parent_reply 위의 조상 체인 추가
       (parent_reply가 있는 케이스만 대상)
Fix 2: 인용 포스트 내용 — original_post의 embed에서 인용 포스트 추출
       (embed 정보가 meme_pipeline에서 저장 안 됐으므로 records에서 보완)

사용법:
  # 테스트 (파일 수정 없음)
  python patch_records.py --dry-run --limit 10

  # 본 실행
  python patch_records.py

  # 백그라운드
  nohup python -u patch_records.py > patch.log 2>&1 &
"""

from __future__ import annotations

import json
import hashlib
import shutil
import time
import argparse
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote

# ── CONFIG ──────────────────────────────────────────────────────
LABELED_JSONL = "./labeled_final/labeled_memes.jsonl"

DATASET_DIRS = [
    "../01_collection/meme_dataset_24_06",
    "../01_collection/meme_dataset_25_02",
    "../01_collection/meme_dataset",
]

MAX_CHAIN_DEPTH = 10


# ════════════════════════════════════════════════════════════════
#  URI 인덱스 구축
# ════════════════════════════════════════════════════════════════

def build_uri_index(dataset_dirs: list[str]) -> dict[str, dict]:
    """
    모든 dataset_dir/records/*.json을 읽어서
    {uri: sub_record_dict} 인덱스 반환.
    """
    index: dict[str, dict] = {}
    total_files = 0

    for d in dataset_dirs:
        records_dir = Path(d) / "records"
        if not records_dir.exists():
            print(f"  [SKIP] {records_dir} 없음")
            continue
        files = list(records_dir.glob("*.json"))
        print(f"  {records_dir}: {len(files):,}개")
        total_files += len(files)
        for fp in files:
            try:
                r = json.loads(fp.read_text(encoding="utf-8"))
                for field in ["meme_reply", "original_post", "parent_reply",
                              "closest_text_reply", "best_reply_before_meme"]:
                    sub = r.get(field) or {}
                    uri = sub.get("uri", "")
                    if uri and uri not in index:
                        index[uri] = sub
            except Exception:
                pass

    print(f"  총 {total_files:,}개 파일 → {len(index):,}개 URI 인덱싱 완료")
    return index


# ════════════════════════════════════════════════════════════════
#  embed 파싱
# ════════════════════════════════════════════════════════════════

def uri_to_uid(uri: str) -> str:
    try:
        parts = uri.replace("at://", "").split("/")
        did_suffix = parts[0].split(":")[-1][-14:]
        rkey = parts[-1]
        return f"bsky_{did_suffix}_{rkey}"
    except Exception:
        return "bsky_" + hashlib.sha256(uri.encode()).hexdigest()[:22]


def uri_to_did(uri: str) -> str:
    try:
        return uri.replace("at://", "").split("/")[0]
    except Exception:
        return ""


def _parse_images(embed: dict, did: str) -> list[dict]:
    if not isinstance(embed, dict):
        return []
    t = embed.get("$type", "")
    if t == "app.bsky.embed.recordWithMedia":
        embed = embed.get("media") or {}
    raw = embed.get("images", []) if isinstance(embed, dict) else []
    out = []
    for img in raw:
        alt  = img.get("alt", "")
        blob = img.get("image", {})
        ref  = blob.get("ref", "")
        if isinstance(ref, dict):
            cid = ref.get("$link", "")
        elif ref:
            cid = str(ref)
        else:
            cid = blob.get("cid", "")
        if cid:
            out.append({
                "cid": cid,
                "alt": alt,
                "source_url": f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg",
            })
    return out


def _parse_external(embed: dict) -> dict:
    if not isinstance(embed, dict):
        return {}
    t = embed.get("$type", "")
    if t in ("app.bsky.embed.external", "app.bsky.embed.external#view"):
        ext = embed.get("external", {})
    elif t in ("app.bsky.embed.recordWithMedia", "app.bsky.embed.recordWithMedia#view"):
        ext = (embed.get("media") or {}).get("external", {})
    else:
        return {}
    if not ext:
        return {}
    return {
        "external_title":       ext.get("title"),
        "external_url":         ext.get("uri"),
        "external_description": ext.get("description"),
    }


def _extract_quoted(embed: dict, uri_index: dict) -> dict | None:
    if not isinstance(embed, dict):
        return None
    t = embed.get("$type", "")

    if t in ("app.bsky.embed.recordWithMedia", "app.bsky.embed.recordWithMedia#view"):
        return _extract_quoted(embed.get("record", {}), uri_index)

    if t in ("app.bsky.embed.record", "app.bsky.embed.record#view"):
        record = embed.get("record", {})
        if not isinstance(record, dict):
            return None

        uri  = record.get("uri", "")
        did  = uri_to_did(uri)
        text = (record.get("value") or {}).get("text", "") or record.get("text", "")

        inner_embed = (record.get("value") or {}).get("embed") or record.get("embed")
        images   = _parse_images(inner_embed, did) if inner_embed else []
        external = _parse_external(inner_embed) if inner_embed else {}

        # uri_index에서 보완
        indexed = uri_index.get(uri, {})
        if not text:
            text = indexed.get("text", "") or ""
        if not images and indexed.get("images"):
            images = indexed["images"]

        # 아직 텍스트/이미지가 없으면 인용된 포스트 URI로 API 추가 조회
        if uri and not text and not images and not external:
            api_data = _fetch_post_thread(uri, depth=0)
            if api_data:
                api_post   = (api_data.get("thread") or {}).get("post") or {}
                api_record = api_post.get("record") or {}
                if not text:
                    text = api_record.get("text", "") or ""
                api_embed = api_record.get("embed") or api_post.get("embed")
                if api_embed and isinstance(api_embed, dict):
                    if not images:
                        images = _parse_images(api_embed, did)
                    if not external:
                        external = _parse_external(api_embed)

        result = {
            "uri":    uri,
            "uid":    uri_to_uid(uri),
            "did":    did,
            "text":   text,
            "images": images,
        }
        result.update(external)
        return result

    return None


# ════════════════════════════════════════════════════════════════
#  Bluesky API: getPostThread (인증 불필요, public)
# ════════════════════════════════════════════════════════════════

_API_BASE = "https://public.api.bsky.app/xrpc"
_api_cache: dict[str, dict] = {}  # uri → thread response


def _fetch_post_thread(uri: str, depth: int = 0) -> dict | None:
    """Bluesky public API로 포스트 thread 조회. 실패 시 None."""
    if uri in _api_cache:
        return _api_cache[uri]
    try:
        encoded = quote(uri, safe="")
        url = f"{_API_BASE}/app.bsky.feed.getPostThread?uri={encoded}&depth={depth}"
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        _api_cache[uri] = data
        return data
    except Exception:
        _api_cache[uri] = None
        return None


def _extract_embed_from_api(uri: str, uri_index: dict) -> dict | None:
    """API로 포스트 가져와서 embed → quoted_post 추출."""
    data = _fetch_post_thread(uri, depth=0)
    if not data:
        return None
    post = (data.get("thread") or {}).get("post") or {}
    record = post.get("record") or {}
    embed  = record.get("embed") or post.get("embed")
    if not embed or not isinstance(embed, dict):
        return None
    return _extract_quoted(embed, uri_index)


# ════════════════════════════════════════════════════════════════
#  Fix 1: ancestor chain
# ════════════════════════════════════════════════════════════════

def build_ancestor_chain(parent_uri: str, root_uri: str,
                         uri_index: dict) -> list[dict]:
    chain   = []
    cur_uri = parent_uri
    visited = {cur_uri}

    for _ in range(MAX_CHAIN_DEPTH):
        cur = uri_index.get(cur_uri, {})
        if not cur:
            break
        anc_uri = cur.get("parent_uri", "")
        if not anc_uri or anc_uri in visited:
            break
        if anc_uri == root_uri:
            break
        visited.add(anc_uri)
        anc = uri_index.get(anc_uri)
        if not anc:
            chain.append({"uri": anc_uri, "uid": uri_to_uid(anc_uri),
                          "text": None, "in_archive": False})
            cur_uri = anc_uri
            continue

        entry = {
            "uri":        anc_uri,
            "uid":        uri_to_uid(anc_uri),
            "did":        anc.get("did", ""),
            "text":       anc.get("text", ""),
            "created_at": anc.get("created_at", ""),
            "images":     anc.get("images", []),
            "in_archive": True,
        }

        # embed가 로컬에 없으면 API로 보완 (이미지/링크/인용 포스트)
        embed = anc.get("embed")
        if embed:
            entry.update(_parse_external(embed))
            quoted = _extract_quoted(embed, uri_index)
            if quoted:
                entry["quoted_post"] = quoted
        else:
            # API로 embed 조회
            api_data = _fetch_post_thread(anc_uri, depth=0)
            if api_data:
                post   = (api_data.get("thread") or {}).get("post") or {}
                record = post.get("record") or {}
                api_embed = record.get("embed") or post.get("embed")
                if api_embed and isinstance(api_embed, dict):
                    did = anc.get("did", uri_to_did(anc_uri))
                    # 이미지
                    api_images = _parse_images(api_embed, did)
                    if api_images and not entry["images"]:
                        entry["images"] = api_images
                    # 외부 링크
                    entry.update(_parse_external(api_embed))
                    # 인용 포스트
                    quoted = _extract_quoted(api_embed, uri_index)
                    if quoted:
                        entry["quoted_post"] = quoted

        chain.append(entry)
        cur_uri = anc_uri

    chain.reverse()
    return chain


def patch_ancestor_chain(record: dict, uri_index: dict) -> bool:
    if "ancestor_chain" in record:
        return False
    parent = record.get("parent_reply")
    if not parent:
        return False
    parent_uri = parent.get("uri", "")
    root_uri   = (record.get("original_post") or {}).get("uri", "")
    if not parent_uri or not root_uri:
        return False
    record["ancestor_chain"] = build_ancestor_chain(parent_uri, root_uri, uri_index)
    return True


# ════════════════════════════════════════════════════════════════
#  Fix 2: quoted post
# ════════════════════════════════════════════════════════════════

def patch_quoted_post(record: dict, uri_index: dict, use_api: bool = True) -> bool:
    if "quoted_post" in record:
        return False
    orig_uri = (record.get("original_post") or {}).get("uri", "")
    if not orig_uri:
        return False

    # 1순위: uri_index에서 embed 확인 (로컬)
    indexed = uri_index.get(orig_uri, {})
    embed = indexed.get("embed")
    if embed:
        quoted = _extract_quoted(embed, uri_index)
        if quoted:
            record["quoted_post"] = quoted
            return True

    # 2순위: Bluesky public API로 조회
    if use_api:
        quoted = _extract_embed_from_api(orig_uri, uri_index)
        if quoted:
            record["quoted_post"] = quoted
            return True

    return False


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Patch labeled_memes.jsonl")
    parser.add_argument("--jsonl",   default=LABELED_JSONL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--no-api",  action="store_true",
                        help="API 조회 없이 로컬 index만 사용 (Fix2 비활성)")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[ERROR] {jsonl_path} 없음")
        return

    # Step 1: 인덱스 구축
    print("[INDEX] 수집된 records 인덱싱 중...")
    t_idx = time.time()
    uri_index = build_uri_index(DATASET_DIRS)
    print(f"  인덱싱 완료: {time.time()-t_idx:.1f}s\n")

    # Step 2: 로드
    print(f"[LOAD] {jsonl_path.name}...")
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    if args.limit:
        records = records[:args.limit]
    print(f"  {len(records):,}개 로드\n")

    # Step 3: 패치
    print(f"[PATCH] 시작 (dry_run={args.dry_run})")
    stats = {"chain_patched": 0, "chain_skipped": 0,
             "quote_patched": 0, "quote_skipped": 0}
    t0 = time.time()

    for i, record in enumerate(records):
        uid = record.get("uid", "?")[:32]
        print(f"  [{i+1}/{len(records)}] {uid}", end="")

        print(" | Fix1:", end="", flush=True)
        if patch_ancestor_chain(record, uri_index):
            n = len(record.get("ancestor_chain", []))
            print(f" +{n}개", end="")
            stats["chain_patched"] += 1
        else:
            print(" skip", end="")
            stats["chain_skipped"] += 1

        print(" | Fix2:", end="", flush=True)
        if patch_quoted_post(record, uri_index, use_api=not args.no_api):
            txt = (record.get("quoted_post") or {}).get("text", "")[:20]
            print(f" found ({txt!r})", end="")
            stats["quote_patched"] += 1
        else:
            print(" none", end="")
            stats["quote_skipped"] += 1

        elapsed = time.time() - t0
        per     = elapsed / (i + 1)
        remain  = per * (len(records) - i - 1)
        print(f" | {elapsed:.0f}s ~{remain:.0f}s left")

    # Step 4: 저장
    if not args.dry_run:
        backup = jsonl_path.with_suffix(".jsonl.bak")
        shutil.copy2(jsonl_path, backup)
        print(f"\n[BACKUP] {backup}")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"[SAVED] {jsonl_path}")

    total_elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  Total:         {len(records):,}")
    print(f"  Chain patched: {stats['chain_patched']:,}")
    print(f"  Chain skipped: {stats['chain_skipped']:,}")
    print(f"  Quote patched: {stats['quote_patched']:,}")
    print(f"  Quote skipped: {stats['quote_skipped']:,}")
    print(f"  Elapsed:       {total_elapsed:.1f}s")
    print(f"  Dry run:       {args.dry_run}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
