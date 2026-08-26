#!/usr/bin/env python3
"""
Context Enrichment Pipeline v2 — TID 기반 타겟 스캔

labeled_memes.jsonl에서 누락된 두 필드를 보강한 새 jsonl을 생성한다.

1. ancestor_chain   — 밈이 re-reply일 때 parent → root 사이 댓글 체인
2. quoted_post      — original_post / ancestor 각 노드가 embed한 인용 포스트

[최적화]
- URI 29,414개의 TID → 예상 날짜 계산 후 날짜별 그룹핑
- 날짜 파일을 한 번만 열어서 필요한 URI 전부 수집 (반복 재탐색 없음)
- 못 찾은 URI만 ±N일 소급 탐색
- 찾은 포스트는 메모리 dict에 보관 → 보강 단계는 O(1) lookup

사용법:
  python enrich_context_pipeline.py                   # 전체 처리
  python enrich_context_pipeline.py --sample 200      # 테스트
  python enrich_context_pipeline.py --fallback-days 3 # 소급 일수 (기본 3)
  python enrich_context_pipeline.py --workers 8       # 파일 파싱 병렬화
"""

import os
import re
import sys
import json
import gzip
import argparse
import threading
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════

CONFIG = {
    "input_jsonl":   "../03_filter_and_label/labeled_final/labeled_memes.jsonl",
    "archive_base":  "/home/exouser/slate_project/bluesky/firehose_archives",
    "output_dir":    "./enriched_output",
    "output_name":   "labeled_memes_enriched.jsonl",
    "fallback_days": 3,
    "default_workers": 4,
}

# ════════════════════════════════════════════════════════════════
#  TID → date
# ════════════════════════════════════════════════════════════════

_TID_CHARS = '234567abcdefghijklmnopqrstuvwxyz'
_TID_MAP   = {c: i for i, c in enumerate(_TID_CHARS)}

def rkey_to_date(rkey: str) -> date | None:
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
        pass
    return None

def uri_to_rkey(uri: str) -> str:
    return uri.split('/')[-1] if uri else ''

def uri_to_did(uri: str) -> str:
    try:
        return uri.replace('at://', '').split('/')[0]
    except Exception:
        return ''

def _safe_uri(val):
    if isinstance(val, list):
        return val[0] if val else None
    if isinstance(val, str):
        return val or None
    return None

def archive_path(base: str, d: date) -> Path:
    return (Path(base) / f"{d.year}-{d.month:02d}"
            / f"{d.year}-{d.month:02d}-{d.day:02d}.json.gz")

# ════════════════════════════════════════════════════════════════
#  embed 파싱
# ════════════════════════════════════════════════════════════════

def parse_embed_images(embed, did: str) -> list:
    if not embed:
        return []

    def _from_dict(emb: dict) -> list:
        t = emb.get('$type', '')
        if 'recordWithMedia' in t:
            media = emb.get('media') or {}
            emb   = media if isinstance(media, dict) else {}
        raw = emb.get('images', []) if isinstance(emb, dict) else []
        imgs = []
        for img in raw:
            alt  = img.get('alt', '')
            blob = img.get('image', {})
            ref  = blob.get('ref', '')
            if isinstance(ref, dict):
                cid = ref.get('$link', '')
            elif ref:
                cid = str(ref)
            else:
                cid = blob.get('cid', '')
            if cid:
                imgs.append({
                    'cid': cid,
                    'alt': alt,
                    'url': f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg",
                })
        return imgs

    if isinstance(embed, dict):
        return _from_dict(embed)
    if isinstance(embed, str) and embed.strip().startswith('{'):
        import ast
        try:
            parsed = ast.literal_eval(embed)
            if isinstance(parsed, dict):
                return _from_dict(parsed)
        except Exception:
            pass
    return []


def parse_quoted_post(embed, did_fallback: str) -> dict | None:
    if not embed:
        return None

    def _extract(emb: dict) -> dict | None:
        t = emb.get('$type', '')
        if 'recordWithMedia' in t:
            rec_wrap = emb.get('record') or {}
            return _extract(rec_wrap)
        if 'embed.record' in t:
            inner  = emb.get('record') or {}
            value  = inner.get('value') or inner
            uri    = _safe_uri(inner.get('uri') or emb.get('uri'))
            text   = value.get('text') or inner.get('text') or ''
            author = inner.get('author') or {}
            q_did  = author.get('did') or (uri_to_did(uri) if uri else did_fallback)
            q_embed = value.get('embed') or inner.get('embed')
            images  = parse_embed_images(q_embed, q_did) if q_embed else []
            ext_title = ext_url = None
            if q_embed and isinstance(q_embed, dict):
                qt = q_embed.get('$type', '')
                if 'external' in qt:
                    ext = q_embed.get('external') or {}
                    ext_title = ext.get('title')
                    ext_url   = ext.get('uri')
            if uri or text:
                return {
                    'uri':            uri,
                    'text':           text,
                    'images':         images,
                    'external_title': ext_title,
                    'external_url':   ext_url,
                }
        return None

    if isinstance(embed, dict):
        return _extract(embed)
    if isinstance(embed, str) and embed.strip().startswith('{'):
        import ast
        try:
            parsed = ast.literal_eval(embed)
            if isinstance(parsed, dict):
                return _extract(parsed)
        except Exception:
            pass
    return None


def extract_external(embed) -> dict:
    if not embed or not isinstance(embed, dict):
        return {}
    t = embed.get('$type', '')
    if 'external' in t:
        ext = embed.get('external') or {}
        return {'external_title': ext.get('title'), 'external_url': ext.get('uri')}
    if 'recordWithMedia' in t:
        media = embed.get('media') or {}
        if isinstance(media, dict) and 'external' in media.get('$type', ''):
            ext = media.get('external') or {}
            return {'external_title': ext.get('title'), 'external_url': ext.get('uri')}
    return {}

# ════════════════════════════════════════════════════════════════
#  아카이브 post 파싱
# ════════════════════════════════════════════════════════════════

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
    if obj.get('type') != 'app.bsky.feed.post' or obj.get('action') == 'delete':
        return None
    uri     = obj.get('uri', '')
    did     = obj.get('author', '')
    created = obj.get('create_time') or obj.get('createdAt') or obj.get('commit_time', '')
    embed   = obj.get('embed') or (obj.get('record') or {}).get('embed')
    images  = parse_embed_images(embed, did) if embed else []

    _reply    = obj.get('reply')
    reply_ref = _reply if isinstance(_reply, dict) else {}
    _parent   = reply_ref.get('parent') or obj.get('parent')
    _root     = reply_ref.get('root')   or obj.get('root')
    parent_uri = _safe_uri(_parent.get('uri')) if isinstance(_parent, dict) else None
    root_uri   = _safe_uri(_root.get('uri'))   if isinstance(_root,   dict) else None

    return {
        'uri':        uri,
        'did':        did,
        'rkey':       uri.split('/')[-1],
        'text':       obj.get('text') or '',
        'created_at': created,
        'parent_uri': parent_uri,
        'root_uri':   root_uri,
        'images':     images,
        'embed':      embed,
        '_dt':        parse_datetime_safe(created),
    }

# ════════════════════════════════════════════════════════════════
#  STEP 1: 필요한 URI 수집 + 날짜별 그룹핑
# ════════════════════════════════════════════════════════════════

def collect_target_uris(records: list) -> dict:
    """반환: {uri → estimated_date | None}"""
    target_uris = {}
    for r in records:
        meme   = r.get('meme_reply') or {}
        orig   = r.get('original_post') or {}
        parent = r.get('parent_reply') or {}
        for uri in [
            meme.get('parent_uri'),
            meme.get('root_uri'),
            orig.get('uri'),
            parent.get('uri'),
        ]:
            if uri and uri not in target_uris:
                target_uris[uri] = rkey_to_date(uri_to_rkey(uri))
    print(f"[URI] 탐색 대상: {len(target_uris):,}개")
    return target_uris


def collect_intermediate_uris(records: list, uri_index: dict) -> dict:
    """
    1차 스캔 후 발견된 parent_uri 포스트들의 parent_uri (중간 체인 노드)를 추가 수집.
    반환: {uri → estimated_date | None}  (새로 발견된 것만)
    """
    new_uris = {}
    for r in records:
        meme       = r.get('meme_reply') or {}
        parent_uri = meme.get('parent_uri') or (r.get('parent_reply') or {}).get('uri')
        root_uri   = meme.get('root_uri') or (r.get('original_post') or {}).get('uri')
        if not parent_uri or not root_uri or parent_uri == root_uri:
            continue

        # parent_uri 포스트가 인덱스에 있으면 그 위 체인을 추적
        parent_post = uri_index.get(parent_uri)
        if not parent_post:
            continue

        current = parent_post.get('parent_uri')
        visited = set()
        while current and current != root_uri and len(visited) < 20:
            if current in visited:
                break
            visited.add(current)
            if current not in uri_index and current not in new_uris:
                new_uris[current] = rkey_to_date(uri_to_rkey(current))
            # 이미 인덱스에 있으면 그 위로 계속
            post = uri_index.get(current)
            if not post:
                break
            current = post.get('parent_uri')

    print(f"[URI] 중간 체인 노드 추가: {len(new_uris):,}개")
    return new_uris


def group_uris_by_date(target_uris: dict, fallback_days: int):
    """반환: by_date {date_str → set(uri)}, no_date set(uri)"""
    by_date = defaultdict(set)
    no_date = set()
    for uri, est_date in target_uris.items():
        if est_date:
            for delta in range(-fallback_days, fallback_days + 1):
                d_str = (est_date + timedelta(days=delta)).strftime('%Y-%m-%d')
                by_date[d_str].add(uri)
        else:
            no_date.add(uri)
    known = len(target_uris) - len(no_date)
    print(f"[URI] TID 추정: {known:,}개 → {len(by_date)}개 날짜 파일 | 날짜 불명: {len(no_date)}개")
    return by_date, no_date

# ════════════════════════════════════════════════════════════════
#  STEP 2: 아카이브 타겟 스캔
# ════════════════════════════════════════════════════════════════

def scan_archive_file(archive_file: Path, wanted_uris: set) -> dict:
    """단일 파일에서 wanted_uris에 해당하는 포스트만 추출."""
    found = {}
    if not archive_file.exists() or not wanted_uris:
        return found

    try:
        opener = gzip.open if str(archive_file).endswith('.gz') else open
        with opener(archive_file, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                # feed.post 아닌 줄은 스킵 (가장 빠른 필터)
                if 'feed.post' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get('type') != 'app.bsky.feed.post':
                    continue
                uri = obj.get('uri', '')
                if uri not in wanted_uris:
                    continue
                post = obj_to_post(obj)
                if post:
                    found[uri] = post
                    if len(found) == len(wanted_uris):
                        break
    except Exception as e:
        print(f"  [WARN] {archive_file.name}: {e}")
    return found


def _scan_one_process(args) -> dict:
    """멀티프로세스용 최상위 함수"""
    fp_str, uris_list = args
    return scan_archive_file(Path(fp_str), set(uris_list))


def build_uri_index(target_uris: dict, archive_base: str,
                    fallback_days: int, workers: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    by_date, no_date = group_uris_by_date(target_uris, fallback_days)

    date_files = []
    for d_str, uris in sorted(by_date.items()):
        try:
            d = date.fromisoformat(d_str)
        except Exception:
            continue
        fp = archive_path(archive_base, d)
        if fp.exists():
            date_files.append((str(fp), list(uris)))

    print(f"[SCAN] 대상 파일: {len(date_files)}개 (ProcessPool workers={workers})")

    uri_index = {}
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scan_one_process, item): item for item in date_files}
        for future in as_completed(futures):
            try:
                found = future.result()
            except Exception as e:
                print(f"  [WARN] 프로세스 오류: {e}")
                found = {}
            uri_index.update(found)
            done += 1
            if done % 5 == 0 or done == len(date_files):
                pct = done / len(date_files) * 100
                print(f"  [스캔] {done}/{len(date_files)} ({pct:.0f}%) | "
                      f"찾은 URI: {len(uri_index):,}개", flush=True)

    missing = set(target_uris.keys()) - set(uri_index.keys())
    print(f"\n[INDEX] {len(uri_index):,} / {len(target_uris):,}개 | 미발견: {len(missing):,}개")
    return uri_index

# ════════════════════════════════════════════════════════════════
#  STEP 3: 보강
# ════════════════════════════════════════════════════════════════

def post_to_node(post: dict) -> dict:
    images = [
        {'cid': img['cid'], 'alt': img.get('alt', ''), 'source_url': img['url']}
        for img in (post.get('images') or [])
    ]
    embed  = post.get('embed')
    did    = post.get('did', '')
    quoted = parse_quoted_post(embed, did)
    ext    = extract_external(embed)
    return {
        'uri':            post['uri'],
        'text':           post.get('text', ''),
        'created_at':     post.get('created_at'),
        'images':         images,
        'quoted_post':    quoted,
        'external_title': ext.get('external_title'),
        'external_url':   ext.get('external_url'),
        'in_archive':     True,
    }


def build_ancestor_chain(record: dict, uri_index: dict) -> list:
    """
    root_uri 바로 다음 댓글부터 parent_uri(밈 직접 부모) 바로 전까지의 중간 체인.
    parent_reply는 이미 레코드에 있으므로 chain에서 제외.

    탐색 방향: parent_uri → (parent_uri의 parent) → ... → root_uri 자식
    역순으로 수집 후 reverse → 오래된 순 반환.
    """
    meme       = record.get('meme_reply') or {}
    parent_uri = (meme.get('parent_uri')
                  or (record.get('parent_reply') or {}).get('uri'))
    root_uri   = (meme.get('root_uri')
                  or (record.get('original_post') or {}).get('uri'))

    if not parent_uri or not root_uri or parent_uri == root_uri:
        return []

    # parent_uri 포스트 자체는 parent_reply로 이미 레코드에 있음 → 제외
    # parent_uri의 parent부터 root 방향으로 올라가며 수집
    parent_post = uri_index.get(parent_uri)
    if not parent_post:
        # parent_uri 포스트를 인덱스에서 못 찾으면 체인 구성 불가
        return []

    chain   = []
    current = parent_post.get('parent_uri')  # parent_uri의 바로 위부터 시작
    visited = set()

    while current and current != root_uri and len(chain) < 20:
        if current in visited:
            break
        visited.add(current)
        post = uri_index.get(current)
        if not post:
            # 아카이브에 없는 노드 — stub으로 기록하고 중단
            chain.append({'uri': current, 'text': None, 'images': [], 'in_archive': False})
            break
        chain.append(post_to_node(post))
        current = post.get('parent_uri')

    chain.reverse()  # 오래된 순 (root에 가까운 순)
    return chain


def needs_ancestor_chain(record: dict) -> bool:
    structure  = record.get('thread_structure') or {}
    label      = structure.get('label') or ''
    meme       = record.get('meme_reply') or {}
    parent_uri = meme.get('parent_uri') or (record.get('parent_reply') or {}).get('uri')
    root_uri   = meme.get('root_uri')   or (record.get('original_post') or {}).get('uri')
    is_re_reply = (label in ('re-reply', 're-re-reply')
                   or (parent_uri and root_uri and parent_uri != root_uri))
    return is_re_reply and len(record.get('ancestor_chain') or []) == 0


def needs_quoted_post(record: dict) -> bool:
    return 'quoted_post' not in record


def enrich_record(record: dict, uri_index: dict) -> tuple[dict, dict]:
    uid      = record.get('uid', 'unknown')
    enriched = dict(record)
    stats    = {'uid': uid, 'added_ancestor_chain': False,
                'ancestor_chain_len': 0, 'added_quoted_post': False}

    # ancestor_chain
    if needs_ancestor_chain(record):
        chain = build_ancestor_chain(record, uri_index)
        enriched['ancestor_chain'] = chain
        stats['added_ancestor_chain'] = True
        stats['ancestor_chain_len']   = len(chain)
    elif 'ancestor_chain' not in enriched:
        enriched['ancestor_chain'] = []

    # quoted_post (original_post embed)
    if needs_quoted_post(record):
        orig_uri  = (record.get('original_post') or {}).get('uri')
        orig_post = uri_index.get(orig_uri) if orig_uri else None
        qp = parse_quoted_post(orig_post.get('embed'), orig_post.get('did', '')) if orig_post else None
        enriched['quoted_post'] = qp
        if qp:
            stats['added_quoted_post'] = True

    # ancestor_chain 노드 quoted_post 보강
    new_chain = []
    for node in (enriched.get('ancestor_chain') or []):
        if node.get('in_archive') and 'quoted_post' not in node:
            np = uri_index.get(node.get('uri', ''))
            if np:
                node = dict(node)
                node['quoted_post'] = parse_quoted_post(np.get('embed'), np.get('did', ''))
        new_chain.append(node)
    enriched['ancestor_chain'] = new_chain
    enriched['enriched_at'] = datetime.now(timezone.utc).isoformat()
    return enriched, stats

# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Context Enrichment Pipeline v2')
    parser.add_argument('--input',         default=CONFIG['input_jsonl'])
    parser.add_argument('--archive',       default=CONFIG['archive_base'])
    parser.add_argument('--output',        default=CONFIG['output_dir'])
    parser.add_argument('--sample',        type=int, default=None)
    parser.add_argument('--fallback-days', type=int, default=CONFIG['fallback_days'])
    parser.add_argument('--workers',       type=int, default=CONFIG['default_workers'])
    args = parser.parse_args()

    t0 = datetime.now()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / CONFIG['output_name']
    report_path = output_dir / 'enrich_report.json'

    # 1. 로드
    records = []
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    print(f"[LOAD] {len(records):,}개: {args.input}")
    if args.sample:
        records = records[:args.sample]
        print(f"[SAMPLE] {args.sample}개만 처리")

    print(f"\n[STATS] ancestor_chain 누락: "
          f"{sum(1 for r in records if needs_ancestor_chain(r)):,}개 | "
          f"quoted_post 누락: {sum(1 for r in records if needs_quoted_post(r)):,}개")

    # 2. URI 수집
    print(f"\n[STEP 1] 탐색 URI 수집...")
    target_uris = collect_target_uris(records)

    # 3. 아카이브 1차 스캔
    print(f"\n[STEP 2] 아카이브 1차 스캔 (workers={args.workers}, ±{args.fallback_days}일)...")
    uri_index = build_uri_index(target_uris, args.archive, args.fallback_days, args.workers)
    print(f"  1차 스캔 소요: {(datetime.now()-t0).seconds}초")

    # 3b. 중간 체인 노드 2차 스캔
    intermediate_uris = collect_intermediate_uris(records, uri_index)
    if intermediate_uris:
        print(f"\n[STEP 2b] 중간 체인 노드 2차 스캔 ({len(intermediate_uris):,}개)...")
        extra_index = build_uri_index(intermediate_uris, args.archive, args.fallback_days, args.workers)
        uri_index.update(extra_index)
        print(f"  2차 스캔 후 총 인덱스: {len(uri_index):,}개")
    print(f"  전체 스캔 소요: {(datetime.now()-t0).seconds}초")

    # 4. 보강
    print(f"\n[STEP 3] 보강 중...")
    all_stats, enriched_records = [], []
    for i, record in enumerate(records):
        enriched, stats = enrich_record(record, uri_index)
        enriched_records.append(enriched)
        all_stats.append(stats)
        if (i + 1) % 2000 == 0:
            print(f"  [{i+1:,}/{len(records):,}]")

    # 5. 저장
    print(f"\n[WRITE] {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in enriched_records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')

    # 6. 리포트
    added_chain  = sum(1 for s in all_stats if s.get('added_ancestor_chain'))
    added_quoted = sum(1 for s in all_stats if s.get('added_quoted_post'))
    avg_chain    = sum(s.get('ancestor_chain_len',0) for s in all_stats) / max(added_chain,1)
    elapsed      = (datetime.now() - t0).seconds

    report = {
        'total_input': len(records), 'total_output': len(enriched_records),
        'added_ancestor_chain': added_chain, 'added_quoted_post': added_quoted,
        'avg_ancestor_chain_len': round(avg_chain, 2),
        'uri_index_size': len(uri_index), 'target_uris': len(target_uris),
        'elapsed_seconds': elapsed, 'processed_at': datetime.now(timezone.utc).isoformat(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\n{'='*60}")
    print(f"  완료: {len(enriched_records):,}개  ({elapsed}초)")
    print(f"  ancestor_chain 추가: {added_chain:,}개 (평균 {avg_chain:.1f}개 노드)")
    print(f"  quoted_post 추가:    {added_quoted:,}개")
    print(f"  URI 인덱스:          {len(uri_index):,} / {len(target_uris):,}개")
    print(f"  출력: {output_path.resolve()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
