#!/usr/bin/env python3
"""
달별 부족분 재수집 파이프라인

postprocess.py --clean 실행 후 생성된 shortage.json을 읽어
각 달별로 부족한 만큼만 정확히 수집해서 기존 jsonl에 append

사용법:
  python refill_pipeline.py \
    --shortage ./labeled_final/shortage.json \
    --output   ./labeled_final
"""

import os
import sys
import json
import random
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# label_pipeline의 함수들 재사용
sys.path.insert(0, str(Path(__file__).parent))
from label_pipeline import (
    CONFIG, COST,
    load_records, load_tree,
    process_record,
)
import re as _re

# ── 영어 감지 (postprocess.py와 동일 로직) ──────────────────────
_UNICODE_FONT_RANGES = [
    (0x1D400, 0x1D7FF),
    (0xFB00,  0xFB06),
]
_BLOCKED_LANGS = {
    "de", "fr", "ar", "ru", "es", "it",
    "nl", "pt", "pl", "tr", "ko", "ja", "zh-cn", "zh-tw",
}

def is_unicode_font_alpha(c: str) -> bool:
    cp = ord(c)
    for start, end in _UNICODE_FONT_RANGES:
        if start <= cp <= end:
            return True
    return False

def is_meaningless_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    total_alpha = sum(1 for c in stripped if c.isalpha())
    if total_alpha < 5:
        return True
    unicode_font = sum(1 for c in stripped if is_unicode_font_alpha(c))
    if unicode_font / max(len(stripped), 1) > 0.3:
        return True
    return False

def is_english(text: str, threshold: float = 0.7) -> bool:
    if not text or not text.strip():
        return True
    if is_meaningless_text(text):
        return True
    if len(text.strip()) < 20:
        return True
    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if not results:
            return True
        top = results[0]
        if top.lang in _BLOCKED_LANGS and top.prob >= 0.80:
            return False
    except Exception:
        pass
    return True

def has_mass_mention(text: str, threshold: int = 3) -> bool:
    mentions = _re.findall(r"@[\w.]+", text or "")
    return len(mentions) >= threshold

DATASET_DIRS = CONFIG["input_dirs"]


# ════════════════════════════════════════════════════════════════
#  유틸리티
# ════════════════════════════════════════════════════════════════

def get_month(record: dict) -> str:
    val = (record.get("original_post") or {}).get("created_at", "")
    if not val:
        val = record.get("created_at", "")
    if val:
        try:
            ts = float(val)
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
        except (ValueError, TypeError):
            if len(str(val)) >= 7:
                return str(val)[:7]
    return "unknown"


def load_existing_uids(jsonl_path: Path) -> set:
    uids = set()
    if not jsonl_path.exists():
        return uids
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    uids.add(json.loads(line).get("uid", ""))
                except Exception:
                    pass
    return uids


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="달별 부족분 재수집")
    parser.add_argument("--shortage", required=True,
                        help="postprocess.py가 생성한 shortage.json 경로")
    parser.add_argument("--output",   default="./labeled_final",
                        help="출력 폴더 (기존 labeled_memes.jsonl에 append)")
    parser.add_argument("--input",    default=None, nargs="+",
                        help="데이터 폴더 (기본: CONFIG input_dirs)")
    args = parser.parse_args()

    # shortage 로드
    shortage_path = Path(args.shortage)
    if not shortage_path.exists():
        print(f"[ERROR] shortage.json 없음: {shortage_path}")
        print("  먼저 python postprocess.py --clean 실행하세요")
        return

    shortage_data = json.loads(shortage_path.read_text(encoding="utf-8"))
    shortage      = shortage_data["shortage"]  # {month: count}
    monthly_target = shortage_data["monthly_target"]

    print(f"[LOAD] shortage.json")
    print(f"  생성일: {shortage_data['generated_at'][:16]}")
    print(f"  부족한 달: {len(shortage)}개월")
    print(f"  총 부족: {shortage_data['total_short']:,}개\n")

    # 출력 설정
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "labeled_memes.jsonl"

    # 기존 uid 수집 (중복 방지)
    print(f"[LOAD] 기존 uid 로드...")
    existing_uids = load_existing_uids(output_jsonl)
    print(f"  기존 처리된 uid: {len(existing_uids):,}개\n")

    # 트리 로드
    print("[LOAD] 트리 로드...")
    tree_3level = load_tree(CONFIG["tree_3level"])
    tree_2level = load_tree(CONFIG["tree_2level"])

    # 입력 폴더
    input_dirs = args.input if args.input else CONFIG["input_dirs"]

    # 전체 레코드 달별 그룹화
    print("[LOAD] 레코드 달별 그룹화...")
    all_records = load_records(input_dirs)
    print(f"  총 {len(all_records):,}개")

    by_month = defaultdict(list)
    for r in all_records:
        uid = r.get("uid", "")
        if uid in existing_uids:
            continue  # 이미 처리된 것 제외
        month = get_month(r)
        if month not in shortage:
            continue  # 부족하지 않은 달 제외
        # 빠른 필터만 (멘션 도배) - langdetect는 처리 시점에
        orig_text = (r.get("original_post") or {}).get("text") or ""
        if has_mass_mention(orig_text):
            continue
        by_month[month].append(r)

    # 달별로 셔플
    for m in by_month:
        random.shuffle(by_month[m])

    print(f"\n{'='*58}")
    print(f"  재수집 시작")
    print(f"{'='*58}")

    success_by_month = defaultdict(int)
    skipped_validation = 0
    total_success = 0

    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        for month, needed in sorted(shortage.items()):
            candidates = by_month.get(month, [])
            print(f"\n  [{month}] 필요: {needed:,}개 / 후보: {len(candidates):,}개")

            if not candidates:
                print(f"  [{month}] ⚠️ 후보 없음, 건너뜀")
                continue

            for record in candidates:
                if success_by_month[month] >= needed:
                    print(f"  [{month}] ✅ {needed:,}개 달성")
                    break

                uid = record.get("uid", "unknown")

                # 처리 시점 영어 필터 (langdetect)
                orig_text   = (record.get("original_post") or {}).get("text") or ""
                parent_text = (record.get("parent_reply") or {}).get("text") or ""
                meme_text   = (record.get("meme_reply") or {}).get("text") or ""
                if not all(is_english(t) for t in [orig_text, parent_text, meme_text] if t.strip()):
                    continue

                print(f"    [{month} {success_by_month[month]+1}/{needed}] {uid[:30]}")

                try:
                    result = process_record(
                        record, tree_3level, tree_2level, output_dir
                    )
                    if result is None:
                        skipped_validation += 1
                        continue

                    out_f.write(
                        json.dumps(result, ensure_ascii=False, default=str) + "\n"
                    )
                    out_f.flush()

                    # 개별 저장
                    record_path = output_dir / "records" / f"{uid}.json"
                    record_path.parent.mkdir(exist_ok=True)
                    record_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8"
                    )

                    existing_uids.add(uid)
                    success_by_month[month] += 1
                    total_success += 1

                except Exception as e:
                    import traceback
                    print(f"    [ERROR] {e}\n{traceback.format_exc()}")

            if success_by_month[month] < needed:
                print(f"  [{month}] ⚠️ 후보 소진: {success_by_month[month]}/{needed}개만 수집")

    # 결과 요약
    print(f"\n{'='*58}")
    print(f"  재수집 완료")
    print(f"{'='*58}")
    print(f"  성공:             {total_success:,}개")
    print(f"  검증 실패:        {skipped_validation:,}개")
    print(f"\n  달별 수집 결과:")
    for month in sorted(shortage.keys()):
        needed  = shortage[month]
        got     = success_by_month[month]
        flag    = "✅" if got >= needed else "⚠️"
        print(f"    {month}: {got:,}/{needed:,} {flag}")
    print(f"\n  출력: {output_dir.resolve()}")
    print(f"\n  [API 비용]")
    print(COST.summary())
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
