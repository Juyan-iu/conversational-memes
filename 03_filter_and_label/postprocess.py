#!/usr/bin/env python3
"""
labeled_final 후처리 스크립트

1. 영어 아닌 레코드 제거
2. 달별 현황 출력
3. 부족한 달 확인

사용법:
  python postprocess.py                          # 현황 확인만
  python postprocess.py --clean                  # 영어 필터 적용
  python postprocess.py --clean --replace        # 원본 파일 교체
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone


# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════
CONFIG = {
    "input":          "./labeled_final/labeled_memes.jsonl",
    "output":         "./labeled_final/labeled_memes_clean.jsonl",
    "english_threshold": 0.7,
    "monthly_target": 20400,  # 달별 850개 × 24개월
    # 수집 기간 고정 (23-09 ~ 25-08 = 24개월)
    "valid_months": [
        "2023-09", "2023-10", "2023-11", "2023-12",
        "2024-01", "2024-02", "2024-03", "2024-04",
        "2024-05", "2024-06", "2024-07", "2024-08",
        "2024-09", "2024-10", "2024-11", "2024-12",
        "2025-01", "2025-02", "2025-03", "2025-04",
        "2025-05", "2025-06", "2025-07", "2025-08",
    ],
}


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


# 유니코드 폰트 변형 영어 범위 (Bold/Italic 등)
# 𝐀-𝐙, 𝑨-𝒁, 𝓐-𝔃 등 → 실제로는 영어
_UNICODE_FONT_RANGES = [
    (0x1D400, 0x1D7FF),  # Mathematical Alphanumeric Symbols
    (0xFB00,  0xFB06),   # Alphabetic Presentation Forms
]
# 한국어처럼 보이는 유니코드 폰트 변형 범위 (혴혰혮 등 = Hangul 범위지만 폰트 변형)
# U+D400~D7FF 범위가 Hangul이지만 폰트 변형으로 쓰이는 경우 있음

def is_unicode_font_alpha(c: str) -> bool:
    """유니코드 폰트 변형 알파벳인지 확인"""
    cp = ord(c)
    for start, end in _UNICODE_FONT_RANGES:
        if start <= cp <= end:
            return True
    return False


def is_meaningless_text(text: str) -> bool:
    """
    유니코드 폰트 변형 영어만 있거나 이모지/기호만 있는 텍스트
    → 언어 감지 의미 없음, 통과
    """
    stripped = text.strip()
    if not stripped:
        return True
    # 실제 알파벳(ASCII) 5자 미만이고 유니코드 폰트 변형이 많으면 통과
    ascii_alpha = sum(1 for c in stripped if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in stripped if c.isalpha())
    if total_alpha < 5:
        return True
    # 유니코드 폰트 변형이 전체의 50% 이상이면 영어로 간주
    unicode_font = sum(1 for c in stripped if is_unicode_font_alpha(c))
    if unicode_font / max(len(stripped), 1) > 0.3:
        return True
    return False


# 명확히 차단할 언어 코드
_BLOCKED_LANGS = {
    "de",  # 독일어
    "fr",  # 프랑스어
    "ar",  # 아랍어
    "ru",  # 러시아어
    "es",  # 스페인어
    "it",  # 이탈리아어
    "nl",  # 네덜란드어
    "pt",  # 포르투갈어
    "pl",  # 폴란드어
    "tr",  # 터키어
    "ko",  # 한국어
    "ja",  # 일본어
    "zh-cn", "zh-tw",  # 중국어
}

def is_english(text: str, threshold: float = 0.7) -> bool:
    """
    명확한 비영어(독일어 등)만 차단
    짧은 텍스트, 슬랭, 이모지 등은 통과
    """
    if not text or not text.strip():
        return True
    if is_meaningless_text(text):
        return True
    # 텍스트가 너무 짧으면 (20자 미만) 오탐 많음 → 통과
    if len(text.strip()) < 20:
        return True
    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if not results:
            return True
        # 가장 높은 확률의 언어
        top = results[0]
        # 명확히 차단 언어이고 확률이 80% 이상일 때만 차단
        if top.lang in _BLOCKED_LANGS and top.prob >= 0.80:
            return False
        return True
    except Exception:
        return True


def english_ratio(text: str) -> float:
    """영어 비율 반환 (디버깅용)"""
    if not text or not text.strip():
        return 1.0
    ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha < 5:
        return 1.0
    return ascii_alpha / total_alpha


def has_mass_mention(text: str, threshold: int = 3) -> bool:
    mentions = re.findall(r"@[\w.]+", text or "")
    return len(mentions) >= threshold


def record_passes(record: dict, threshold: float = 0.7) -> tuple[bool, str]:
    """
    레코드 통과 여부 + 실패 이유 반환
    """
    orig_text   = (record.get("original_post") or {}).get("text") or ""
    parent_text = (record.get("parent_reply") or {}).get("text") or ""
    meme_text   = (record.get("meme_reply") or {}).get("text") or ""

    # 멘션 도배
    if has_mass_mention(orig_text):
        return False, "mass_mention"

    # 영어 필터 (원포스트 + 부모댓글 + 밈댓글 텍스트)
    for label, text in [("orig", orig_text), ("parent", parent_text), ("meme", meme_text)]:
        if text.strip() and not is_english(text, threshold):
            return False, f"non_english:{label}:{english_ratio(text):.2f}"

    return True, "ok"


# ════════════════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="labeled_final 후처리")
    parser.add_argument("--input",   default=CONFIG["input"])
    parser.add_argument("--output",  default=CONFIG["output"])
    parser.add_argument("--clean",   action="store_true",
                        help="영어/멘션 필터 적용해서 새 파일 생성")
    parser.add_argument("--replace", action="store_true",
                        help="--clean과 함께: 원본 파일을 clean 파일로 교체")
    parser.add_argument("--target",  type=int, default=CONFIG["monthly_target"],
                        help=f"월별 목표 개수 (기본: {CONFIG['monthly_target']})")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 파일 없음: {input_path}")
        return

    # 전체 로드
    print(f"[LOAD] {input_path}")
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    print(f"  총 {len(records):,}개 로드\n")

    # ── 현황 분석 ──────────────────────────────────────────────
    by_month = defaultdict(list)
    removed_by_reason = defaultdict(int)
    removed_by_month  = defaultdict(int)
    clean_records     = []

    for r in records:
        month = get_month(r)
        passes, reason = record_passes(r, CONFIG["english_threshold"])
        reason_key = reason.split(":")[0]  # non_english:orig:0.12 → non_english
        if passes:
            by_month[month].append(r)
            clean_records.append(r)
        else:
            removed_by_reason[reason_key] += 1
            removed_by_month[month]       += 1
            # 비영어 샘플 출력 (처음 20개)
            if reason_key == "non_english" and removed_by_reason["non_english"] <= 20:
                orig   = ((r.get("original_post") or {}).get("text") or "")[:80]
                parent = ((r.get("parent_reply") or {}).get("text") or "")[:80]
                meme   = ((r.get("meme_reply") or {}).get("text") or "")[:80]
                print(f"  [비영어#{removed_by_reason['non_english']}]")
                print(f"    orig  : {orig!r}")
                if parent: print(f"    parent: {parent!r}")
                if meme:   print(f"    meme  : {meme!r}")

    # 수집 기간 고정 (valid_months 기준)
    valid_months = CONFIG["valid_months"]
    months = valid_months  # 고정 12개월
    n_months = len(months)
    per_month = args.target // n_months  # 20000 / 24 = 833개/월

    # ── 출력 ───────────────────────────────────────────────────
    print(f"{'='*58}")
    print(f"  전체 현황")
    print(f"{'='*58}")
    print(f"  전체 레코드:       {len(records):,}개")
    print(f"  필터 통과:         {len(clean_records):,}개")
    print(f"  제거 (비영어):     {removed_by_reason['non_english']:,}개")
    print(f"  제거 (멘션도배):   {removed_by_reason['mass_mention']:,}개")
    print(f"  월 목표:           {per_month:,}개 ({args.target:,} / {n_months}개월)")

    print(f"\n{'='*58}")
    print(f"  달별 현황 (clean 기준)")
    print(f"{'='*58}")
    print(f"  {'월':12} {'보유':>8} {'목표':>8} {'부족':>8} {'제거':>8}")
    print(f"  {'-'*48}")

    total_short = 0
    shortage = {}
    for month in months:
        count   = len(by_month.get(month, []))
        removed = removed_by_month.get(month, 0)
        short   = max(0, per_month - count)
        total_short += short
        if short > 0:
            shortage[month] = short
        flag = " ⚠️" if short > 0 else " ✅"
        print(f"  {month:12} {count:>8,} {per_month:>8,} {short:>8,} {removed:>8,}{flag}")

    if "unknown" in by_month:
        print(f"  {'unknown':12} {len(by_month['unknown']):>8,}")

    print(f"  {'-'*48}")
    print(f"  {'합계':12} {len(clean_records):>8,} {args.target:>8,} {total_short:>8,}")

    # 부족한 달 명시
    if total_short > 0:
        print(f"\n{'='*58}")
        print(f"  추가 수집 필요")
        print(f"{'='*58}")
        for month, short in shortage.items():
            if short > 0:
                print(f"  {month}: {short:,}개 추가 필요")
        print(f"\n  총 {total_short:,}개 추가 필요")

        # shortage.json 저장
        shortage_path = Path(args.input).parent / "shortage.json"
        shortage_data = {
            "generated_at": datetime.now().isoformat(),
            "monthly_target": per_month,
            "total_short": total_short,
            "shortage": {m: s for m, s in shortage.items() if s > 0}
        }
        shortage_path.write_text(
            json.dumps(shortage_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\n[SAVE] shortage.json → {shortage_path}")
        print(f"\n  추가 수집 명령어:")
        print(f"  python refill_pipeline.py \\")
        print(f"    --shortage {shortage_path} \\")
        print(f"    --output ./labeled_final")
    else:
        print(f"\n  ✅ 모든 달 목표 달성!")

    # ── clean 파일 생성 ────────────────────────────────────────
    if args.clean:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in clean_records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"\n[SAVE] {output_path} ({len(clean_records):,}개)")

        if args.replace:
            backup = input_path.with_suffix(".backup.jsonl")
            input_path.rename(backup)
            output_path.rename(input_path)
            print(f"[REPLACE] 원본 → {backup}")
            print(f"[REPLACE] clean → {input_path}")


if __name__ == "__main__":
    main()
