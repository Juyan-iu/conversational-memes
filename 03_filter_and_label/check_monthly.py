#!/usr/bin/env python3
"""
달별로 라벨링 결과 샘플 뽑아서 HTML 뷰어 생성

사용법:
  python check_monthly.py --month 2024-06          # 특정 달 20개
  python check_monthly.py --month 2024-06 --n 30   # 특정 달 30개
  python check_monthly.py --list                   # 전체 달 목록 확인
  python check_monthly.py --all                    # 전체 달 각 20개씩 HTML 생성
"""

import json
import argparse
import random
import subprocess
from pathlib import Path
from collections import defaultdict


DATASET_DIRS = [
    "../01_collection/meme_dataset_24_06",
    "../01_collection/meme_dataset_25_02",
    "../01_collection/meme_dataset",
]


def get_month(record: dict) -> str:
    from datetime import datetime, timezone
    # 1순위: original_post.created_at (unix timestamp or ISO)
    for field in ["created_at", "createdAt"]:
        val = (record.get("original_post") or {}).get(field, "")
        if not val:
            val = record.get(field, "")
        if not val:
            continue
        # unix timestamp
        try:
            ts = float(val)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.strftime("%Y-%m")
        except (ValueError, TypeError):
            pass
        # ISO string
        if len(str(val)) >= 7:
            return str(val)[:7]
    return "unknown"


def load_all() -> dict:
    """전체 레코드를 달별로 그룹화"""
    by_month = defaultdict(list)
    total = 0
    for d in DATASET_DIRS:
        p = Path(d) / "records"
        if not p.exists():
            continue
        files = list(p.glob("*.json"))
        for f in files:
            try:
                r = json.loads(f.read_text("utf-8"))
                by_month[get_month(r)].append(r)
                total += 1
            except Exception:
                pass
    print(f"총 {total:,}개 로드 완료")
    return by_month


def sample_month(by_month: dict, month: str, n: int) -> list:
    records = by_month.get(month, [])
    if not records:
        print(f"[{month}] 데이터 없음")
        return []
    sampled = random.sample(records, min(n, len(records)))
    print(f"[{month}] {len(records):,}개 중 {len(sampled)}개 샘플링")
    return sampled


def save_sample_jsonl(records: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"  저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="달별 라벨링 결과 확인")
    parser.add_argument("--month",  default=None, help="확인할 달 (예: 2024-06)")
    parser.add_argument("--n",      type=int, default=20, help="샘플 수 (기본: 20)")
    parser.add_argument("--list",   action="store_true", help="전체 달 목록 확인")
    parser.add_argument("--all",    action="store_true", help="전체 달 각 --n개씩 HTML 생성")
    parser.add_argument("--input",  default="./labeled_final/labeled_memes.jsonl",
                        help="라벨링 결과 jsonl 경로")
    parser.add_argument("--output-dir", default="./monthly_checks",
                        help="출력 폴더")
    args = parser.parse_args()

    # 라벨링된 결과 로드
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 라벨링 결과 없음: {input_path}")
        print("  --input 옵션으로 경로 지정해주세요")
        return

    print(f"\n라벨링 결과 로드: {input_path}")
    labeled = {}
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    uid = r.get("uid", "")
                    month = get_month(r)
                    if uid:
                        labeled[uid] = (month, r)
                except Exception:
                    pass

    # 달별 그룹화
    by_month = defaultdict(list)
    for uid, (month, r) in labeled.items():
        by_month[month].append(r)

    months = sorted(k for k in by_month.keys() if k != "unknown")
    print(f"총 {len(labeled):,}개 라벨링 완료 / {len(months)}개월")

    # --list: 달별 분포 출력
    if args.list:
        print(f"\n{'='*45}")
        print(f"  달별 라벨링 결과 분포")
        print(f"{'='*45}")
        for m in months:
            print(f"  {m}: {len(by_month[m]):,}개")
        if "unknown" in by_month:
            print(f"  unknown: {len(by_month['unknown']):,}개")
        print(f"{'='*45}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --all: 전체 달 HTML 생성
    if args.all:
        print(f"\n전체 {len(months)}개월 각 {args.n}개씩 HTML 생성...")
        for month in months:
            records = by_month[month]
            sampled = random.sample(records, min(args.n, len(records)))
            jsonl_path = output_dir / f"{month}.jsonl"
            save_sample_jsonl(sampled, jsonl_path)
            html_path = output_dir / f"{month}.html"
            subprocess.run([
                "python", "view_results.py",
                "--input", str(jsonl_path),
                "--output", str(html_path)
            ])
        print(f"\n완료! {output_dir}/ 에 HTML 파일 생성됨")
        print("월별 파일 목록:")
        for f in sorted(output_dir.glob("*.html")):
            print(f"  {f.name}")
        return

    # --month: 특정 달 HTML 생성
    if args.month:
        if args.month not in by_month:
            print(f"[ERROR] {args.month} 데이터 없음")
            print(f"  사용 가능한 달: {', '.join(months)}")
            return

        records = by_month[args.month]
        sampled = random.sample(records, min(args.n, len(records)))
        jsonl_path = output_dir / f"{args.month}.jsonl"
        save_sample_jsonl(sampled, jsonl_path)
        html_path = output_dir / f"{args.month}.html"
        subprocess.run([
            "python", "view_results.py",
            "--input", str(jsonl_path),
            "--output", str(html_path)
        ])
        print(f"\n완료! → {html_path}")
        return

    # 옵션 없으면 list 출력
    print("\n옵션을 지정해주세요:")
    print("  --list           달 목록 확인")
    print("  --month 2024-06  특정 달 확인")
    print("  --all            전체 달 HTML 생성")


if __name__ == "__main__":
    main()
