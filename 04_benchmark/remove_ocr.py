"""
HTML에서 OCR 행만 제거하는 스크립트

Usage:
  python remove_ocr.py input.html
  python remove_ocr.py input.html -o output.html
"""

import argparse
from pathlib import Path


def remove_ocr(input_path: Path, output_path: Path):
    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)

    removed = 0
    result = []
    for line in lines:
        stripped = line.strip()
        # <div class="ocr-row"> 로 시작하는 줄 제거
        if stripped.startswith('<div class="ocr-row">'):
            removed += 1
            continue
        result.append(line)

    output_path.write_text("".join(result), encoding="utf-8")
    print(f"제거된 OCR 행: {removed}개")
    print(f"저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HTML에서 OCR 행 제거")
    parser.add_argument("input", help="입력 HTML 파일 경로")
    parser.add_argument("-o", "--output", default=None, help="출력 파일 경로 (기본: input_no_ocr.html)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {input_path}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_no_ocr.html")

    remove_ocr(input_path, output_path)


if __name__ == "__main__":
    main()
