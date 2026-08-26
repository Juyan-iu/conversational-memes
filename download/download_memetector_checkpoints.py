#!/usr/bin/env python3
"""
Download the five MemeTector v4 checkpoint files used by the filtering step.

The files are hosted on Google Drive and are saved as:

  ../02_meme_classification/checkpoints/fold1_best.pth
  ...
  ../02_meme_classification/checkpoints/fold5_best.pth

Usage:

  python download_memetector_checkpoints.py

  python download_memetector_checkpoints.py \
    --out-dir ../02_meme_classification/checkpoints \
    --overwrite
"""

from __future__ import annotations

import argparse
import html
import http.cookiejar
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


DEFAULT_CHECKPOINTS = [
    {
        "name": "fold1_best.pth",
        "url": "https://drive.google.com/file/d/1EktgLNayP4rQRaKdEbKqBTZ6yX84Id-o/view?usp=sharing",
    },
    {
        "name": "fold2_best.pth",
        "url": "https://drive.google.com/file/d/1dUpQqmv8NbmIEgzHCdYjcMeSiNd63sry/view?usp=sharing",
    },
    {
        "name": "fold3_best.pth",
        "url": "https://drive.google.com/file/d/1TV8ZTD9J4TQGFO0RyZhA89gcJutmDRxf/view?usp=sharing",
    },
    {
        "name": "fold4_best.pth",
        "url": "https://drive.google.com/file/d/1Z24E-ZGxigDrJgD12-_yvUCRalyZOJTM/view?usp=sharing",
    },
    {
        "name": "fold5_best.pth",
        "url": "https://drive.google.com/file/d/1wg2QHh0qxSrXpMP1pAvCORkQxUoEw9iy/view?",
    },
]

DEFAULT_OUT_DIR = "../02_meme_classification/checkpoints"
USER_AGENT = "ConversationalMemeCheckpointDownloader/1.0"


def extract_file_id(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"[-_A-Za-z0-9]{20,}", value):
        return value

    parsed = urlparse(value)
    query_id = parse_qs(parsed.query).get("id", [None])[0]
    if query_id:
        return query_id

    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        return match.group(1)

    raise ValueError(f"Could not parse Google Drive file id from: {value}")


def drive_download_url(file_id: str, confirm: str | None = None) -> str:
    params = {"export": "download", "id": file_id}
    if confirm:
        params["confirm"] = confirm
    return "https://drive.google.com/uc?" + urlencode(params)


def request(url: str):
    return Request(url, headers={"User-Agent": USER_AGENT})


def confirm_token_from_cookies(cookie_jar: http.cookiejar.CookieJar) -> str | None:
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            return cookie.value
    return None


def confirm_url_from_html(text: str, file_id: str) -> str | None:
    decoded = html.unescape(text)
    decoded = (
        decoded
        .replace("\\u003d", "=")
        .replace("\\u0026", "&")
        .replace("\\/", "/")
    )

    direct_match = re.search(
        r"https://(?:drive\.google\.com|drive\.usercontent\.google\.com)/(?:uc|download)\?[^\"'<>\\\s]+",
        decoded,
    )
    if direct_match:
        url = direct_match.group(0)
        if "confirm=" in url or "uuid=" in url:
            return url

    form_match = re.search(r"<form\b[^>]*\bid=[\"']download-form[\"'][^>]*>.*?</form>", decoded, re.S | re.I)
    if not form_match:
        form_match = re.search(r"<form\b[^>]*>.*?</form>", decoded, re.S | re.I)

    if form_match:
        form = form_match.group(0)
        action_match = re.search(r"\baction=[\"']([^\"']+)[\"']", form, re.I)
        if action_match:
            action = urljoin("https://drive.google.com", action_match.group(1))
            params = {"id": file_id, "export": "download"}
            for input_tag in re.findall(r"<input\b[^>]*>", form, re.I):
                name_match = re.search(r"\bname=[\"']([^\"']+)[\"']", input_tag, re.I)
                value_match = re.search(r"\bvalue=[\"']([^\"']*)[\"']", input_tag, re.I)
                if name_match:
                    params[name_match.group(1)] = value_match.group(1) if value_match else ""
            if "download" in action or "confirm" in params or "uuid" in params:
                return action + ("&" if "?" in action else "?") + urlencode(params)

    hidden_confirm = re.search(
        r"\bname=[\"']confirm[\"'][^>]*\bvalue=[\"']([^\"']+)[\"']",
        decoded,
        re.I,
    )
    if hidden_confirm:
        return drive_download_url(file_id, confirm=hidden_confirm.group(1))

    match = re.search(r"confirm=([0-9A-Za-z_-]+)", decoded)
    if match:
        return drive_download_url(file_id, confirm=match.group(1))

    match = re.search(r'href="([^"]*?(?:/uc\?|/download\?)[^"]+)"', decoded)
    if match:
        href = match.group(1)
        if href.startswith("/"):
            href = "https://drive.google.com" + href
        return href

    return None


def is_download_response(response) -> bool:
    disposition = response.headers.get("Content-Disposition", "")
    content_type = response.headers.get("Content-Type", "")
    if "attachment" in disposition.lower():
        return True
    if "text/html" in content_type.lower():
        return False
    return True


def stream_to_file(response, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = response.headers.get("Content-Length")
    total_int = int(total) if total and total.isdigit() else None
    written = 0
    next_report = 50 * 1024 * 1024

    with out_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if written >= next_report:
                if total_int:
                    print(f"[DOWNLOAD] {written / 1024 / 1024:.1f} MiB / {total_int / 1024 / 1024:.1f} MiB", flush=True)
                else:
                    print(f"[DOWNLOAD] {written / 1024 / 1024:.1f} MiB", flush=True)
                next_report += 50 * 1024 * 1024

    return written


def download_drive_file(file_id: str, out_path: Path, overwrite: bool, timeout: int) -> int:
    if out_path.exists() and not overwrite:
        print(f"[SKIP] {out_path} already exists. Pass --overwrite to replace it.")
        return 0

    cookie_jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))

    url = drive_download_url(file_id)
    response = opener.open(request(url), timeout=timeout)

    token = confirm_token_from_cookies(cookie_jar)
    if token:
        response.close()
        response = opener.open(request(drive_download_url(file_id, confirm=token)), timeout=timeout)
    elif not is_download_response(response):
        page = response.read().decode("utf-8", errors="replace")
        response.close()
        confirm_url = confirm_url_from_html(page, file_id)
        if not confirm_url:
            raise RuntimeError(f"Google Drive confirmation page could not be parsed for file id {file_id}.")
        response = opener.open(request(confirm_url), timeout=timeout)

    print(f"[SAVE] {out_path}", flush=True)
    written = stream_to_file(response, out_path)
    response.close()
    print(f"[DONE] wrote {written:,} bytes to {out_path}")
    return written


def looks_like_torch_checkpoint(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    head = path.read_bytes()[:16]
    if head.startswith(b"PK\x03\x04"):
        return True
    if head.startswith(b"\x80"):
        return True
    return False


def first_text_preview(path: Path) -> str:
    try:
        return path.read_bytes()[:160].decode("utf-8", errors="replace").replace("\n", "\\n")
    except Exception:
        return ""


def checkpoint_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.file_id and not args.url:
        return [
            {"name": spec["name"], "file_id": extract_file_id(spec["url"])}
            for spec in DEFAULT_CHECKPOINTS
        ]

    values = args.file_id or args.url
    if len(values) != 5:
        raise ValueError("Pass exactly five --file-id or --url values, one per fold.")

    return [
        {"name": f"fold{index}_best.pth", "file_id": extract_file_id(value)}
        for index, value in enumerate(values, start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MemeTector v4 checkpoints from Google Drive.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"Output directory. Default: {DEFAULT_OUT_DIR}")
    parser.add_argument("--file-id", action="append", default=[], help="Google Drive file id. Repeat exactly five times to override defaults.")
    parser.add_argument("--url", action="append", default=[], help="Google Drive sharing URL. Repeat exactly five times to override defaults.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing checkpoint files.")
    parser.add_argument("--no-verify", action="store_true", help="Skip torch-checkpoint magic-byte verification.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)

    try:
        specs = checkpoint_specs(args)
        print(
            "Downloading checkpoints in fold order:\n"
            + "\n".join(f"  {spec['name']}: {spec['file_id']}" for spec in specs)
        )
        failures = []
        for spec in specs:
            out_path = out_dir / spec["name"]
            written = download_drive_file(spec["file_id"], out_path, args.overwrite, args.timeout)
            if written == 0 and out_path.exists() and not args.overwrite:
                print(f"[VERIFY] existing {out_path}")

            if not args.no_verify and not looks_like_torch_checkpoint(out_path):
                preview = first_text_preview(out_path)
                failures.append(
                    {
                        "file": str(out_path),
                        "file_id": spec["file_id"],
                        "preview": preview,
                    }
                )
                print(f"[WARN] {out_path} does not look like a PyTorch checkpoint.")

        if failures:
            print("[ERROR] One or more downloaded files failed checkpoint verification.", file=sys.stderr)
            for failure in failures:
                print(f"  {failure['file']} ({failure['file_id']}) preview={failure['preview']!r}", file=sys.stderr)
            print("        If these are valid non-PyTorch checkpoint files, re-run with --no-verify.", file=sys.stderr)
            return 1

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[DONE] checkpoints are ready in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
