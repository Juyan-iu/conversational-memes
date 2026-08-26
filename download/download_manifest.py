#!/usr/bin/env python3
"""
Download the released UID manifest for the collection-pool hydration step.

The manifest is hosted as a Google Drive file rather than committed to the
repository because it is about 100 MB.

Usage:

  python download_manifest.py

  python download_manifest.py \
    --out data/collection_pool_uid_manifest.jsonl \
    --overwrite
"""

from __future__ import annotations

import argparse
import html
import http.cookiejar
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


DEFAULT_MANIFEST_URL = "https://drive.google.com/file/d/1z-4NvEMU5asI0j5IX9dMahbVzzNmZ_Gd/view?usp=sharing"
DEFAULT_FILE_ID = "1z-4NvEMU5asI0j5IX9dMahbVzzNmZ_Gd"
DEFAULT_OUT = "data/collection_pool_uid_manifest.jsonl"
USER_AGENT = "ConversationalMemeManifestDownloader/1.0"


def extract_file_id(value: str | None) -> str:
    if not value:
        return DEFAULT_FILE_ID

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
    match = re.search(r"confirm=([0-9A-Za-z_-]+)", decoded)
    if match:
        return drive_download_url(file_id, confirm=match.group(1))

    match = re.search(r'href="([^"]*?/uc\?[^"]+)"', decoded)
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
    next_report = 10 * 1024 * 1024

    with out_path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            if written >= next_report:
                if total_int:
                    print(f"[DOWNLOAD] {written / 1024 / 1024:.1f} MiB / {total_int / 1024 / 1024:.1f} MiB", flush=True)
                else:
                    print(f"[DOWNLOAD] {written / 1024 / 1024:.1f} MiB", flush=True)
                next_report += 10 * 1024 * 1024

    return written


def download_manifest(file_id: str, out_path: Path, overwrite: bool) -> int:
    if out_path.exists() and not overwrite:
        print(f"[SKIP] {out_path} already exists. Pass --overwrite to replace it.")
        return 0

    cookie_jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))

    url = drive_download_url(file_id)
    response = opener.open(request(url), timeout=60)

    token = confirm_token_from_cookies(cookie_jar)
    if token:
        response.close()
        response = opener.open(request(drive_download_url(file_id, confirm=token)), timeout=60)
    elif not is_download_response(response):
        page = response.read().decode("utf-8", errors="replace")
        response.close()
        confirm_url = confirm_url_from_html(page, file_id)
        if not confirm_url:
            raise RuntimeError("Google Drive confirmation page could not be parsed.")
        response = opener.open(request(confirm_url), timeout=60)

    print(f"[SAVE] {out_path}", flush=True)
    written = stream_to_file(response, out_path)
    response.close()
    print(f"[DONE] wrote {written:,} bytes to {out_path}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the released UID manifest from Google Drive.")
    parser.add_argument("--url", default=DEFAULT_MANIFEST_URL, help="Google Drive sharing URL.")
    parser.add_argument("--file-id", default=None, help="Google Drive file id. Overrides --url.")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Output path. Default: {DEFAULT_OUT}")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output file if it already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        file_id = args.file_id or extract_file_id(args.url)
        written = download_manifest(file_id, Path(args.out), overwrite=args.overwrite)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0 if written >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
