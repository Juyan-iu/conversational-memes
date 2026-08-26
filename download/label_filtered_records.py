#!/usr/bin/env python3
"""
Validate and label filtered conversational meme records.

This public script intentionally keeps only paper-facing fields:

- meme_validation
- stance_labels
- visual_description
- labeled_at

It does not emit discourse-function labels by default.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import mimetypes
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


POST_IMAGE_FIELDS = (
    "meme_reply",
    "original_post",
    "parent_reply",
    "quoted_post",
    "best_reply_before_meme",
    "closest_text_reply",
    "closest_sibling_text_reply",
    "comparison_reply",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_openai_client():
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - user environment dependent
        raise RuntimeError("Missing dependency: install the openai Python package.") from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


class CostTracker:
    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.calls += 1
        if not usage:
            return
        self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)

    def as_dict(self) -> dict[str, int]:
        return {
            "api_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def token_param_name(model: str) -> str:
    return "max_completion_tokens" if any(marker in model for marker in ("gpt-5", "o1", "o3")) else "max_tokens"


def call_model(
    client: Any,
    cost: CostTracker,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    sleep: float,
) -> str:
    if sleep:
        time.sleep(sleep)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        **{token_param_name(model): max_tokens},
    )
    cost.add(response)
    content = response.choices[0].message.content or ""
    return content.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


@dataclass
class RecordItem:
    record: dict[str, Any]
    source_path: Path | None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[RecordItem]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield RecordItem(json.loads(line), None)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc


def discover_records(input_path: Path) -> tuple[Iterable[RecordItem], Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".jsonl":
            return iter_jsonl(input_path), input_path.parent
        return [RecordItem(read_json(input_path), input_path)], input_path.parent

    if (input_path / "records").exists():
        records_dir = input_path / "records"
        source_root = input_path
    else:
        records_dir = input_path
        source_root = input_path.parent if input_path.name == "records" else input_path
    return (RecordItem(read_json(path), path) for path in sorted(records_dir.glob("*.json"))), source_root


def selected_items(items: Iterable[RecordItem], args: argparse.Namespace) -> Iterable[RecordItem]:
    wanted = set(args.uid or [])
    seen = 0
    yielded = 0
    for item in items:
        uid = str(item.record.get("uid", ""))
        if wanted and uid not in wanted:
            continue
        if not args.include_unpassed_filter and item.record.get("is_meme") is False:
            continue
        if seen < args.offset:
            seen += 1
            continue
        if args.limit is not None and yielded >= args.limit:
            break
        seen += 1
        yielded += 1
        yield item


def post_text(post: Any) -> str:
    if not isinstance(post, dict):
        return ""
    text = post.get("text")
    return text.strip() if isinstance(text, str) else ""


def build_context_text(record: dict[str, Any]) -> str:
    parts = []
    original = post_text(record.get("original_post"))
    quoted = post_text(record.get("quoted_post"))
    parent = post_text(record.get("parent_reply"))
    if original:
        parts.append(f"[Original Post] {original}")
    if quoted:
        parts.append(f"[Quoted Post] {quoted}")
    if parent:
        parts.append(f"[Parent Reply] {parent}")
    return "\n".join(parts)


def resolve_local_path(local_path: str | None, root: Path) -> Path | None:
    if not local_path:
        return None
    path = Path(local_path)
    return path if path.is_absolute() else root / path


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def meme_image_content(record: dict[str, Any], source_root: Path, allow_remote: bool) -> list[dict[str, Any]]:
    content = []
    for image in ((record.get("meme_reply") or {}).get("images") or []):
        path = resolve_local_path(image.get("local_path"), source_root)
        if path and path.exists():
            url = image_to_data_url(path)
        elif allow_remote and image.get("source_url"):
            url = image["source_url"]
        else:
            continue
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def normalize_validation(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "is_valid_meme": bool(raw.get("is_valid_meme")),
        "confidence": float(raw.get("confidence") or 0.0),
        "template_name": raw.get("template_name"),
        "reason": str(raw.get("reason") or ""),
    }


def validate_single_image(
    client: Any,
    cost: CostTracker,
    model: str,
    image_content: dict[str, Any],
    sleep: float,
) -> dict[str, Any]:
    prompt = (
        "Analyze this image and determine if it is an internet meme.\n\n"
        "A meme must meet ALL of the following criteria:\n"
        "1. Has visible text/caption overlaid ON the image itself.\n"
        "2. The text is part of the meme format, not just a watermark, logo, or news subtitle.\n"
        "3. Uses a recognizable meme format or template designed to be remixed or parodied.\n\n"
        "Answer false if there is no text, it is a regular photo/screenshot, or it is an ad/infographic.\n"
        "Text can be in any language. When in doubt, answer false.\n\n"
        "Respond ONLY with JSON:\n"
        '{"is_valid_meme": true, "confidence": 0.0, "template_name": "name or null", "reason": "one sentence"}\n'
        "The confidence value must be between 0.0 and 1.0."
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, image_content]}]
    try:
        return normalize_validation(parse_json_object(call_model(client, cost, model, messages, 180, sleep)))
    except Exception as exc:
        return {"is_valid_meme": False, "confidence": 0.0, "template_name": None, "reason": f"parse_or_api_failed: {exc}"}


def validate_record(
    client: Any,
    cost: CostTracker,
    model: str,
    images_content: list[dict[str, Any]],
    threshold: float,
    sleep: float,
) -> dict[str, Any]:
    if not images_content:
        return {
            "passed": False,
            "valid_ratio": 0.0,
            "threshold": threshold,
            "validations": [],
            "reason": "no meme image available",
        }

    validations = [validate_single_image(client, cost, model, image, sleep) for image in images_content]
    valid_count = sum(1 for row in validations if row.get("is_valid_meme"))
    ratio = valid_count / len(validations)
    passed = ratio >= threshold
    return {
        "passed": passed,
        "valid_ratio": round(ratio, 4),
        "threshold": threshold,
        "validations": validations,
        "reason": f"{valid_count}/{len(validations)} meme images valid",
    }


def classify_stance(
    client: Any,
    cost: CostTracker,
    model: str,
    record: dict[str, Any],
    images_content: list[dict[str, Any]],
    sleep: float,
) -> dict[str, bool]:
    context = build_context_text(record) or "(no textual context available)"
    utterance = post_text(record.get("meme_reply")) or "(no meme reply text)"
    prompt = (
        "Classify the stance of the current meme reply in its conversation context.\n\n"
        f"Previous Context:\n{context}\n\n"
        f"Current Meme Reply Text:\n{utterance}\n\n"
        "Return ONLY JSON with boolean values for these keys:\n"
        "- sarcastic: true if the utterance is sarcastic or ironic\n"
        "- humorous: true if it is humorous, funny, or playful\n"
        "- offensive: true if it attacks, demeans, threatens, or uses hateful/aggressive language\n\n"
        '{"sarcastic": false, "humorous": false, "offensive": false}'
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *images_content]}]
    try:
        raw = parse_json_object(call_model(client, cost, model, messages, 120, sleep))
        return {
            "sarcastic": bool(raw.get("sarcastic")),
            "humorous": bool(raw.get("humorous")),
            "offensive": bool(raw.get("offensive")),
        }
    except Exception:
        return {"sarcastic": False, "humorous": False, "offensive": False}


def describe_visual(
    client: Any,
    cost: CostTracker,
    model: str,
    record: dict[str, Any],
    images_content: list[dict[str, Any]],
    sleep: float,
) -> str | None:
    if not images_content:
        return None
    utterance = post_text(record.get("meme_reply"))
    prompt = (
        "Describe this meme image in ONE concise sentence that:\n"
        "1. Describes the visual elements, such as characters, objects, expressions, and setting.\n"
        "2. Includes what those visual elements symbolize or represent as a metaphor.\n"
        "3. Does NOT mention or quote any visible text/caption in the image.\n"
        "4. Is self-contained enough that someone could recreate the meme from the sentence.\n\n"
        f"Meme reply text, if any: {utterance}\n\n"
        "Reply with one sentence only. No quotes."
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, *images_content]}]
    try:
        return call_model(client, cost, model, messages, 120, sleep).strip().strip('"')
    except Exception:
        return None


def relative_image_destination(source: Path, role: str, uid: str) -> Path:
    parts = list(source.parts)
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[idx:])
    return Path("images") / role / uid / source.name


def copy_record_images(record: dict[str, Any], source_root: Path, out_root: Path) -> tuple[int, int]:
    copied = 0
    missing = 0
    uid = str(record.get("uid", "unknown"))
    for role in POST_IMAGE_FIELDS:
        post = record.get(role)
        if not isinstance(post, dict):
            continue
        for image in post.get("images") or []:
            local_path = image.get("local_path")
            source = resolve_local_path(local_path, source_root)
            if not source or not source.exists():
                if local_path:
                    image["source_local_path"] = local_path
                    image["copy_error"] = "local_image_missing"
                    missing += 1
                continue
            rel_dest = relative_image_destination(source, role, uid)
            dest = out_root / rel_dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != dest.resolve():
                shutil.copy2(source, dest)
            image["local_path"] = str(rel_dest)
            copied += 1
    return copied, missing


def write_record(record: dict[str, Any], out_dir: Path) -> Path:
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    path = records_dir / f"{record['uid']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and label filtered conversational meme records.")
    parser.add_argument("--input", default="filtered_records", help="Filtered output directory, records directory, JSON, or JSONL.")
    parser.add_argument("--out", default="labeled_records")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--model-visual", default=None, help="Optional model for image validation and visual description.")
    parser.add_argument("--validation-threshold", type=float, default=0.8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--uid", action="append", default=[], help="Label only this UID. Can repeat.")
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay between API calls.")
    parser.add_argument("--allow-remote-images", action="store_true", help="Use CDN image URLs if local images are missing.")
    parser.add_argument("--include-unpassed-filter", action="store_true", help="Also process records with is_meme=false.")
    parser.add_argument("--keep-invalid", action="store_true", help="Write records that fail GPT meme validation, with null labels.")
    parser.add_argument("--no-copy-images", action="store_true", help="Do not copy local images into the labeled output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        return 1

    try:
        client = load_openai_client()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    visual_model = args.model_visual or args.model
    raw_items, source_root = discover_records(input_path)
    items = selected_items(raw_items, args)
    cost = CostTracker()

    stats = {
        "selected": 0,
        "labeled": 0,
        "validation_failed": 0,
        "missing_images": 0,
        "failed": 0,
        "images_copied": 0,
        "images_missing_on_copy": 0,
        "model": args.model,
        "model_visual": visual_model,
        "validation_threshold": args.validation_threshold,
        "started_at": utc_now(),
        "finished_at": None,
    }
    failures: list[dict[str, Any]] = []

    labeled_jsonl = out_dir / "labeled_records.jsonl"
    skipped_jsonl = out_dir / "skipped_records.jsonl"

    with labeled_jsonl.open("w", encoding="utf-8") as labeled_handle, skipped_jsonl.open("w", encoding="utf-8") as skipped_handle:
        for item in items:
            stats["selected"] += 1
            record = item.record
            uid = record.get("uid", "unknown")
            print(f"[{stats['selected']}] {uid}")

            try:
                images_content = meme_image_content(record, source_root, args.allow_remote_images)
                if not images_content:
                    stats["missing_images"] += 1

                validation = validate_record(
                    client=client,
                    cost=cost,
                    model=visual_model,
                    images_content=images_content,
                    threshold=args.validation_threshold,
                    sleep=args.sleep,
                )

                if not validation["passed"]:
                    stats["validation_failed"] += 1
                    skipped = {
                        "uid": uid,
                        "uri": record.get("uri"),
                        "meme_validation": validation,
                        "skipped_at": utc_now(),
                    }
                    skipped_handle.write(json.dumps(skipped, ensure_ascii=False, sort_keys=True) + "\n")
                    if not args.keep_invalid:
                        continue

                out_record = copy.deepcopy(record)
                out_record["meme_validation"] = validation
                if validation["passed"]:
                    out_record["stance_labels"] = classify_stance(client, cost, args.model, record, images_content, args.sleep)
                    out_record["visual_description"] = describe_visual(client, cost, visual_model, record, images_content, args.sleep)
                else:
                    out_record["stance_labels"] = None
                    out_record["visual_description"] = None

                out_record["labeled_at"] = utc_now()
                out_record["label_metadata"] = {
                    "source_input": str(input_path),
                    "model": args.model,
                    "model_visual": visual_model,
                    "paper_facing_fields_only": True,
                    "discourse_labels_included": False,
                }
                if not args.no_copy_images:
                    copied, missing = copy_record_images(out_record, source_root, out_dir)
                    stats["images_copied"] += copied
                    stats["images_missing_on_copy"] += missing
                write_record(out_record, out_dir)
                labeled_handle.write(json.dumps(out_record, ensure_ascii=False, sort_keys=True) + "\n")
                stats["labeled"] += 1
            except Exception as exc:
                stats["failed"] += 1
                failures.append({"uid": uid, "error": str(exc)})
                print(f"  [FAIL] {exc}")

    stats["finished_at"] = utc_now()
    report = {"stats": {**stats, **cost.as_dict()}, "failures": failures}
    (out_dir / "label_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["stats"], indent=2))
    return 0 if stats["selected"] > 0 and stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
