#!/usr/bin/env python3
"""
Filter hydrated conversational meme records with MemeTector v4.

This is the public reproduction filtering step after
hydrate_from_uid_manifest.py. It reads hydrated records, loads the trained
MemeTector checkpoints, scores the meme-reply image, and writes records that
pass the meme threshold.

Sample:

  python filter_with_memetector.py \
    --input hydrated_records \
    --model-dir ../02_meme_classification/checkpoints \
    --out filtered_records \
    --limit 10
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CLIP_NAME = "openai/clip-vit-large-patch14-336"
DEFAULT_OCR_TEXT = "an image"
IMG_SIZE = 336
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


def load_ml_dependencies():
    try:
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.transforms as T
        from torch.amp import autocast
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:  # pragma: no cover - user environment dependent
        raise RuntimeError(
            "Missing filtering dependencies. Install PyTorch, torchvision, "
            "transformers, Pillow, and numpy in the active environment."
        ) from exc

    return {
        "np": np,
        "Image": Image,
        "torch": torch,
        "nn": nn,
        "F": F,
        "T": T,
        "autocast": autocast,
        "CLIPModel": CLIPModel,
        "CLIPProcessor": CLIPProcessor,
    }


def build_model_classes(deps: dict[str, Any]):
    torch = deps["torch"]
    nn = deps["nn"]
    F = deps["F"]
    CLIPModel = deps["CLIPModel"]

    class GatedFusion(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.gate_net = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )
            self.ln_i = nn.LayerNorm(dim)
            self.ln_t = nn.LayerNorm(dim)

        def forward(self, image_features, text_features):
            gate = self.gate_net(torch.cat([image_features, text_features], dim=-1))
            fused = gate * self.ln_i(image_features) + (1 - gate) * self.ln_t(text_features)
            return fused, gate

    class MemeDetectorV4(nn.Module):
        def __init__(self, clip_name: str, dropout: float = 0.4):
            super().__init__()
            self.clip = CLIPModel.from_pretrained(clip_name)
            dim = self.clip.config.projection_dim
            self.fusion = GatedFusion(dim)
            self.classifier = nn.Sequential(
                nn.LayerNorm(dim * 3 + 1),
                nn.Linear(dim * 3 + 1, 768),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(768, 256),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(256, 2),
            )

        def encode_image(self, pixel_values):
            pooled = self.clip.vision_model(pixel_values=pixel_values, return_dict=True).pooler_output
            return self.clip.visual_projection(pooled)

        def encode_text(self, input_ids, attention_mask):
            pooled = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).pooler_output
            return self.clip.text_projection(pooled)

        def forward(self, pixel_values, input_ids, attention_mask):
            image_features = F.normalize(self.encode_image(pixel_values), dim=-1)
            text_features = F.normalize(self.encode_text(input_ids, attention_mask), dim=-1)
            cosine = (image_features * text_features).sum(-1, keepdim=True)
            fused, _ = self.fusion(image_features, text_features)
            return self.classifier(torch.cat([image_features, text_features, fused, cosine], dim=-1))

    return MemeDetectorV4


def build_transform(deps: dict[str, Any]):
    T = deps["T"]
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE))])


def select_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def checkpoint_paths(model_dir: Path, single_fold: int | None) -> list[Path]:
    if single_fold is not None:
        exact = model_dir / f"fold{single_fold}_best.pth"
        if exact.exists():
            return [exact]
        matches = sorted(model_dir.glob(f"*fold{single_fold}*.pth")) + sorted(model_dir.glob(f"*fold{single_fold}*.pt"))
        return matches[:1]

    fold_paths = [model_dir / f"fold{fold}_best.pth" for fold in range(1, 6)]
    existing = [path for path in fold_paths if path.exists()]
    if existing:
        return existing
    return sorted(model_dir.glob("*.pth")) + sorted(model_dir.glob("*.pt"))


def clean_state_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        state = raw.get("model_state_dict") or raw.get("state_dict") or raw
    else:
        state = raw
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state dict")
    return {key.replace("module.", "", 1): value for key, value in state.items()}


def load_models(
    deps: dict[str, Any],
    model_dir: Path,
    clip_name: str,
    device: str,
    single_fold: int | None,
) -> tuple[list[Any], list[str]]:
    torch = deps["torch"]
    MemeDetectorV4 = build_model_classes(deps)
    paths = checkpoint_paths(model_dir, single_fold)
    if not paths:
        raise FileNotFoundError(
            f"No checkpoint files found in {model_dir}. Expected fold1_best.pth ... fold5_best.pth "
            "or pass --model-dir to the trained checkpoint directory."
        )

    models = []
    for path in paths:
        model = MemeDetectorV4(clip_name)
        state = clean_state_dict(torch.load(path, map_location=device))
        model.load_state_dict(state)
        model.to(device).eval()
        models.append(model)
        print(f"[MODEL] loaded {path}")
    return models, [str(path) for path in paths]


def classify_batch(
    deps: dict[str, Any],
    pil_images: list[Any],
    processor: Any,
    models: list[Any],
    device: str,
    threshold: float,
    ocr_text: str,
) -> list[tuple[bool, float]]:
    np = deps["np"]
    torch = deps["torch"]
    autocast = deps["autocast"]
    transform = build_transform(deps)
    texts = [ocr_text.strip() or DEFAULT_OCR_TEXT for _ in pil_images]
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    all_probs = []
    for model in models:
        encoded = processor(
            images=[transform(img.convert("RGB")) for img in pil_images],
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        pixel_values = encoded["pixel_values"].to(device)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad(), autocast(device_type=device_type, dtype=dtype, enabled=device_type == "cuda"):
            probs = torch.softmax(model(pixel_values, input_ids, attention_mask).float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        del pixel_values, input_ids, attention_mask

    avg = np.mean(all_probs, axis=0)
    return [(float(row[1]) >= threshold, round(float(row[1]), 4)) for row in avg]


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

    files = sorted(records_dir.glob("*.json"))
    return (RecordItem(read_json(path), path) for path in files), source_root


def selected_items(items: Iterable[RecordItem], args: argparse.Namespace) -> Iterable[RecordItem]:
    wanted = set(args.uid or [])
    seen = 0
    yielded = 0
    for item in items:
        uid = str(item.record.get("uid", ""))
        if wanted and uid not in wanted:
            continue
        if seen < args.offset:
            seen += 1
            continue
        if args.limit is not None and yielded >= args.limit:
            break
        seen += 1
        yielded += 1
        yield item


def resolve_local_path(local_path: str | None, root: Path) -> Path | None:
    if not local_path:
        return None
    path = Path(local_path)
    return path if path.is_absolute() else root / path


def find_meme_image(record: dict[str, Any], source_root: Path) -> tuple[Path | None, str | None]:
    for image in ((record.get("meme_reply") or {}).get("images") or []):
        path = resolve_local_path(image.get("local_path"), source_root)
        if path and path.exists():
            return path, image.get("local_path")
    return None, None


def role_list(copy_images: str) -> tuple[str, ...]:
    if copy_images == "none":
        return ()
    if copy_images == "meme":
        return ("meme_reply",)
    return POST_IMAGE_FIELDS


def relative_image_destination(source: Path, role: str, uid: str) -> Path:
    parts = list(source.parts)
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[idx:])
    return Path("images") / role / uid / source.name


def copy_record_images(record: dict[str, Any], source_root: Path, out_root: Path, copy_images: str) -> tuple[int, int]:
    copied = 0
    missing = 0
    uid = str(record.get("uid", "unknown"))
    for role in role_list(copy_images):
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
    parser = argparse.ArgumentParser(description="Filter hydrated records with MemeTector v4.")
    parser.add_argument("--input", default="hydrated_records", help="Hydrated output directory, records directory, JSON, or JSONL.")
    parser.add_argument("--model-dir", default="../02_meme_classification/checkpoints")
    parser.add_argument("--out", default="filtered_records")
    parser.add_argument("--clip-name", default=DEFAULT_CLIP_NAME)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, mps, or a torch device string.")
    parser.add_argument("--single-fold", type=int, choices=[1, 2, 3, 4, 5], default=None)
    parser.add_argument("--ocr-text", default=DEFAULT_OCR_TEXT, help="Text input paired with each image, matching collection defaults.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--uid", action="append", default=[], help="Filter only this UID. Can repeat.")
    parser.add_argument(
        "--copy-images",
        choices=("none", "meme", "all"),
        default="all",
        help="Copy local images into the filtered output. Default: all local images present.",
    )
    parser.add_argument("--write-all", action="store_true", help="Write rejected records too, with is_meme=false.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        return 1
    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        print("        Pass --model-dir PATH to the MemeTector v4 checkpoints.")
        return 1

    try:
        deps = load_ml_dependencies()
        torch = deps["torch"]
        device = select_device(torch, args.device)
        print(json.dumps({
            "input": str(input_path),
            "model_dir": str(model_dir),
            "out": str(out_dir),
            "device": device,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
        }, indent=2))
        models, model_paths = load_models(deps, model_dir, args.clip_name, device, args.single_fold)
        processor = deps["CLIPProcessor"].from_pretrained(args.clip_name)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    raw_items, default_source_root = discover_records(input_path)
    items = selected_items(raw_items, args)
    Image = deps["Image"]

    started_at = utc_now()
    stats = {
        "total_selected": 0,
        "processed": 0,
        "kept": 0,
        "rejected": 0,
        "missing_meme_image": 0,
        "failed": 0,
        "images_copied": 0,
        "images_missing_on_copy": 0,
        "threshold": args.threshold,
        "clip_name": args.clip_name,
        "model_paths": model_paths,
        "started_at": started_at,
        "finished_at": None,
    }
    failures: list[dict[str, Any]] = []
    jsonl_path = out_dir / "filtered_records.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    batch: list[tuple[dict[str, Any], Path, str | None]] = []

    def flush_batch() -> None:
        if not batch:
            return
        pil_images = []
        usable = []
        for record, image_path, original_local_path in batch:
            try:
                pil_images.append(Image.open(image_path).convert("RGB"))
                usable.append((record, image_path, original_local_path))
            except Exception as exc:
                stats["failed"] += 1
                failures.append({"uid": record.get("uid"), "error": f"image_open_failed: {exc}"})

        if not usable:
            batch.clear()
            return

        results = classify_batch(
            deps=deps,
            pil_images=pil_images,
            processor=processor,
            models=models,
            device=device,
            threshold=args.threshold,
            ocr_text=args.ocr_text,
        )
        with jsonl_path.open("a", encoding="utf-8") as jsonl:
            for (record, image_path, original_local_path), (is_meme, meme_prob) in zip(usable, results):
                stats["processed"] += 1
                if is_meme:
                    stats["kept"] += 1
                else:
                    stats["rejected"] += 1

                if not is_meme and not args.write_all:
                    continue

                out_record = copy.deepcopy(record)
                out_record["meme_prob"] = meme_prob
                out_record["threshold"] = args.threshold
                out_record["is_meme"] = bool(is_meme)
                out_record["filter_metadata"] = {
                    "filtered_at": utc_now(),
                    "model": "MemeTector v4",
                    "model_dir": str(model_dir),
                    "clip_name": args.clip_name,
                    "single_fold": args.single_fold,
                    "num_models": len(models),
                    "ocr_text": args.ocr_text,
                    "source_input": str(input_path),
                    "source_image_path": str(image_path),
                    "source_local_path": original_local_path,
                }
                copied, missing = copy_record_images(out_record, default_source_root, out_dir, args.copy_images)
                stats["images_copied"] += copied
                stats["images_missing_on_copy"] += missing
                write_record(out_record, out_dir)
                jsonl.write(json.dumps(out_record, ensure_ascii=False, sort_keys=True) + "\n")
        batch.clear()

    for item in items:
        stats["total_selected"] += 1
        uid = item.record.get("uid", "unknown")
        image_path, local_path = find_meme_image(item.record, default_source_root)
        if not image_path:
            stats["missing_meme_image"] += 1
            failures.append({"uid": uid, "error": "missing_local_meme_image"})
            continue
        batch.append((item.record, image_path, local_path))
        if len(batch) >= args.batch_size:
            print(f"[PROGRESS] selected={stats['total_selected']} processed={stats['processed']} kept={stats['kept']}")
            flush_batch()

    flush_batch()
    stats["finished_at"] = utc_now()
    report = {"stats": stats, "failures": failures}
    (out_dir / "filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))
    return 0 if stats["processed"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
