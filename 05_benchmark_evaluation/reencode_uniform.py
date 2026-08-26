#!/usr/bin/env python3
"""
Uniform re-encoding pass: decode and re-save ALL FOUR option images of every
item with identical parameters (RGB, JPEG quality 90, EXIF stripped), so that
compression history and metadata cannot distinguish options.

Builds a parallel tree; the source tree is not modified.

Usage:
  cd 05_benchmark_evaluation
  python reencode_uniform.py \
      --src data/vlmeval_neutralized \
      --out data/vlmeval_neutralized_v2
Then re-run the probe:
  python artifact_probe.py --provider openai --model gpt-4o \
      --sample 200 --seed 0 --data-root data/vlmeval_neutralized_v2
  (rename/remove artifact_probe_openai_gpt-4o.json first, or the old
   checkpoint will skip every item)
"""
import argparse, shutil
from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_items = n_imgs = 0
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        dst = out / d.name
        dst.mkdir(exist_ok=True)
        for f in d.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                img = Image.open(f).convert("RGB")
                # save everything as .jpg with identical params; keep the stem
                # (the loader matches gold by stem == folder name, and
                # labels.json keys include extensions, so rewrite those below)
                target = dst / (f.stem + ".jpg")
                img.save(target, "JPEG", quality=args.quality, optimize=True)
                n_imgs += 1
            else:
                shutil.copy2(f, dst / f.name)
        # fix labels.json keys if extensions changed
        lab = dst / "labels.json"
        if lab.exists():
            import json
            labels = json.loads(lab.read_text())
            labels = {Path(k).stem + ".jpg": v for k, v in labels.items()}
            lab.write_text(json.dumps(labels))
        n_items += 1
        if n_items % 50 == 0:
            print(f"  {n_items} items...")

    print(f"[DONE] {n_items} items, {n_imgs} images re-encoded -> {out}")


if __name__ == "__main__":
    main()
