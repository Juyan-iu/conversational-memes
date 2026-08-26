#!/usr/bin/env python3
"""
Pilot benchmark neutralization for the MEMECONV rebuttal.

Goal (advisor's suggested experiment): pass the ORIGINAL meme (and the raw
"random"/easy distractor) through the same inpaint-and-render pipeline used to
build the text/visual distractors, re-rendering their OWN original captions.
After this, all four options carry the same rendering artifacts, so:
  - the artifact-detection probe should return to ~25% chance, and
  - a context ablation on the neutralized items should show a real drop.

This script builds a parallel copy of the converted eval data with neutralized
gold + easy images. Distractor images, info.txt, and labels.json are copied
unchanged. Nothing in the source tree is modified.

Usage (run inside the 04_benchmark environment, which has easyocr / lama):
  cd 04_benchmark
  python neutralize_pilot.py \
      --data-root ../05_benchmark_evaluation/data/vlmeval_converted \
      --out       ../05_benchmark_evaluation/data/vlmeval_neutralized \
      --ids       ../05_benchmark_evaluation/ablation_sample_1000_seed42.txt \
      --sample 200

Then evaluate on it from 05_benchmark_evaluation:
  python artifact_probe.py --provider google --model gemini-2.5-pro \
      --sample 200 --seed 0 --data-root data/vlmeval_neutralized
  python vlm_eval_pipeline.py --skip-convert --model gemini-2.5-pro \
      --prompt-version noctx --sample 200 --data-root data/vlmeval_neutralized
  (check whether --skip-convert respects --data-root in your pipeline; if not,
   point CONVERTED_DATA_DIR or copy the neutralized tree over the expected path)
"""
import argparse, json, random, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image
from benchmark_pipeline import (extract_captions, analyze_caption_style,
                                inpaint_captions, draw_caption_at_bbox)


def neutralize_image(img: Image.Image):
    """OCR -> style -> LaMa inpaint -> re-render the ORIGINAL caption text."""
    captions = extract_captions(img)
    if not captions:
        return None, 0  # nothing detected; caller decides what to do
    styles = [analyze_caption_style(img, cap) for cap in captions]
    cleaned = inpaint_captions(img, captions)
    result = cleaned
    for cap, style in zip(captions, styles):
        result = draw_caption_at_bbox(result, cap["text"], style=style)
    return result, len(captions)


def find_gold_and_random(item_dir: Path):
    gold = next((p for p in item_dir.iterdir()
                 if p.stem == item_dir.name and p.suffix.lower() in
                 (".jpg", ".jpeg", ".png", ".webp")), None)
    labels_path = item_dir / "labels.json"
    rand = None
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        for fname, slot in labels.items():
            if slot == "random":
                rand = item_dir / fname
                break
    return gold, rand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=None, help="restrict to these item ids (one per line)")
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()

    src = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dirs = sorted([d for d in src.iterdir() if d.is_dir()], key=lambda d: d.name)
    if args.ids:
        keep = set(Path(args.ids).read_text().split())
        dirs = [d for d in dirs if d.name in keep]
    rng = random.Random(42)
    if args.sample and args.sample < len(dirs):
        dirs = rng.sample(dirs, args.sample)
    print(f"[NEUTRALIZE] {len(dirs)} items -> {out}")

    stats = {"ok_both": 0, "gold_only": 0, "no_gold_captions": 0,
             "random_no_captions": 0, "skipped": 0}
    manifest = []
    for i, d in enumerate(dirs, 1):
        dst = out / d.name
        if dst.exists() and (dst / "NEUTRALIZED.json").exists():
            continue  # resumable
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(d, dst)

        gold, rand = find_gold_and_random(dst)
        if gold is None:
            stats["skipped"] += 1
            print(f"  [skip] {d.name}: gold image not found")
            shutil.rmtree(dst)
            continue

        entry = {"uid": d.name, "gold_neutralized": False, "random_neutralized": False}
        img = Image.open(gold).convert("RGB")
        neut, ncap = neutralize_image(img)
        if neut is None:
            # gold with no detectable captions cannot be neutralized; excluding
            # it keeps the pilot clean (paper says 100% of items have OCR
            # captions, so this should be rare)
            stats["no_gold_captions"] += 1
            print(f"  [drop] {d.name}: no captions detected on gold")
            shutil.rmtree(dst)
            continue
        neut.save(gold, quality=90) if gold.suffix.lower() in (".jpg", ".jpeg") else neut.save(gold)
        entry["gold_neutralized"] = True
        entry["gold_captions"] = ncap

        if rand is not None and rand.exists():
            rimg = Image.open(rand).convert("RGB")
            rneut, rncap = neutralize_image(rimg)
            if rneut is not None:
                rneut.save(rand, quality=90) if rand.suffix.lower() in (".jpg", ".jpeg") else rneut.save(rand)
                entry["random_neutralized"] = True
                entry["random_captions"] = rncap
                stats["ok_both"] += 1
            else:
                stats["random_no_captions"] += 1
                stats["gold_only"] += 1
        else:
            stats["gold_only"] += 1

        (dst / "NEUTRALIZED.json").write_text(json.dumps(entry))
        manifest.append(entry)
        if i % 10 == 0:
            print(f"  {i}/{len(dirs)}  {stats}")

    (out / "neutralize_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[DONE] {stats}")
    print(f"manifest -> {out/'neutralize_manifest.json'}")
    print("\nNote for the write-up: items where the random option had no detectable")
    print("captions ('random_no_captions') remain asymmetric on that one option;")
    print("report their count and, if needed, analyze the 'ok_both' subset separately.")


if __name__ == "__main__":
    main()
