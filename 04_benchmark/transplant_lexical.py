#!/usr/bin/env python3
"""
Real-caption transplant for the lexical distractor (design alternative to
LLM-generated captions).

Rationale: generated captions leave a semantic fingerprint that survives all
surface-level fixes (detection stuck at 32-33% across 7 factor ablations).
Transplanting captions actually written by users on OTHER memes removes the
"generated" signal entirely; both gold and lexical then carry re-rendered
human-written text.

Matching constraints per target item:
  - source is a different item (self excluded)
  - same caption count as the target's gold (1-panel vs 2-panel compatibility)
  - total caption length within +/-50% of the target gold's total length
  - among compatible sources, pick the highest conversation-caption content
    word overlap (lexical relatedness, per the distractor spec);
    ties broken randomly, each source reused at most 3 times

Two subcommands (run from 04_benchmark, bench_env):

  1) Build the caption pool by OCR-ing all gold images (GPU, one-off):
     python transplant_lexical.py pool \
         --converted ../05_benchmark_evaluation/data/vlmeval_converted \
         --out captions_pool.json

  2) Apply transplants (no API, renders locally):
     python transplant_lexical.py apply \
         --pool captions_pool.json \
         --orig ../05_benchmark_evaluation/data/vlmeval_converted \
         --tree ../05_benchmark_evaluation/data/vlmeval_corrected \
         [--ids ../05_benchmark_evaluation/probe_sample_500_ids.txt]

Then re-judge gate 1 (rename probe checkpoint first) and gate 2.
"""
import argparse, json, random, re, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image
from benchmark_pipeline import (extract_captions, analyze_caption_style,
                                inpaint_captions, draw_caption_at_bbox)

STOP = set("the a an and or but of to in on at for with is are was were be been i you he she it we they this that my your not no yes so just like".split())


def words(text):
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and len(w) > 2]


def cmd_pool(args):
    conv = Path(args.converted)
    out = {}
    ckpt = Path(args.out)
    if ckpt.exists():
        out = json.loads(ckpt.read_text())
        print(f"[RESUME] {len(out)} already in pool")
    dirs = sorted(p for p in conv.iterdir() if p.is_dir())
    for i, d in enumerate(dirs, 1):
        if d.name in out:
            continue
        gold = next((p for p in d.iterdir()
                     if p.stem == d.name and p.suffix.lower() in
                     (".jpg", ".jpeg", ".png", ".webp")), None)
        if gold is None:
            continue
        try:
            caps = extract_captions(Image.open(gold).convert("RGB"))
        except Exception as e:
            print(f"  [err] {d.name}: {e}")
            continue
        out[d.name] = {"texts": [c["text"] for c in caps], "n": len(caps)}
        if i % 100 == 0:
            ckpt.write_text(json.dumps(out))
            print(f"  {i}/{len(dirs)}", flush=True)
    ckpt.write_text(json.dumps(out))
    n_nonempty = sum(1 for v in out.values() if v["n"] > 0)
    print(f"[DONE] pool: {len(out)} items, {n_nonempty} with captions -> {args.out}")


def cmd_apply(args):
    pool = json.loads(Path(args.pool).read_text())
    orig, tree = Path(args.orig), Path(args.tree)
    rng = random.Random(42)
    use_count = Counter()

    dirs = sorted(p for p in tree.iterdir() if p.is_dir())
    if args.ids:
        keep = set(Path(args.ids).read_text().split())
        dirs = [d for d in dirs if d.name in keep]
    print(f"[TRANSPLANT] {len(dirs)} items", flush=True)

    stats = {"ok": 0, "no_pool_self": 0, "no_match": 0, "no_lex_source": 0,
             "no_captions": 0, "bbox_mismatch": 0}

    for i, d in enumerate(dirs, 1):
        marker = d / "REBUILD.json"
        rec = json.loads(marker.read_text()) if marker.exists() else {"uid": d.name}
        if rec.get("lexical", {}).get("transplanted"):
            continue

        self_entry = pool.get(d.name)
        if not self_entry or self_entry["n"] == 0:
            stats["no_pool_self"] += 1
            continue
        target_n = self_entry["n"]
        target_len = sum(len(t) for t in self_entry["texts"])

        conv_text = re.sub(r"\[Tone:.*?\]", "",
                           (d / "info.txt").read_text(errors="ignore"))
        conv_words = set(words(conv_text))
        gold_words = set(words(" ".join(self_entry["texts"])))

        # candidate sources
        cands = []
        for uid, e in pool.items():
            if uid == d.name or e["n"] != target_n or e["n"] == 0:
                continue
            if use_count[uid] >= 3:
                continue
            tot = sum(len(t) for t in e["texts"])
            if not (0.5 * target_len <= tot <= 1.5 * target_len):
                continue
            src_words = set(words(" ".join(e["texts"])))
            # GUARD: exclude near-duplicates of the target's own gold caption
            # (same template reposts would create a second correct answer)
            union = gold_words | src_words
            if union and len(gold_words & src_words) / len(union) > 0.5:
                continue
            ov = len(conv_words & src_words)
            cands.append((ov, rng.random(), uid))
        if not cands:
            stats["no_match"] += 1
            continue
        cands.sort(reverse=True)
        src_uid = cands[0][2]
        src_texts = pool[src_uid]["texts"]
        use_count[src_uid] += 1

        # render onto the ORIGINAL converted lexical image (single inpaint)
        labels = json.loads((d / "labels.json").read_text())
        lex_name = next((f for f, s in labels.items() if s == "lexical"), None)
        orig_dir = orig / d.name
        src_img_path = next((p for p in orig_dir.iterdir()
                             if p.stem == Path(lex_name).stem
                             and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")),
                            None) if (lex_name and orig_dir.exists()) else None
        if src_img_path is None:
            stats["no_lex_source"] += 1
            continue
        img = Image.open(src_img_path).convert("RGB")
        captions = extract_captions(img)
        if not captions:
            stats["no_captions"] += 1
            continue
        if len(captions) != len(src_texts):
            # counts diverged between gold-OCR and lexical-OCR; take largest bboxes
            captions = sorted(captions, key=lambda c: (c["bbox"][2]-c["bbox"][0])
                              * (c["bbox"][3]-c["bbox"][1]), reverse=True)[:len(src_texts)]
            if len(captions) < len(src_texts):
                stats["bbox_mismatch"] += 1
                continue
            captions = sorted(captions, key=lambda c: c["bbox"][1])  # top-to-bottom
        styles = [analyze_caption_style(img, c) for c in captions]
        out_img = inpaint_captions(img, extract_captions(img))
        for cap, style, text in zip(captions, styles, src_texts):
            out_img = draw_caption_at_bbox(out_img, text, style=style)
        target = (d / lex_name).with_suffix(".jpg")
        out_img.convert("RGB").save(target, "JPEG", quality=90, optimize=True)

        rec.setdefault("lexical", {})
        rec["lexical"].update({"transplanted": True, "source_uid": src_uid,
                               "captions": src_texts, "file": target.name})
        marker.write_text(json.dumps(rec, ensure_ascii=False))
        stats["ok"] += 1
        if i % 25 == 0:
            print(f"  {i}/{len(dirs)}  {stats}", flush=True)

    print(f"\n[DONE] {stats}")
    print("Next: rename the probe checkpoint, re-run artifact_probe (gate 1),")
    print("then conv/noctx (gate 2) on the same sample.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pool")
    p1.add_argument("--converted", required=True)
    p1.add_argument("--out", default="captions_pool.json")
    p2 = sub.add_parser("apply")
    p2.add_argument("--pool", required=True)
    p2.add_argument("--orig", required=True)
    p2.add_argument("--tree", required=True)
    p2.add_argument("--ids", default=None)
    args = ap.parse_args()
    {"pool": cmd_pool, "apply": cmd_apply}[args.cmd](args)
