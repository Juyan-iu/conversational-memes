#!/usr/bin/env python3
"""
Recalibrate lexical distractor captions (v5) in an existing corrected tree.

Diagnosis: v4 captions containing proper nouns / numbers / very specific
entities create a caption-image incongruity cue (the new caption visibly does
not belong on that meme template). Fix: condition generation on the meme's
visual description so the caption plausibly fits the image, and forbid names,
numbers, and highly specific entities.

Only the lexical distractor image is touched. Gold/easy neutralization from
the existing rebuild is preserved. The lexical SOURCE image is taken from the
ORIGINAL converted tree (one inpaint cycle, not stacked on the v4 render).

Run from 04_benchmark (bench_env):
  export OPENAI_API_KEY=...
  nohup python -u recalibrate_lexical.py \
      --orig ../05_benchmark_evaluation/data/vlmeval_converted \
      --tree ../05_benchmark_evaluation/data/vlmeval_corrected \
      --jsonl ../03_filter_and_label/labeled_final/labeled_memes.jsonl \
      > recal.log 2>&1 &

Then re-run the acceptance probe on the tree.
"""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image
from benchmark_pipeline import (extract_captions, analyze_caption_style,
                                inpaint_captions, draw_caption_at_bbox)

GEN_PROMPT = (
    "You write short internet meme captions.\n\n"
    "The meme image shows: {visdesc}\n\n"
    "Conversation:\n{conv}\n\n"
    "Write ONE short caption (under 8 words) for the image described above. "
    "Requirements: it must plausibly fit that image; it may use one or two "
    "ORDINARY topic words from the conversation, but must NOT contain names, "
    "numbers, places, or other highly specific entities; and it must be a "
    "WRONG reply to this specific conversation, taking a clearly different "
    "stance, emotion, or situation than the conversation calls for. It must "
    "read like a caption a real person would write. Do not use quotes. Do "
    "not explain. Output the caption only."
)


def gen_caption(client, model, visdesc, conv):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": GEN_PROMPT.format(
            visdesc=visdesc[:600], conv=conv[:1200])}],
        max_completion_tokens=40)
    cap = (resp.choices[0].message.content or "").strip().strip('"').strip()
    return re.sub(r"\s+", " ", cap)[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, help="original converted tree (lexical source images)")
    ap.add_argument("--tree", required=True, help="corrected tree to update in place")
    ap.add_argument("--jsonl", required=True, help="labeled_memes.jsonl (visual descriptions)")
    ap.add_argument("--gen-model", default="gpt-5.4-mini")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--only-uids", default=None, help="optional txt of uids to redo (default: all)")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI()

    # visual descriptions by uid
    visdesc = {}
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                uid = r.get("uid") or r.get("id")
                vd = (r.get("visual_description")
                      or (r.get("labels", {}).get("meme_reply", {}) or {}).get("visual_description")
                      or "")
                if uid and vd:
                    visdesc[uid] = vd
    print(f"[VISDESC] {len(visdesc)} descriptions loaded", flush=True)

    orig, tree = Path(args.orig), Path(args.tree)
    dirs = sorted(p for p in tree.iterdir() if p.is_dir())
    if args.only_uids:
        keep = set(Path(args.only_uids).read_text().split())
        dirs = [d for d in dirs if d.name in keep]
    print(f"[RECAL] {len(dirs)} items", flush=True)

    stats = {"ok": 0, "no_visdesc": 0, "no_lex_source": 0,
             "no_captions": 0, "gen_failed": 0}

    for i, d in enumerate(dirs, 1):
        marker = d / "REBUILD.json"
        rec = json.loads(marker.read_text()) if marker.exists() else {"uid": d.name}
        if rec.get("lexical", {}).get("recalibrated_v5"):
            continue  # resumable

        labels = json.loads((d / "labels.json").read_text())
        lex_name = next((f for f, s in labels.items() if s == "lexical"), None)
        if lex_name is None:
            stats["no_lex_source"] += 1; continue
        # source: original converted tree (match by stem; ext may differ)
        src_dir = orig / d.name
        src = next((p for p in src_dir.iterdir()
                    if p.stem == Path(lex_name).stem and p.suffix.lower() in
                    (".jpg", ".jpeg", ".png", ".webp")), None) if src_dir.exists() else None
        if src is None:
            stats["no_lex_source"] += 1
            print(f"  [skip] {d.name}: lexical source not found", flush=True); continue

        vd = visdesc.get(d.name, "")
        if not vd:
            stats["no_visdesc"] += 1
        conv = re.sub(r"\[Tone:.*?\]", "", (d / "info.txt").read_text(errors="ignore"))

        cap = None
        for attempt in range(3):
            try:
                cap = gen_caption(client, args.gen_model,
                                  vd or "a generic internet meme image", conv)
                if cap:
                    break
            except Exception:
                time.sleep(2 ** attempt)
        if not cap:
            stats["gen_failed"] += 1; continue

        img = Image.open(src).convert("RGB")
        captions = extract_captions(img)
        if not captions:
            stats["no_captions"] += 1; continue
        styles = [analyze_caption_style(img, c) for c in captions]
        largest = max(range(len(captions)),
                      key=lambda j: (captions[j]["bbox"][2]-captions[j]["bbox"][0])
                                    * (captions[j]["bbox"][3]-captions[j]["bbox"][1]))
        result = inpaint_captions(img, captions)
        result = draw_caption_at_bbox(result, cap, style=styles[largest])
        target = d / (Path(lex_name).stem + ".jpg")
        result.convert("RGB").save(target, "JPEG", quality=args.quality, optimize=True)

        rec.setdefault("lexical", {})
        rec["lexical"].update({"naturalized": True, "recalibrated_v5": True,
                               "new_caption": cap, "file": target.name})
        marker.write_text(json.dumps(rec, ensure_ascii=False))
        stats["ok"] += 1
        if i % 50 == 0:
            print(f"  {i}/{len(dirs)}  {stats}", flush=True)

    print(f"\n[DONE] {stats}", flush=True)
    print("Re-run the acceptance probe on the tree, then gate 2 (conv/noctx).")


if __name__ == "__main__":
    main()
