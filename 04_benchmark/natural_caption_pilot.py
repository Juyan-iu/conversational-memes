#!/usr/bin/env python3
"""
Natural-caption pilot (final fix candidate for the residual artifact).

Diagnosis so far: after equalizing rendering (all four options inpainted and
re-rendered) and uniform re-encoding, a residual detection signal remains
(~30%, p~.04), and a text-only probe shows the same signal from caption text
alone (32.3%, p=.012). The lexical distractor's caption is YAKE keywords
pasted together and does not read like human writing.

Fix tested here: regenerate the lexical distractor's caption with an LLM so it
(a) still references the conversation topic (preserving the distractor's
"lexically related but pragmatically wrong" design) but (b) reads like a
caption a person would write. Then re-render it at the same location and
re-encode uniformly.

Run from 04_benchmark (needs easyocr, lama, PIL, openai):
  pip install openai   # if not present in this env
  export OPENAI_API_KEY=...
  python natural_caption_pilot.py \
      --src ../05_benchmark_evaluation/data/vlmeval_neutralized_v2 \
      --out ../05_benchmark_evaluation/data/vlmeval_neutralized_v3

Then probe from 05_benchmark_evaluation:
  rm -f artifact_probe_openai_gpt-4o.json  (or rename)
  python artifact_probe.py --provider openai --model gpt-4o \
      --sample 200 --seed 0 --data-root data/vlmeval_neutralized_v3
"""
import argparse, json, re, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from PIL import Image
from benchmark_pipeline import (extract_captions, analyze_caption_style,
                                inpaint_captions, draw_caption_at_bbox)

GEN_PROMPT = (
    "You write short internet meme captions.\n\n"
    "Conversation:\n{conv}\n\n"
    "Write ONE short meme caption (under 8 words) that uses one or two key "
    "words from this conversation's topic, but would be a WRONG reply to this "
    "specific conversation: it should take a clearly different stance, "
    "emotion, or situation than the conversation calls for. It must still "
    "read like a caption a real person would write. Do not use quotes. Do "
    "not explain. Output the caption only."
)


def natural_caption(client, model, conv):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": GEN_PROMPT.format(conv=conv[:1500])}],
        max_completion_tokens=40)
    cap = (resp.choices[0].message.content or "").strip().strip('"').strip()
    return re.sub(r"\s+", " ", cap)[:70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen-model", default="gpt-5.4-mini")
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dirs = sorted(p for p in src.iterdir() if p.is_dir())
    print(f"[PILOT] {len(dirs)} items -> {out}")
    stats = {"ok": 0, "no_captions_on_distractor": 0, "gen_failed": 0}

    for i, d in enumerate(dirs, 1):
        dst = out / d.name
        if dst.exists() and (dst / "NATCAP.json").exists():
            continue  # resumable
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(d, dst)

        # locate the lexical distractor image via labels.json
        labels = json.loads((dst / "labels.json").read_text())
        lex_name = next((f for f, slot in labels.items() if slot == "lexical"), None)
        lex_path = dst / lex_name if lex_name else None
        if lex_path is None or not lex_path.exists():
            print(f"  [skip] {d.name}: lexical distractor not found"); continue

        conv = (dst / "info.txt").read_text(errors="ignore")
        conv = re.sub(r"\[Tone:.*?\]", "", conv)  # strip tone note

        # 1) generate a natural caption from the conversation
        cap_text = None
        for attempt in range(3):
            try:
                cap_text = natural_caption(client, args.gen_model, conv)
                if cap_text:
                    break
            except Exception as e:
                time.sleep(2 ** attempt)
        if not cap_text:
            stats["gen_failed"] += 1
            print(f"  [skip] {d.name}: caption generation failed"); continue

        # 2) re-render: OCR the current keyword caption, inpaint, draw new text
        img = Image.open(lex_path).convert("RGB")
        captions = extract_captions(img)
        if not captions:
            stats["no_captions_on_distractor"] += 1
            print(f"  [skip] {d.name}: no captions detected on lexical distractor")
            continue
        styles = [analyze_caption_style(img, cap) for cap in captions]
        cleaned = inpaint_captions(img, captions)
        # draw the new caption once at the largest bbox; clear the others
        largest = max(range(len(captions)),
                      key=lambda j: (captions[j]["bbox"][2]-captions[j]["bbox"][0])
                                    * (captions[j]["bbox"][3]-captions[j]["bbox"][1]))
        result = cleaned
        result = draw_caption_at_bbox(result, cap_text, style=styles[largest])
        result.convert("RGB").save(lex_path.with_suffix(".jpg"), "JPEG",
                                   quality=args.quality, optimize=True)
        if lex_path.suffix.lower() != ".jpg":
            lex_path.unlink()
            labels = {(f if f != lex_name else Path(lex_name).stem + ".jpg"): s
                      for f, s in labels.items()}
            (dst / "labels.json").write_text(json.dumps(labels))

        (dst / "NATCAP.json").write_text(json.dumps(
            {"uid": d.name, "new_caption": cap_text}))
        stats["ok"] += 1
        if i % 10 == 0:
            print(f"  {i}/{len(dirs)}  {stats}")

    print(f"\n[DONE] {stats}")
    print("Spot-check a few rendered captions before running the probe.")


if __name__ == "__main__":
    main()
