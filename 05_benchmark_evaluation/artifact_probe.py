#!/usr/bin/env python3
"""
Artifact-detection probe for the MEMECONV rebuttal (Reviewer exx5 W3 / Praw W1).

Question: can a model identify the inpainted/re-rendered option (the lexical/text
distractor) from rendering artifacts ALONE — no conversation shown?

If detection accuracy is near the 25% chance level, rendering artifacts do not
reveal the manipulated option, so text-distractor errors in the main benchmark
reflect caption-context fit rather than surface cues.

Usage:
  cd 05_benchmark_evaluation
  export OPENAI_API_KEY=...        # and/or GOOGLE_API_KEY / GEMINI_API_KEY
  python artifact_probe.py --provider openai --model gpt-5.4-mini --sample 300
  python artifact_probe.py --provider google --model gemini-2.5-pro --sample 300

Notes:
  - Uses the same dataset loader as the main evaluation (runners.base.load_dataset),
    so items/letters match the main runs.
  - --sample uses fixed seed 42 (same convention as the ablation sampling).
  - Results saved to artifact_probe_<provider>_<model>.json (resumable).
"""

import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from runners.base import load_dataset, extract_letter, prepare_image_data_url

PROBE_PROMPT = (
    "You are shown four meme images labeled A, B, C, and D.\n\n"
    "Exactly ONE of them has been digitally manipulated: its original caption "
    "was removed by inpainting and replaced with newly rendered text.\n\n"
    "Look for signs of editing: inpainting smudges or blur, inconsistent "
    "background texture behind the text, font rendering that does not match "
    "the image style, unnatural text placement, or compression differences.\n\n"
    "Which image (A, B, C, or D) was manipulated?\n"
    "Respond with ONLY one letter. Do NOT explain."
)


def ask_openai(model, images):
    from openai import OpenAI
    client = OpenAI()
    content = [{"type": "text", "text": PROBE_PROMPT}]
    for letter in "ABCD":
        content.append({"type": "text", "text": f"Image {letter}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": prepare_image_data_url(images[letter])}})
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}],
        max_tokens=10, temperature=0)
    return resp.choices[0].message.content.strip()


def ask_google(model, images):
    from google import genai
    from google.genai import types
    client = genai.Client()
    contents = [PROBE_PROMPT]
    for letter in "ABCD":
        contents.append(f"Image {letter}:")
        contents.append(types.Part.from_bytes(
            data=Path(images[letter]).read_bytes(), mime_type="image/jpeg"))
    resp = client.models.generate_content(
        model=model, contents=contents,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=256, thinking_config=types.ThinkingConfig(thinking_budget=128)))
    return (resp.text or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None, help="converted eval data dir (same as main pipeline)")
    ap.add_argument("--provider", choices=["openai", "google"], default="openai")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="A/B/C/D shuffle seed — MUST match the main runs")
    args = ap.parse_args()

    data_root = args.data_root
    if data_root is None:
        # same default the main pipeline uses after conversion
        cand = Path(__file__).parent / "eval_data_converted"
        if not cand.exists():
            sys.exit("Pass --data-root <converted eval data dir> (the dir the main pipeline evaluates).")
        data_root = cand

    items = load_dataset(data_root, seed=args.seed)
    # keep only items where the lexical distractor letter is known
    items = [it for it in items if "lexical" in it.slot_type_by_letter.values()]
    print(f"[LOAD] {len(items)} items with a labeled lexical distractor")

    rng = random.Random(42)
    items = sorted(items, key=lambda it: it.id)
    if args.sample and args.sample < len(items):
        items = rng.sample(items, args.sample)
    print(f"[SAMPLE] probing {len(items)} items")

    out_path = Path(__file__).parent / f"artifact_probe_{args.provider}_{args.model.replace('/','_')}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}

    ask = ask_openai if args.provider == "openai" else ask_google
    for i, it in enumerate(items, 1):
        if it.id in done:
            continue
        gold = next(l for l, t in it.slot_type_by_letter.items() if t == "lexical")
        for attempt in range(4):
            try:
                raw = ask(args.model, it.images)
                pred = extract_letter(raw) or "?"
                done[it.id] = {"pred": pred, "gold_lexical": gold}
                break
            except Exception as e:
                msg = str(e).lower()
                if any(s in msg for s in ("rate", "429", "timeout", "500", "502", "503")):
                    time.sleep(2 ** attempt + random.random()); continue
                done[it.id] = {"pred": f"ERR:{e}"[:80], "gold_lexical": gold}
                break
        if i % 10 == 0:
            out_path.write_text(json.dumps(done, indent=2))
            print(f"  {i}/{len(items)}")
    out_path.write_text(json.dumps(done, indent=2))

    # ---- score ----
    valid = [(v["pred"], v["gold_lexical"]) for v in done.values()
             if v["pred"] in ("A", "B", "C", "D")]
    n = len(valid)
    k = sum(p == g for p, g in valid)
    acc = k / n if n else 0.0
    print(f"\n[RESULT] artifact detection: {k}/{n} = {acc:.3f}  (chance = 0.250)")
    try:
        from scipy import stats
        p = stats.binomtest(k, n, 0.25, alternative="greater").pvalue
        print(f"         binomial test vs chance: p = {p:.4f}")
        print("         p > .05  -> detection at chance: artifacts do NOT reveal the manipulated option")
        print("         p <= .05 -> model detects artifacts above chance: report honestly / qualify")
    except ImportError:
        print("         (pip install scipy for the binomial test)")


if __name__ == "__main__":
    main()
