#!/usr/bin/env python3
"""
Caption-conversation lexical overlap analysis for the MEMECONV rebuttal
(Reviewer Praw W2: is the Relevance gain context understanding or vocabulary copying?)

Pipeline (run steps independently; each step caches its output):

  1. transcribe   : extract caption text from generated meme images via an LVLM
  2. overlap      : compute caption-vs-conversation n-gram overlap per item/condition
  3. analyze      : merge with human ratings, fit models:
                      (a) Relevance ~ condition + item FE + rater FE   (replicates paper)
                      (b) Relevance ~ condition + overlap + item FE + rater FE
                      (c) Relevance delta restricted to low/zero-overlap captions

Inputs
------
  --results   results_*.json from 06_meme_generation (has uid, images{a,b}, prompts)
  --jsonl     labeled_memes.jsonl (to rebuild conversation text per uid)
  --ratings   CSV with columns: uid, rater, condition, relevance
              condition must be 'a' (no context) or 'b' (with context).
              Extra columns (quality, etc.) are kept and can be analyzed with --dv.

Usage
-----
  export OPENAI_API_KEY=...
  python overlap_analysis.py transcribe --results results_XXX.json --images-root generated/
  python overlap_analysis.py overlap    --results results_XXX.json --jsonl ../03_filter_and_label/labeled_final/labeled_memes.jsonl
  python overlap_analysis.py analyze    --ratings ratings.csv [--dv relevance]

Requires: pip install pandas statsmodels openai
"""

import argparse, base64, json, re, sys, time
from pathlib import Path

HERE = Path(__file__).parent
CAPTIONS_JSON = HERE / "captions.json"
OVERLAP_CSV   = HERE / "overlap.csv"

STOPWORDS = set("""a an the and or but if then this that these those i you he she it we they
me him her us them my your his its our their of in on at to for with from by as is are was
were be been being do does did have has had not no so just very really can could will would
should may might must about into over under out up down off than too own same s t don won""".split())

# ---------------------------------------------------------------- transcribe

TRANSCRIBE_PROMPT = (
    "Transcribe ALL text visible in this meme image (captions, speech bubbles, "
    "labels). Output ONLY the transcribed text, preserving line breaks. "
    "If there is no text, output exactly: [NO TEXT]"
)

def cmd_transcribe(args):
    from openai import OpenAI
    client = OpenAI()
    results = json.loads(Path(args.results).read_text())
    captions = json.loads(CAPTIONS_JSON.read_text()) if CAPTIONS_JSON.exists() else {}

    for r in results:
        uid = r["uid"]
        imgs = r.get("images", {})
        for cond in ("a", "b"):
            key = f"{uid}|{cond}"
            if key in captions:
                continue
            # find the image entry whose key corresponds to this condition:
            # accepts "a", "A", "A_gpt-image-2", "prompt_a", etc.
            img_entry = None
            for k, v in imgs.items():
                kl = k.lower()
                if kl == cond or kl.startswith(cond + "_") or kl == f"prompt_{cond}":
                    img_entry = v
                    break
            if img_entry is None:
                print(f"[skip] {key}: no image key matching condition '{cond}' in {list(imgs)}"); continue
            # resolve path: as-is (relative to cwd), then under images_root
            candidates = [Path(img_entry)]
            if args.images_root:
                candidates += [Path(args.images_root) / img_entry,
                               Path(args.images_root) / uid / Path(img_entry).name]
            img_path = next((p for p in candidates if p.exists()), None)
            if img_path is None:
                print(f"[skip] {key}: file not found (tried {[str(c) for c in candidates]})"); continue
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": TRANSCRIBE_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ]}],
                        max_completion_tokens=300)
                    captions[key] = resp.choices[0].message.content.strip()
                    print(f"[ok] {key}: {captions[key][:60]!r}")
                    break
                except Exception as e:
                    print(f"[retry {attempt+1}] {key}: {e}"); time.sleep(2 ** attempt)
            CAPTIONS_JSON.write_text(json.dumps(captions, ensure_ascii=False, indent=2))
    print(f"\nSaved {len(captions)} captions -> {CAPTIONS_JSON}")
    print("SPOT-CHECK at least 15 transcriptions against the images before trusting the numbers.")

# ---------------------------------------------------------------- overlap

def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS and len(w) > 1]

def _ngrams(tokens, n):
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def overlap_scores(caption, conversation):
    ct, vt = _tokens(caption), _tokens(conversation)
    if not ct:
        return {"uni": 0.0, "bi": 0.0, "n_caption_tokens": 0}
    uni = len(set(ct) & set(vt)) / len(set(ct))
    cb, vb = _ngrams(ct, 2), _ngrams(vt, 2)
    bi = len(cb & vb) / len(cb) if cb else 0.0
    return {"uni": round(uni, 4), "bi": round(bi, 4), "n_caption_tokens": len(ct)}

def cmd_overlap(args):
    sys.path.insert(0, str(Path(args.gen_dir).resolve()))
    from generate_memes_compare import build_context_text  # same context definition as generation
    import pandas as pd

    records = {}
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r.get("uid") or r.get("id")] = r

    captions = json.loads(CAPTIONS_JSON.read_text())
    results  = json.loads(Path(args.results).read_text())
    rows = []
    for r in results:
        uid = r["uid"]
        rec = records.get(uid)
        if rec is None:
            print(f"[warn] uid {uid} not in jsonl"); continue
        conv = build_context_text(rec)
        for cond in ("a", "b"):
            cap = captions.get(f"{uid}|{cond}", "")
            if cap in ("", "[NO TEXT]"):
                sc = {"uni": 0.0, "bi": 0.0, "n_caption_tokens": 0}
            else:
                sc = overlap_scores(cap, conv)
            rows.append({"uid": uid, "condition": cond, "caption": cap, **sc})
    df = pd.DataFrame(rows)
    df.to_csv(OVERLAP_CSV, index=False)
    print(df.groupby("condition")[["uni", "bi"]].describe().round(3))
    print(f"\nSaved -> {OVERLAP_CSV}")
    # headline check: does context condition copy more vocabulary?
    a = df[df.condition == "a"]["uni"]; b = df[df.condition == "b"]["uni"]
    from scipy import stats as st
    t = st.ttest_rel(b.values, a.values) if len(a) == len(b) else st.ttest_ind(b, a)
    print(f"\nUnigram overlap  a(no-ctx)={a.mean():.3f}  b(ctx)={b.mean():.3f}  p={t.pvalue:.4f}")

# ---------------------------------------------------------------- analyze

def cmd_analyze(args):
    import pandas as pd
    import statsmodels.formula.api as smf

    ov = pd.read_csv(OVERLAP_CSV)
    rt = pd.read_csv(args.ratings)
    for col in ("uid", "rater", "condition", args.dv):
        if col not in rt.columns:
            sys.exit(f"ratings csv missing column: {col}")
    df = rt.merge(ov[["uid", "condition", "uni", "bi"]], on=["uid", "condition"], how="left")
    if df.uni.isna().any():
        print(f"[warn] {df.uni.isna().sum()} rating rows had no overlap match (dropped)")
        df = df.dropna(subset=["uni"])
    df["ctx"] = (df.condition == "b").astype(int)

    print(f"\n== (a) Paper replication: {args.dv} ~ ctx + item FE + rater FE ==")
    m1 = smf.ols(f"{args.dv} ~ ctx + C(uid) + C(rater)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["uid"]})
    print(m1.summary().tables[1].as_text().split("\n")[0:3][-1])
    print(f"ctx effect: b={m1.params['ctx']:.3f}  p={m1.pvalues['ctx']:.4f}")

    print(f"\n== (b) Controlling overlap: {args.dv} ~ ctx + unigram_overlap + item FE + rater FE ==")
    m2 = smf.ols(f"{args.dv} ~ ctx + uni + C(uid) + C(rater)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["uid"]})
    print(f"ctx effect: b={m2.params['ctx']:.3f}  p={m2.pvalues['ctx']:.4f}")
    print(f"overlap   : b={m2.params['uni']:.3f}  p={m2.pvalues['uni']:.4f}")

    print(f"\n== (c) Low-overlap subset (uni <= median of ctx condition) ==")
    thresh = df[df.ctx == 1].uni.median()
    sub = df[df.uni <= thresh]
    m3 = smf.ols(f"{args.dv} ~ ctx + C(uid) + C(rater)", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["uid"]})
    print(f"n={len(sub)} rows, threshold uni<={thresh:.3f}")
    print(f"ctx effect: b={m3.params['ctx']:.3f}  p={m3.pvalues['ctx']:.4f}")

    print("\nInterpretation guide:")
    print(" - (b) ctx stays sig & similar size -> gain NOT explained by copying (good for the paper)")
    print(" - (b) ctx shrinks toward 0 while overlap is sig -> gain is mediated by copying; report honestly,")
    print("   soften Abstract/Discussion as already committed in the response.")

# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe")
    t.add_argument("--results", required=True)
    t.add_argument("--images-root", default=None, help="dir containing generated images (if paths in json are stale)")
    t.add_argument("--model", default="gpt-5.4-mini")

    o = sub.add_parser("overlap")
    o.add_argument("--results", required=True)
    o.add_argument("--jsonl", required=True)
    o.add_argument("--gen-dir", default="06_meme_generation", help="dir containing generate_memes_compare.py")

    a = sub.add_parser("analyze")
    a.add_argument("--ratings", required=True)
    a.add_argument("--dv", default="relevance")

    args = ap.parse_args()
    {"transcribe": cmd_transcribe, "overlap": cmd_overlap, "analyze": cmd_analyze}[args.cmd](args)
