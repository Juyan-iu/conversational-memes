#!/usr/bin/env python3
"""
Compare conv (original) vs noctx (ablation) runs on the same items.

Usage:
  python ablation_compare.py \
      --conv  results/<original_gemini_run>/gemini-2.5-pro_predictions.jsonl \
      --noctx results/<noctx_run>/gemini-2.5-pro_predictions.jsonl \
      [--ids ablation_sample_1000_seed42.txt]

If --ids is given, both runs are filtered to those items (use this when the
conv run covers all 5,000 items but the ablation ran on a sample).
Outputs paired accuracies, per-distractor error shift, and McNemar's test.
"""
import argparse, json, math
from pathlib import Path


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def acc(rows):
    ok = sum(1 for r in rows if r["pred"] in "ABCD" and r["correct"])
    n = sum(1 for r in rows if r["pred"] in "ABCD")
    return ok, n


def wrong_dist(rows):
    from collections import Counter
    c = Counter(r["picked_slot_type"] for r in rows
                if r["pred"] in "ABCD" and not r["correct"])
    tot = sum(c.values())
    return {k: (v, v / tot if tot else 0) for k, v in sorted(c.items())}, tot


def mcnemar(b, c):
    """Exact McNemar via binomial on discordant pairs (two-sided)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", required=True)
    ap.add_argument("--noctx", required=True)
    ap.add_argument("--ids", default=None)
    args = ap.parse_args()

    conv, noctx = load(args.conv), load(args.noctx)
    ids = set(noctx)
    if args.ids:
        ids &= set(Path(args.ids).read_text().split())
    ids &= set(conv)
    print(f"paired items: {len(ids)}")
    if len(ids) < len(noctx):
        print(f"  (noctx has {len(noctx)}, conv has {len(conv)}; intersection used)")

    cv = [conv[i] for i in ids]
    nc = [noctx[i] for i in ids]

    ok_c, n_c = acc(cv)
    ok_n, n_n = acc(nc)
    print(f"\nWITH context   : {ok_c}/{n_c} = {ok_c/n_c:.3f}")
    print(f"WITHOUT context: {ok_n}/{n_n} = {ok_n/n_n:.3f}")
    print(f"drop: {(ok_c/n_c - ok_n/n_n)*100:.1f} points")

    # McNemar on items valid in both
    both = [i for i in ids if conv[i]["pred"] in "ABCD" and noctx[i]["pred"] in "ABCD"]
    b = sum(1 for i in both if conv[i]["correct"] and not noctx[i]["correct"])
    c = sum(1 for i in both if not conv[i]["correct"] and noctx[i]["correct"])
    print(f"\nMcNemar discordant pairs: conv-only-correct={b}, noctx-only-correct={c}")
    print(f"McNemar exact p = {mcnemar(b, c):.2e}")

    for label, rows in (("WITH context", cv), ("WITHOUT context", nc)):
        d, tot = wrong_dist(rows)
        print(f"\nWrong-pick distribution — {label} ({tot} wrong):")
        for k, (v, frac) in d.items():
            print(f"  {k:<10} {v:>5}  ({frac:.3f})")

    print("\nRebuttal sentence template:")
    print(f'  "On the same {len(ids)} items, accuracy drops from {ok_c/n_c*100:.1f}% with '
          f'context to {ok_n/n_n*100:.1f}% without it ({(ok_c/n_c-ok_n/n_n)*100:.1f} points; '
          f'McNemar p={mcnemar(b,c):.0e})."')


if __name__ == "__main__":
    main()
