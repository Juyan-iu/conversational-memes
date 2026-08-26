#!/usr/bin/env python3
"""
Old-vs-corrected benchmark comparison tables (for the revision).

Takes labeled prediction files from any number of runs and produces:
  - per-model accuracy table (markdown + LaTeX)
  - wrong-pick distribution table per run
  - optional paired old-vs-new delta column when labels share a model name
    with prefixes "old:" / "new:"

Usage:
  python comparison_table.py \
      --run "old:gpt-4o=results/<old_run>/gpt-4o_predictions.jsonl" \
      --run "new:gpt-4o=results/<new_run>/gpt-4o_predictions.jsonl" \
      --run "new:gemini-2.5-pro=results/<run>/gemini-2.5-pro_predictions.jsonl" \
      --out comparison_tables.md

Each predictions.jsonl row needs: pred (A-D or other), correct (bool),
wrong_type (lexical/visual/random) or the fields your pipeline writes; the
loader tolerates missing wrong_type by recomputing nothing (reports only acc).
"""
import argparse, json, re
from collections import Counter
from pathlib import Path


def load_run(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    valid = [r for r in rows if str(r.get("pred", "")) in "ABCD"]
    acc = sum(1 for r in valid if r.get("correct")) / len(valid) if valid else 0.0
    wrong = [r for r in valid if not r.get("correct")]
    dist = Counter(r.get("wrong_type") or r.get("picked_type") or "?" for r in wrong)
    total_wrong = sum(dist.values()) or 1
    dist_pct = {k: v / total_wrong for k, v in dist.items()}
    return {"n": len(valid), "acc": acc, "wrong_n": len(wrong), "dist": dist_pct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help='label=path, label like "old:gpt-4o" or "new:gpt-4o"')
    ap.add_argument("--out", default="comparison_tables.md")
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label.strip()] = load_run(path.strip())

    models = sorted({re.sub(r"^(old|new):", "", k) for k in runs})
    lines = ["# Benchmark comparison: original vs corrected construction\n",
             "## Accuracy\n",
             "| Model | Original | Corrected | Δ (pts) |",
             "|---|---|---|---|"]
    latex = ["\\begin{tabular}{lccc}", "\\toprule",
             "Model & Original & Corrected & $\\Delta$ \\\\", "\\midrule"]
    for m in models:
        old = runs.get(f"old:{m}")
        new = runs.get(f"new:{m}")
        o = f"{old['acc']*100:.1f}" if old else "--"
        n = f"{new['acc']*100:.1f}" if new else "--"
        d = f"{(new['acc']-old['acc'])*100:+.1f}" if old and new else "--"
        lines.append(f"| {m} | {o} | {n} | {d} |")
        latex.append(f"{m} & {o} & {n} & {d} \\\\")
    latex += ["\\bottomrule", "\\end{tabular}"]

    lines += ["\n## Wrong-pick distribution (share of errors)\n",
              "| Run | n | acc | lexical | visual | random |",
              "|---|---|---|---|---|---|"]
    for label, r in sorted(runs.items()):
        d = r["dist"]
        lines.append(f"| {label} | {r['n']} | {r['acc']*100:.1f}% | "
                     f"{d.get('lexical',0)*100:.1f}% | {d.get('visual',0)*100:.1f}% | "
                     f"{d.get('random',0)*100:.1f}% |")

    lines += ["\n## LaTeX (accuracy table)\n", "```latex"] + latex + ["```"]
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")
    for label, r in sorted(runs.items()):
        print(f"  {label}: acc={r['acc']*100:.1f}% (n={r['n']})")


if __name__ == "__main__":
    main()
