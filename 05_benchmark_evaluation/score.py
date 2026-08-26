"""Score per-item prediction JSONL files.

Expected file naming: <model>_ctx.jsonl and <model>_noctx.jsonl in the results dir.

Example:
  python score.py results/
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _accuracy(rows: list[dict]) -> float:
    valid = [r for r in rows if r.get("error") is None]
    if not valid:
        return 0.0
    return sum(r["pred"] == r["gold"] for r in valid) / len(valid)


def _per_distractor(rows: list[dict]) -> dict[str, float]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        t = r.get("distractor_type") or "unlabeled"
        by_type[t].append(r)
    return {t: _accuracy(rs) for t, rs in by_type.items()}


def report(results_dir: str | Path = "results") -> None:
    results_dir = Path(results_dir)
    by_model: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for p in sorted(results_dir.glob("*.jsonl")):
        stem = p.stem
        if stem.endswith("_ctx"):
            model, ctx = stem[:-4], "ctx"
        elif stem.endswith("_noctx"):
            model, ctx = stem[:-6], "noctx"
        else:
            continue
        by_model[model][ctx] = _load(p)

    if not by_model:
        print(f"No *_ctx.jsonl / *_noctx.jsonl files in {results_dir}")
        return

    # Main table.
    print(f"{'model':<22} {'w/o ctx':>8} {'w/ ctx':>8} {'lift':>8} {'n':>6}")
    print("-" * 58)
    for model, conds in sorted(by_model.items()):
        wo_rows = conds.get("noctx", [])
        w_rows = conds.get("ctx", [])
        wo = _accuracy(wo_rows) if wo_rows else 0.0
        w = _accuracy(w_rows) if w_rows else 0.0
        n = max(len(wo_rows), len(w_rows))
        lift = w - wo if (wo_rows and w_rows) else 0.0
        print(f"{model:<22} {wo:>8.3f} {w:>8.3f} {lift:>+8.3f} {n:>6}")

    # Per-distractor-type breakdown.
    print("\nPer-distractor accuracy (w/ context):")
    for model, conds in sorted(by_model.items()):
        rows = conds.get("ctx", [])
        if not rows:
            continue
        per = _per_distractor(rows)
        parts = [f"{t}={a:.3f}" for t, a in sorted(per.items())]
        print(f"  {model:<22} {'  '.join(parts)}")

    # Error counts.
    print("\nErrors:")
    for model, conds in sorted(by_model.items()):
        for label, rows in conds.items():
            errs = [r for r in rows if r.get("error")]
            if errs:
                print(f"  {model} ({label}): {len(errs)}/{len(rows)} failed — first: {errs[0]['error']}")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "results")
