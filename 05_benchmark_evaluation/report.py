"""Aggregate pilot.py runs into academic-ready tables and plots.

Reads each run folder under ``results/`` (the ones pilot.py writes —
``<timestamp>_<model>/`` containing ``summary.json`` + ``<model>_predictions.jsonl``)
and emits a single report folder with:

  - results.csv             # one row per run; raw aggregate
  - results.md              # markdown table (paste into README / Discord)
  - results.tex             # LaTeX booktabs table (paste into the paper)
  - accuracy.{png,pdf}      # bar plot, accuracy + Wilson 95% CI, random baseline
  - error_breakdown.{png,pdf} # stacked bar — what distractor type the model
                              # picked when it was wrong (lex / vis / random)

Usage:
  # Default: every run folder under results/, sorted by accuracy desc
  python report.py

  # Only the most recent multi-model invocation
  # (groups runs by the timestamp prefix pilot.py adds)
  python report.py --latest

  # Pick specific runs explicitly (shell-glob is your friend)
  python report.py results/20260427_132846_*

  # Custom output directory
  python report.py --out reports/rq1-pilot/

Requires:
  - matplotlib  (optional; script still emits tables without it)
  - numpy       (only for plots; ships with matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
REPORTS_DIR = HERE / "reports"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion. Better than normal-approx for small n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> dict | None:
    """Read one run folder. Returns {summary, rows} or None if invalid."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return None

    pred_files = list(run_dir.glob("*_predictions.jsonl"))
    if not pred_files:
        return None
    rows = []
    for line in pred_files[0].read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"summary": summary, "rows": rows, "run_dir": run_dir}


def discover_runs(args_runs: list[str], latest: bool) -> list[Path]:
    """Resolve --runs / --latest / default into a list of run folders."""
    if args_runs:
        return [Path(p) for p in args_runs]

    if not RESULTS_DIR.exists():
        sys.exit(f"[error] results dir not found: {RESULTS_DIR}")

    candidates = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    if not candidates:
        sys.exit(f"[error] no run folders under {RESULTS_DIR}")

    if not latest:
        return candidates

    # Group by timestamp prefix "YYYYMMDD_HHMMSS" — same prefix = same pilot.py call.
    def prefix(d: Path) -> str:
        parts = d.name.split("_", 2)
        return "_".join(parts[:2]) if len(parts) >= 2 else d.name

    latest_prefix = prefix(candidates[-1])
    return [d for d in candidates if prefix(d) == latest_prefix]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_one(run: dict) -> dict:
    """One row of aggregated stats per run."""
    s = run["summary"]
    rows = run["rows"]
    valid = [x for x in rows if not x.get("error")]
    n = len(valid)
    n_correct = sum(int(x.get("correct", False)) for x in valid)
    acc = n_correct / n if n else 0.0
    lo, hi = wilson_ci(n_correct, n)

    # Distractor-type confusion: of the wrong predictions, which slot type
    # did the model pick? Tags from labels.json: lexical / visual / random.
    wrong = [x for x in valid if not x.get("correct")]
    picks: dict[str, int] = defaultdict(int)
    for w in wrong:
        picks[w.get("picked_slot_type", "?")] += 1

    return {
        "run_id": s.get("run_id", run["run_dir"].name),
        "model": s.get("model", "?"),
        "n": n,
        "n_correct": n_correct,
        "n_wrong": len(wrong),
        "accuracy": acc,
        "ci_lo": lo,
        "ci_hi": hi,
        "wrong_lexical": picks.get("lexical", 0),
        "wrong_visual": picks.get("visual", 0),
        "wrong_random": picks.get("random", 0),
        "wrong_unlabeled": picks.get("unlabeled", 0) + picks.get("?", 0),
        "wall_seconds": float(s.get("wall_seconds", 0.0)),
        "error_count": int(s.get("error_count", 0)),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _wrong_share(a: dict, key: str) -> str:
    total = a["n_wrong"]
    if total == 0:
        return "—"
    return f"{a[key]}/{total} ({a[key]/total:.2f})"


def write_csv(agg: list[dict], path: Path) -> None:
    if not agg:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        for a in agg:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()})


def write_markdown(agg: list[dict], path: Path) -> None:
    lines = [
        "| Model | N | Accuracy (95% CI) | Wrong→Lex | Wrong→Vis | Wrong→Rand |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in agg:
        lines.append(
            f"| `{a['model']}` | {a['n']} | "
            f"{a['accuracy']:.3f} ({a['ci_lo']:.3f}–{a['ci_hi']:.3f}) | "
            f"{_wrong_share(a, 'wrong_lexical')} | "
            f"{_wrong_share(a, 'wrong_visual')} | "
            f"{_wrong_share(a, 'wrong_random')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_latex(agg: list[dict], path: Path) -> None:
    """Booktabs-style table. Requires \\usepackage{booktabs} in the paper."""
    rows = []
    for a in agg:
        model = a["model"].replace("_", r"\_")
        rows.append(
            f"  {model} & {a['n']} & {a['accuracy']:.3f} "
            f"& [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}] "
            f"& {a['wrong_lexical']} & {a['wrong_visual']} & {a['wrong_random']} \\\\"
        )
    body = "\n".join(rows)
    path.write_text(
        "% Auto-generated by report.py — paste into the paper.\n"
        "% Requires: \\usepackage{booktabs}\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrrlrrr}\n"
        "\\toprule\n"
        " & & & & \\multicolumn{3}{c}{Wrong picks by type} \\\\\n"
        "\\cmidrule(lr){5-7}\n"
        "Model & N & Acc & 95\\% CI & Lex & Vis & Rand \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Zero-shot meme-selection accuracy on the 4-AFC task "
        "(1 correct + 1 lexical + 1 visual + 1 random distractor). "
        "``Wrong picks by type'' counts which distractor type the model chose "
        "when it was wrong; CIs are Wilson 95\\%.}\n"
        "\\label{tab:rq1-results}\n"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Plots (optional — matplotlib)
# ---------------------------------------------------------------------------

def plot_accuracy(agg: list[dict], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(agg) + 2), 4))
    models = [a["model"] for a in agg]
    accs = [a["accuracy"] for a in agg]
    err_lo = [a["accuracy"] - a["ci_lo"] for a in agg]
    err_hi = [a["ci_hi"] - a["accuracy"] for a in agg]

    bars = ax.bar(models, accs, yerr=[err_lo, err_hi],
                  capsize=4, color="#4c72b0", edgecolor="black", linewidth=0.4)
    # Random-chance baseline for 4-AFC
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=0.9, label="Random (25%)")

    # Numeric labels on top
    for bar, a in zip(bars, agg):
        ax.text(bar.get_x() + bar.get_width() / 2, a["ci_hi"] + 0.015,
                f"{a['accuracy']:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Meme-selection accuracy by model (Wilson 95% CI)")
    plt.xticks(rotation=20, ha="right")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return True


def plot_error_breakdown(agg: list[dict], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    # Skip if every model has zero errors (no story to tell)
    if all(a["n_wrong"] == 0 for a in agg):
        return False

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(agg) + 2), 4))
    models = [a["model"] for a in agg]
    x = np.arange(len(models))

    lex = np.array([a["wrong_lexical"] for a in agg], dtype=float)
    vis = np.array([a["wrong_visual"] for a in agg], dtype=float)
    rnd = np.array([a["wrong_random"] for a in agg], dtype=float)

    # Normalize to proportion of wrong picks (more comparable across N)
    totals = lex + vis + rnd
    totals[totals == 0] = 1  # avoid /0; bars stay zero-height
    lex_p, vis_p, rnd_p = lex / totals, vis / totals, rnd / totals

    ax.bar(x, lex_p, label="Lexical", color="#d62728")
    ax.bar(x, vis_p, bottom=lex_p, label="Visual", color="#9467bd")
    ax.bar(x, rnd_p, bottom=lex_p + vis_p, label="Random", color="#7f7f7f")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Share of wrong predictions")
    ax.set_ylim(0, 1.05)
    ax.set_title("When wrong, which distractor type did the model pick?")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate pilot.py runs into paper-ready tables and plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python report.py                              # all runs under results/\n"
            "  python report.py --latest                     # most recent pilot.py call only\n"
            "  python report.py results/20260427_132846_*    # explicit runs (shell glob)\n"
            "  python report.py --out reports/rq1-final/     # custom output dir\n"
            "  python report.py --sort model                 # sort by name instead of acc\n"
        ),
    )
    ap.add_argument("runs", nargs="*", default=None, metavar="RUN_DIR",
                    help="Run folders to aggregate. Default: all under results/.")
    ap.add_argument("--latest", action="store_true",
                    help="Only the most recent pilot.py invocation "
                         "(groups runs by the YYYYMMDD_HHMMSS prefix).")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="Output directory. Default: reports/<timestamp>/")
    ap.add_argument("--sort", choices=["accuracy", "model", "run_id"], default="accuracy",
                    help="Row order in tables/plots. Default: accuracy desc.")
    args = ap.parse_args()

    run_dirs = discover_runs(args.runs, args.latest)
    runs = [r for r in (load_run(d) for d in run_dirs) if r is not None]
    if not runs:
        sys.exit("[error] no valid runs (need summary.json + *_predictions.jsonl)")

    agg = [aggregate_one(r) for r in runs]

    if args.sort == "accuracy":
        agg.sort(key=lambda a: a["accuracy"], reverse=True)
    elif args.sort == "model":
        agg.sort(key=lambda a: a["model"])
    else:
        agg.sort(key=lambda a: a["run_id"])

    out_dir = (Path(args.out) if args.out
               else REPORTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(agg, out_dir / "results.csv")
    write_markdown(agg, out_dir / "results.md")
    write_latex(agg, out_dir / "results.tex")
    plotted_acc = plot_accuracy(agg, out_dir / "accuracy.png")
    plotted_err = plot_error_breakdown(agg, out_dir / "error_breakdown.png")

    shown_out = out_dir.relative_to(HERE) if out_dir.is_relative_to(HERE) else out_dir
    print(f"\nReport ({len(agg)} runs) → {shown_out}/")
    print("  results.csv           (raw rows)")
    print("  results.md            (markdown table)")
    print("  results.tex           (LaTeX booktabs)")
    if plotted_acc:
        print("  accuracy.{png,pdf}    (bar + Wilson 95% CI)")
    if plotted_err:
        print("  error_breakdown.{png,pdf}  (stacked: lex/vis/rand)")
    if not plotted_acc:
        print("  [!] matplotlib not installed — plots skipped. "
              "  Run: pip install matplotlib")

    print("\n" + (out_dir / "results.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
