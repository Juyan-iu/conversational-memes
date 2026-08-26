"""One-command pilot runner for the meme-selection task (paper RQ1).

Task format: 4 meme images labeled A/B/C/D (shuffled at load) + conversation
context. The model picks one letter. Accuracy is reported overall and broken
down by the distractor type (lexical / visual / random) that the model picked
when it was wrong, if labels.json is available per item.

Usage:
  # Default: REALLMS, all items under data/placeholder
  python pilot.py

  # Smoke test on 1 item
  python pilot.py --num 1

  # Pick a specific runner
  python pilot.py --model gpt-4o
  python pilot.py --model qwen25-omni-7b

  # Run several runners back-to-back (each writes its own results folder)
  python pilot.py --model gpt-4o gemini-2.5-pro

  # Run every registered runner (commercial APIs need their *_API_KEY set;
  # HF runners need the ai-meme-gpu env on a GPU box)
  python pilot.py --model all

  # Combine: 5-item smoke test across all runners
  python pilot.py --model all --num 5

  # Real dataset
  python pilot.py --data-root /path/to/real/data

Results land in results/<run_id>/:
  - <model>_predictions.jsonl     per-item predictions
  - summary.json                  metadata + accuracies + timing
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    # override=True so .env wins over an empty/stale value already exported in the shell.
    load_dotenv(override=True)
except ImportError:
    pass

from tqdm import tqdm

from runners.base import load_dataset

RUNNERS = {
    # Free / school-API:
    "reallms":           "runners.reallms_runner",        # Llama 4 Scout via IU REALLMS (free, API)
    # Local HF (need ai-meme-gpu env + GPU):
    "qwen25-vl-7b":      "runners.hf_runner_qwen",        # Qwen2.5-VL-7B
    "qwen25-omni-7b":    "runners.hf_runner_omni",        # Qwen2.5-Omni-7B (set QWEN_OMNI_USE_SC=1 for SC variant)
    "internvl3-8b":      "runners.hf_runner_internvl",    # InternVL3-8B
    "qvq-72b":           "runners.hf_runner_qvq",         # QvQ-72B-Preview (reasoning VLM; needs 2×80GB or QVQ_USE_4BIT=1)
    # Commercial APIs (need OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY):
    "gpt-4o":            "runners.openai_runner",         # OpenAI GPT-4o
    "claude-sonnet-4-5": "runners.anthropic_runner",      # Claude Sonnet 4.5 (extended thinking)
    "gemini-2.5-pro":    "runners.google_runner",         # Gemini 2.5 Pro
}

HERE = Path(__file__).parent
DEFAULT_DATA_ROOT = (HERE / "data" / "placeholder").resolve()


def run_all(runner, items, out_path: Path, label: str) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out_path.open("w") as f:
        for it in tqdm(items, desc=label, leave=False):
            try:
                pred = runner.run(it)
                err = None
            except Exception as e:
                pred, err = "ERR", f"{type(e).__name__}: {e}"
            # Which distractor slot did the wrong answer come from?
            picked_slot_type = it.slot_type_by_letter.get(pred, "?") if pred in "ABCD" else "?"
            row = {
                "id": it.id,
                "gold": it.answer,
                "pred": pred,
                "correct": pred == it.answer,
                "picked_slot_type": picked_slot_type,
                "slot_type_by_letter": it.slot_type_by_letter,
                "gold_filename": it.gold_filename,
                "error": err,
            }
            f.write(json.dumps(row) + "\n")
            rows.append(row)
    return rows


def overall_accuracy(rows: list[dict]) -> tuple[float, int, int]:
    valid = [r for r in rows if r["error"] is None]
    if not valid:
        return 0.0, 0, len(rows)
    correct = sum(r["correct"] for r in valid)
    return correct / len(valid), correct, len(valid)


def per_distractor_confusion(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """How often did the model pick each distractor type when it was wrong?
    Returns {type: (wrong_count, total_wrong)}.
    """
    wrong = [r for r in rows if r["error"] is None and not r["correct"]]
    total = len(wrong)
    if not total:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for r in wrong:
        counts[r["picked_slot_type"]] += 1
    return {t: (c, total) for t, c in counts.items()}


def _resolve_models(requested: list[str]) -> list[str]:
    """Validate --model values and expand 'all'. Preserves order, dedupes."""
    valid = list(RUNNERS) + ["all"]
    out: list[str] = []
    for m in requested:
        if m == "all":
            out = list(RUNNERS)  # 'all' replaces everything; order = RUNNERS dict order
            return out
        if m not in RUNNERS:
            sys.exit(
                f"[error] unknown --model value: {m!r}\n"
                f"        valid choices: {', '.join(valid)}"
            )
        if m not in out:
            out.append(m)
    return out


def run_one_model(
    model_key: str,
    items: list,
    data_root: Path,
    seed: int,
    timestamp: str,
    run_id_override: str | None,
    multi_model: bool,
) -> dict:
    """Run a single model end-to-end. Prints results, writes summary, returns it."""
    runner = importlib.import_module(RUNNERS[model_key])

    if run_id_override:
        # When the user provides --run-id and we're running multiple models,
        # disambiguate by appending the model key.
        run_id = f"{run_id_override}_{model_key}" if multi_model else run_id_override
    else:
        run_id = f"{timestamp}_{model_key}"

    out_dir = HERE / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    shown_root = data_root.relative_to(HERE) if data_root.is_relative_to(HERE) else data_root
    print(f"\nPilot run: model={model_key}  n={len(items)}  "
          f"data_root={shown_root}  run_id={run_id}")
    print("=" * 72)

    t0 = time.time()
    rows = run_all(
        runner, items,
        out_path=out_dir / f"{model_key}_predictions.jsonl",
        label=model_key,
    )
    elapsed = time.time() - t0

    acc, ok, n = overall_accuracy(rows)
    print(f"\nResults ({elapsed:.1f}s wall-clock):")
    print(f"  Overall accuracy: {acc:.3f}  ({ok}/{n})")

    conf = per_distractor_confusion(rows)
    if conf:
        print("\n  When wrong, which distractor type did the model pick?")
        for t in sorted(conf):
            c, tot = conf[t]
            print(f"    {t:<10} {c}/{tot}  ({c/tot:.3f})")

    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"\n  Errors: {len(errors)}")
        for r in errors[:3]:
            print(f"    {r['id']}: {r['error']}")

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_key,
        "n_items": len(items),
        "data_root": str(data_root),
        "seed": seed,
        "wall_seconds": round(elapsed, 2),
        "accuracy": acc,
        "correct": ok,
        "total_valid": n,
        "wrong_by_distractor_type": {t: c for t, (c, _) in conf.items()},
        "error_count": len(errors),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved: {out_dir}/\n         summary.json, {model_key}_predictions.jsonl")
    return summary


def _run_reporter(summaries: list[dict], run_id_override: str | None, timestamp: str) -> None:
    """Invoke report.py on just the runs we produced this invocation."""
    run_dirs = [HERE / "results" / s["run_id"] for s in summaries
                if "run_id" in s and (HERE / "results" / s["run_id"]).exists()]
    if not run_dirs:
        print("\n[report] no run folders to aggregate (all models failed).", file=sys.stderr)
        return

    out_name = f"{run_id_override}_report" if run_id_override else f"{timestamp}_report"
    out_dir = HERE / "reports" / out_name

    try:
        import report  # local module — same dir
    except ImportError as e:
        print(f"\n[report] could not import report.py: {e}", file=sys.stderr)
        return

    print()  # blank line before the report header
    saved_argv = sys.argv
    sys.argv = ["report.py", "--out", str(out_dir), *[str(d) for d in run_dirs]]
    try:
        report.main()
    finally:
        sys.argv = saved_argv


def _print_cross_model_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"Cross-model comparison ({len(summaries)} runs):")
    print(f"  {'model':<22} {'acc':>7} {'correct':>10} {'errors':>8} {'wall(s)':>9}")
    print(f"  {'-'*22} {'-'*7} {'-'*10} {'-'*8} {'-'*9}")
    for s in summaries:
        print(f"  {s['model']:<22} {s['accuracy']:>7.3f} "
              f"{s['correct']:>4}/{s['total_valid']:<5} {s['error_count']:>8} "
              f"{s['wall_seconds']:>9.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="One-command pilot runner for meme-selection MC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available models:\n"
            f"  {', '.join(RUNNERS)}\n"
            "  all  (shortcut — runs every registered model in sequence)\n\n"
            "Examples:\n"
            "  python pilot.py --num 1                                  # smoke test, default model\n"
            "  python pilot.py --model gpt-4o --num 5                   # 5 items on GPT-4o\n"
            "  python pilot.py --model gpt-4o gemini-2.5-pro            # two models, full dataset\n"
            "  python pilot.py --model all --num 5                      # 5 items, every model"
        ),
    )
    ap.add_argument(
        "--model", nargs="+", default=["reallms"], metavar="MODEL",
        help=(
            "One or more runners to use (space-separated). "
            "Use 'all' to run every registered runner. "
            f"Choices: {', '.join(list(RUNNERS) + ['all'])}. "
            "Default: reallms."
        ),
    )
    ap.add_argument("--num", type=int, default=None, metavar="N",
                    help="Run only the first N items. Default: all items in --data-root.")
    ap.add_argument("--data-root", default=None,
                    help=f"Folder tree root. Default: {DEFAULT_DATA_ROOT.relative_to(HERE.parent)}")
    ap.add_argument("--run-id", default=None,
                    help="Name the output folder. Default: <timestamp>_<model>. "
                         "When multiple models run, each gets its own folder "
                         "(suffixed with the model key).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Shuffle seed for A/B/C/D label assignment. Default: 0.")
    ap.add_argument("--report", action="store_true",
                    help="After all models finish, run report.py on this batch "
                         "to write paper-ready tables (csv/md/tex) and plots "
                         "(accuracy + error breakdown) under reports/<timestamp>/.")
    args = ap.parse_args()

    models = _resolve_models(args.model)

    data_root = Path(args.data_root) if args.data_root else DEFAULT_DATA_ROOT
    if not data_root.exists():
        sys.exit(
            f"[error] data-root not found: {data_root}\n"
            f"[hint] generate a placeholder item with: python data/make_sample.py"
        )

    items = load_dataset(data_root, seed=args.seed)
    if args.num is not None:
        items = items[: args.num]
    if not items:
        sys.exit(f"[error] No items found under {data_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries: list[dict] = []
    for model_key in models:
        try:
            summary = run_one_model(
                model_key=model_key,
                items=items,
                data_root=data_root,
                seed=args.seed,
                timestamp=timestamp,
                run_id_override=args.run_id,
                multi_model=len(models) > 1,
            )
            summaries.append(summary)
        except Exception as e:
            # Don't let one model's failure (e.g. missing API key) abort the whole run.
            print(f"\n[!] {model_key} aborted: {type(e).__name__}: {e}", file=sys.stderr)
            summaries.append({
                "model": model_key, "accuracy": 0.0, "correct": 0,
                "total_valid": 0, "error_count": len(items),
                "wall_seconds": 0.0, "fatal_error": f"{type(e).__name__}: {e}",
            })

    if len(summaries) > 1:
        _print_cross_model_table(summaries)

    if args.report:
        _run_reporter(summaries, run_id_override=args.run_id, timestamp=timestamp)

    # Exit non-zero only if every model had errors.
    return 0 if any(s.get("error_count", 0) == 0 for s in summaries) else 2


if __name__ == "__main__":
    sys.exit(main())
