"""Drive one runner over the dataset and dump predictions to JSONL.

Single-pass entry point. For the full pilot (scoreboard + summary), use
`pilot.py` instead.

Example:
  python run_eval.py --model reallms --data-root data/placeholder \
      --out results/reallms_predictions.jsonl
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tqdm import tqdm

from runners.base import load_dataset

RUNNERS = {
    "reallms":           "runners.reallms_runner",
    "qwen25-vl-7b":      "runners.hf_runner_qwen",
    "qwen25-omni-7b":    "runners.hf_runner_omni",
    "internvl3-8b":      "runners.hf_runner_internvl",
    "qvq-72b":           "runners.hf_runner_qvq",
    "gpt-4o":            "runners.openai_runner",
    "claude-sonnet-4-5": "runners.anthropic_runner",
    "gemini-2.5-pro":    "runners.google_runner",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RUNNERS))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N items.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runner = importlib.import_module(RUNNERS[args.model])
    items = load_dataset(args.data_root, seed=args.seed)
    if args.limit:
        items = items[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for it in tqdm(items, desc=args.model):
            try:
                pred = runner.run(it)
                err = None
            except Exception as e:
                pred, err = "ERR", f"{type(e).__name__}: {e}"
            picked_slot_type = it.slot_type_by_letter.get(pred, "?") if pred in "ABCD" else "?"
            f.write(json.dumps({
                "id": it.id,
                "gold": it.answer,
                "pred": pred,
                "correct": pred == it.answer,
                "picked_slot_type": picked_slot_type,
                "slot_type_by_letter": it.slot_type_by_letter,
                "gold_filename": it.gold_filename,
                "model": args.model,
                "error": err,
            }) + "\n")
    print(f"Wrote {len(items)} predictions to {out}")


if __name__ == "__main__":
    main()
