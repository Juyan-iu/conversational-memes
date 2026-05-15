#!/usr/bin/env python3
"""
VLM Evaluation Pipeline for Meme Selection Task (RQ1)

One-command runner: converts benchmark_data → vlm-eval format, then runs models.

Usage:
  # Convert benchmark_data and run all API models
  python vlm_eval_pipeline.py --data-root ../04_benchmark/benchmark_data --model gpt-4o

  # Smoke test (3 items)
  python vlm_eval_pipeline.py --data-root ../04_benchmark/benchmark_data --model gpt-4o --num 3

  # Multiple models + auto report
  python vlm_eval_pipeline.py --data-root ../04_benchmark/benchmark_data \
      --model gpt-4o claude-sonnet-4-5 gemini-2.5-pro --report

  # Local HF models (need GPU env)
  python vlm_eval_pipeline.py --data-root ../04_benchmark/benchmark_data \
      --model qwen25-omni-7b internvl3-8b qvq-72b

  # Skip conversion (already converted)
  python vlm_eval_pipeline.py --skip-convert --model gpt-4o --num 10

Available models:
  API:   gpt-4o, claude-sonnet-4-5, gemini-2.5-pro, reallms
  Local: qwen25-vl-7b, qwen25-omni-7b, internvl3-8b, qvq-72b
  all  → runs every registered model
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from tqdm import tqdm
from runners.base import load_dataset

# ════════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent

# Converted dataset lives here (auto-generated from benchmark_data)
CONVERTED_DATA_DIR = HERE / "data" / "vlmeval_converted"

RUNNERS = {
    "reallms":           "runners.reallms_runner",
    "qwen25-vl-7b":      "runners.hf_runner_qwen",
    "qwen25-omni-7b":    "runners.hf_runner_omni",
    "internvl3-8b":      "runners.hf_runner_internvl",
    "qvq-72b":           "runners.hf_runner_qvq",
    "gpt-4o":            "runners.openai_runner",
    "claude-sonnet-4-5": "runners.anthropic_runner",
    "gemini-2.5-pro":    "runners.google_runner",
    "llama-4-scout":     "runners.together_runner",
    "qwen3-vl-32b":      "runners.hf_runner_qwen3vl32b",
    "qwen3-vl-32b":      "runners.hf_runner_qwen3vl32b",
    "qwen3-vl-32b":      "runners.hf_runner_qwen3vl32b",
}


# ════════════════════════════════════════════════════════════════
#  Conversion: benchmark_data → vlm-eval format
# ════════════════════════════════════════════════════════════════

def uri_to_url(uri: str) -> str:
    if not uri:
        return ""
    try:
        parts = uri.replace("at://", "").split("/")
        if len(parts) >= 3:
            return f"https://bsky.app/profile/{parts[0]}/post/{parts[2]}"
    except Exception:
        pass
    return uri


def build_info_txt(meta: dict) -> tuple[str, list[str]]:
    """
    Full conversation thread with inline [IMAGE:N] placeholders.
    Images are placed exactly where they appear in the conversation.

    Returns:
        (info_txt, image_urls)
        info_txt: text with [IMAGE:0], [IMAGE:1], ... placeholders
        image_urls: list of CDN URLs in order of appearance
    """
    ctx = meta.get("context") or {}

    orig_text        = (ctx.get("original_post_text") or ctx.get("original_post") or "").strip()
    orig_images      = ctx.get("original_post_images") or []
    orig_ext_title   = (ctx.get("original_post_external_title") or "").strip()
    orig_ext_url     = (ctx.get("original_post_external_url") or "").strip()

    quoted_text      = (ctx.get("quoted_post_text") or "").strip()
    quoted_images    = ctx.get("quoted_post_images") or []
    quoted_ext_title = (ctx.get("quoted_post_external_title") or "").strip()
    quoted_ext_url   = (ctx.get("quoted_post_external_url") or "").strip()

    ancestor_chain   = ctx.get("ancestor_chain") or []

    parent_text      = (ctx.get("parent_reply_text") or ctx.get("parent_reply") or "").strip()
    parent_images    = ctx.get("parent_reply_images") or []
    parent_ext_title = (ctx.get("parent_reply_external_title") or "").strip()

    meme_text        = (ctx.get("meme_text") or "").strip()

    labels     = meta.get("labels") or {}
    stance     = (labels.get("meme_reply") or {}).get("stance") or {}
    stance_tags = [k.capitalize() for k, v in stance.items() if v]

    parts      = []
    image_urls = []  # ordered list of CDN URLs

    def add_images(urls, label="Image"):
        """Add labeled image placeholders and register URLs."""
        for i, url in enumerate(urls or []):
            if url:
                idx = len(image_urls)
                image_urls.append(url)
                suffix = f" {i+1}" if len(urls) > 1 else ""
                parts.append(f"[{label}{suffix}: IMAGE:{idx}]")

    # 1. Root post
    if orig_text or orig_ext_title or orig_images:
        parts.append("===== ROOT POST =====")
        if orig_text:
            parts.append(orig_text)
        add_images(orig_images, label="Original post image")
        if orig_ext_title:
            link = f" ({orig_ext_url})" if orig_ext_url else ""
            parts.append(f"[Link: {orig_ext_title}{link}]")

    # 2. Quoted/embedded post (repost content)
    if quoted_text or quoted_ext_title or quoted_images:
        parts.append("\n===== QUOTED/REPOSTED =====")
        if quoted_text:
            parts.append(quoted_text)
        add_images(quoted_images, label="Quoted/reposted image")
        if quoted_ext_title:
            link = f" ({quoted_ext_url})" if quoted_ext_url else ""
            parts.append(f"[Link: {quoted_ext_title}{link}]")

    # 3. Ancestor chain (oldest first)
    for i, node in enumerate(ancestor_chain):
        n_text      = (node.get("text") or "").strip()
        n_images    = node.get("images") or []
        n_ext_title = (node.get("external_title") or "").strip()
        n_ext_url   = (node.get("external_url") or "").strip()
        n_qt_text   = (node.get("quoted_post_text") or "").strip()
        n_qt_images = node.get("quoted_post_images") or []
        if n_text or n_images or n_ext_title or n_qt_text or n_qt_images:
            parts.append(f"\n===== THREAD REPLY [{i+1}] =====")
            if n_text:
                parts.append(n_text)
            add_images(n_images, label="Reply image")
            if n_ext_title:
                link = f" ({n_ext_url})" if n_ext_url else ""
                parts.append(f"[Link: {n_ext_title}{link}]")
            if n_qt_text or n_qt_images:
                parts.append("[Quoted:]")
                if n_qt_text:
                    parts.append(n_qt_text[:150])
                add_images(n_qt_images, label="Quoted image")

    # 4. Parent reply
    if parent_text or parent_images or parent_ext_title:
        parts.append("\n===== PARENT REPLY =====")
        if parent_text:
            parts.append(parent_text)
        add_images(parent_images, label="Parent reply image")
        if parent_ext_title:
            parts.append(f"[Link: {parent_ext_title}]")

    # 5. Meme placement + stance
    parts.append("\n===== MEME REPLY =====")
    if meme_text:
        parts.append(f'"{meme_text}"')
    else:
        parts.append("(no text caption)")
    if stance_tags:
        parts.append(f"[Tone: {', '.join(stance_tags)}]")
    parts.append("→ A meme image was posted here. Select which one.")

    if not any(p for p in parts if not p.startswith("=") and not p.startswith("\n=") and not p.startswith("→")):
        parts.insert(0, "(no conversation text — judge by visual context only)\n")

    return "\n".join(parts) + "\n", image_urls


def convert_benchmark_data(src: Path, dst: Path, limit: int | None = None) -> int:
    """Convert benchmark_data → vlm-eval format. Returns number of items converted."""
    summary_path = src / "benchmark_summary.jsonl"
    if not summary_path.exists():
        sys.exit(f"[error] benchmark_summary.jsonl not found in {src}")

    metas = []
    with open(summary_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    metas.append(json.loads(line))
                except Exception:
                    pass

    if limit:
        metas = metas[:limit]

    dst.mkdir(parents=True, exist_ok=True)
    success = skip = 0

    for meta in metas:
        uid      = meta.get("uid", "")
        item_src = src / uid
        item_dst = dst / uid

        orig_src = item_src / "A_original.jpg"
        lex_src  = item_src / "B_text_distractor.jpg"
        vis_src  = item_src / "C_visual_distractor.jpg"
        rnd_src  = item_src / "D_easy_distractor.jpg"

        if not all(p.exists() for p in [orig_src, lex_src, vis_src, rnd_src]):
            skip += 1
            continue

        item_dst.mkdir(parents=True, exist_ok=True)

        info_txt, ctx_image_urls_inline = build_info_txt(meta)
        (item_dst / "info.txt").write_text(info_txt, encoding="utf-8")
        shutil.copy2(orig_src, item_dst / f"{uid}.jpg")
        shutil.copy2(lex_src,  item_dst / "B_text_distractor.jpg")
        shutil.copy2(vis_src,  item_dst / "C_visual_distractor.jpg")
        shutil.copy2(rnd_src,  item_dst / "D_easy_distractor.jpg")

        (item_dst / "labels.json").write_text(json.dumps({
            "B_text_distractor.jpg":   "lexical",
            "C_visual_distractor.jpg": "visual",
            "D_easy_distractor.jpg":   "random",
        }, indent=2) + "\n", encoding="utf-8")

        # discourse.json — discourse labels for prompt version B
        disc_labels = meta.get("labels") or {}
        (item_dst / "discourse.json").write_text(
            json.dumps(disc_labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )

        # context_images.json — ordered CDN URLs matching [IMAGE:N] in info.txt
        (item_dst / "context_images.json").write_text(
            json.dumps(ctx_image_urls_inline, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )

        success += 1

    print(f"[CONVERT] {success} items → {dst}  (skipped: {skip})")
    return success


# ════════════════════════════════════════════════════════════════
#  Eval runner (from pilot.py)
# ════════════════════════════════════════════════════════════════

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


def run_all_circular(runner, items, out_dir: Path, label: str) -> list[dict]:
    """
    Circular evaluation: run each item 4 times, rotating the correct answer
    through positions A, B, C, D. Eliminates position bias.

    For each item, builds 4 permutations where the correct meme lands on
    each letter in turn. Final score = fraction of 4 runs answered correctly.

    Saves:
      - <model>_circular_run{0-3}.jsonl  — raw predictions per rotation
      - <model>_circular_summary.jsonl   — aggregated per-item score
    """
    import copy
    from pathlib import Path as _Path

    out_dir.mkdir(parents=True, exist_ok=True)
    all_runs: list[list[dict]] = []

    for rot in range(4):
        run_rows: list[dict] = []
        out_path = out_dir / f"{label}_circular_run{rot}.jsonl"
        with out_path.open("w") as f:
            for it in tqdm(items, desc=f"{label} rot{rot}", leave=False):
                # Rotate images so correct answer lands on letter ABCD[rot]
                letters = list("ABCD")
                # Find current correct letter
                correct_letter = it.answer
                correct_path   = it.images[correct_letter]
                # Build new order: correct at position rot, others fill remaining
                others = [it.images[l] for l in letters if l != correct_letter]
                new_order = others[:rot] + [correct_path] + others[rot:]
                new_images = dict(zip(letters, new_order))
                new_slot_type = {}
                for l, path in new_images.items():
                    if path == correct_path:
                        new_slot_type[l] = "correct"
                    else:
                        orig_l = next(k for k, v in it.images.items() if v == path)
                        new_slot_type[l] = it.slot_type_by_letter.get(orig_l, "unlabeled")

                new_item = copy.copy(it)
                new_item.images = new_images
                new_item.answer = letters[rot]
                new_item.slot_type_by_letter = new_slot_type

                try:
                    pred = runner.run(new_item)
                    err  = None
                except Exception as e:
                    pred, err = "ERR", f"{type(e).__name__}: {e}"

                picked_slot_type = new_slot_type.get(pred, "?") if pred in "ABCD" else "?"
                row = {
                    "id":                it.id,
                    "rotation":          rot,
                    "gold":              new_item.answer,
                    "pred":              pred,
                    "correct":           pred == new_item.answer,
                    "picked_slot_type":  picked_slot_type,
                    "slot_type_by_letter": new_slot_type,
                    "gold_filename":     it.gold_filename,
                    "error":             err,
                }
                f.write(json.dumps(row) + "\n")
                run_rows.append(row)
        all_runs.append(run_rows)
        print(f"  rot{rot} accuracy: {sum(r['correct'] for r in run_rows if not r['error'])}/{sum(1 for r in run_rows if not r['error'])}")

    # Aggregate: per-item score across 4 rotations
    summary_rows: list[dict] = []
    summary_path = out_dir / f"{label}_circular_summary.jsonl"
    item_ids = [it.id for it in items]
    with summary_path.open("w") as f:
        for i, uid in enumerate(item_ids):
            rot_results = [all_runs[rot][i] for rot in range(4)]
            valid = [r for r in rot_results if not r["error"]]
            n_correct = sum(r["correct"] for r in valid)
            # Distractor types picked when wrong
            wrong_types = [r["picked_slot_type"] for r in valid if not r["correct"]]
            row = {
                "id":           uid,
                "score":        n_correct / max(len(valid), 1),
                "correct_runs": n_correct,
                "valid_runs":   len(valid),
                "wrong_picked": wrong_types,
                "error_runs":   sum(1 for r in rot_results if r["error"]),
            }
            f.write(json.dumps(row) + "\n")
            summary_rows.append(row)

    return summary_rows


def circular_accuracy(summary_rows: list[dict]) -> tuple[float, int, int]:
    """Accuracy from circular summary: item counted correct if score >= 0.5 (>=2/4)."""
    valid = [r for r in summary_rows if r["valid_runs"] > 0]
    if not valid:
        return 0.0, 0, 0
    # Standard: correct if model gets it right in majority of rotations
    correct = sum(1 for r in valid if r["score"] >= 0.5)
    return correct / len(valid), correct, len(valid)


def overall_accuracy(rows: list[dict]) -> tuple[float, int, int]:
    valid = [r for r in rows if r["error"] is None]
    if not valid:
        return 0.0, 0, len(rows)
    correct = sum(r["correct"] for r in valid)
    return correct / len(valid), correct, len(valid)


def per_distractor_confusion(rows: list[dict]) -> dict[str, tuple[int, int]]:
    wrong = [r for r in rows if r["error"] is None and not r["correct"]]
    total = len(wrong)
    if not total:
        return {}
    counts: dict[str, int] = defaultdict(int)
    for r in wrong:
        counts[r["picked_slot_type"]] += 1
    return {t: (c, total) for t, c in counts.items()}


def run_one_model(model_key: str, items: list, timestamp: str,
                  run_id_override: str | None, multi_model: bool) -> dict:
    import os as _os
    runner = importlib.import_module(RUNNERS[model_key])

    prompt_ver = _os.environ.get("PROMPT_VERSION", "conv")
    if run_id_override:
        run_id = f"{run_id_override}_{model_key}" if multi_model else run_id_override
    else:
        run_id = f"{timestamp}_{model_key}_{prompt_ver}"

    out_dir = HERE / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning: model={model_key}  n={len(items)}  run_id={run_id}")
    print("=" * 72)

    use_circular = getattr(run_one_model, "_circular", False)

    t0 = time.time()
    if use_circular:
        print(f"  [CIRCULAR] 4-rotation evaluation (position bias elimination)")
        summary_rows = run_all_circular(runner, items, out_dir=out_dir, label=model_key)
        elapsed = time.time() - t0
        acc, ok, n = circular_accuracy(summary_rows)
        errors = [r for r in summary_rows if r["error_runs"] > 0]

        print(f"\nResults ({elapsed:.1f}s) [circular]:")
        print(f"  Overall accuracy: {acc:.3f}  ({ok}/{n})")

        # Distractor confusion from summary
        from collections import Counter as _Counter
        all_wrong = [t for r in summary_rows for t in r["wrong_picked"]]
        conf_counts = _Counter(all_wrong)
        total_wrong = len(all_wrong)
        if total_wrong:
            print("\n  When wrong, distractor type picked:")
            for t, c in sorted(conf_counts.items()):
                print(f"    {t:<12} {c}/{total_wrong}  ({c/total_wrong:.3f})")
        conf = {t: (c, total_wrong) for t, c in conf_counts.items()}
    else:
        rows = run_all(runner, items,
                       out_path=out_dir / f"{model_key}_predictions.jsonl",
                       label=model_key)
        elapsed = time.time() - t0
        acc, ok, n = overall_accuracy(rows)
        errors = [r for r in rows if r["error"]]

        print(f"\nResults ({elapsed:.1f}s):")
        print(f"  Overall accuracy: {acc:.3f}  ({ok}/{n})")

        conf = per_distractor_confusion(rows)
        if conf:
            print("\n  When wrong, distractor type picked:")
            for t in sorted(conf):
                c, tot = conf[t]
                print(f"    {t:<12} {c}/{tot}  ({c/tot:.3f})")

    if errors:
        print(f"\n  Errors: {len(errors)}")
        err_list = errors[:3]
        for r in err_list:
            print(f"    {r['id']}: {r.get('error', '')}")

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_key,
        "prompt_version": _os.environ.get("PROMPT_VERSION", "conv"),
        "circular": use_circular,
        "n_items": len(items),
        "wall_seconds": round(elapsed, 2),
        "accuracy": acc,
        "correct": ok,
        "total_valid": n,
        "wrong_by_distractor_type": {t: c for t, (c, _) in conf.items()},
        "error_count": len(errors),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved: {out_dir.relative_to(HERE)}/")
    return summary


def _print_cross_model_table(summaries: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"Cross-model comparison ({len(summaries)} runs):")
    print(f"  {'model':<22} {'acc':>7} {'correct':>10} {'errors':>8} {'wall(s)':>9}")
    print(f"  {'-'*22} {'-'*7} {'-'*10} {'-'*8} {'-'*9}")
    for s in summaries:
        print(f"  {s['model']:<22} {s['accuracy']:>7.3f} "
              f"{s['correct']:>4}/{s['total_valid']:<5} {s['error_count']:>8} "
              f"{s['wall_seconds']:>9.1f}")


def _run_reporter(summaries: list[dict], run_id_override: str | None, timestamp: str) -> None:
    run_dirs = [HERE / "results" / s["run_id"] for s in summaries
                if "run_id" in s and (HERE / "results" / s["run_id"]).exists()]
    if not run_dirs:
        return
    out_name = f"{run_id_override}_report" if run_id_override else f"{timestamp}_report"
    out_dir = HERE / "reports" / out_name
    try:
        import report
    except ImportError as e:
        print(f"\n[report] could not import report.py: {e}", file=sys.stderr)
        return
    saved_argv = sys.argv
    sys.argv = ["report.py", "--out", str(out_dir), *[str(d) for d in run_dirs]]
    try:
        report.main()
    finally:
        sys.argv = saved_argv


def _resolve_models(requested: list[str]) -> list[str]:
    out: list[str] = []
    for m in requested:
        if m == "all":
            return list(RUNNERS)
        if m not in RUNNERS:
            sys.exit(f"[error] unknown model: {m!r}\n"
                     f"        choices: {', '.join(list(RUNNERS) + ['all'])}")
        if m not in out:
            out.append(m)
    return out


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description="VLM Evaluation Pipeline — meme selection task (RQ1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", nargs="+", default=["gpt-4o"], metavar="MODEL",
                    help=f"Model(s) to run. Choices: {', '.join(list(RUNNERS) + ['all'])}. "
                         "Space-separated for multiple. Default: gpt-4o")
    ap.add_argument("--data-root", default=None,
                    help="benchmark_data directory (will be auto-converted). "
                         f"Default: {CONVERTED_DATA_DIR}")
    ap.add_argument("--uid-list", default=None,
                    help="텍스트 파일 경로 (한 줄에 uid 하나). 해당 uid만 평가.")
    ap.add_argument("--num", type=int, default=None,
                    help="Limit to first N items (smoke test). Default: all")
    ap.add_argument("--skip-convert", action="store_true",
                    help="Skip conversion step (use already-converted data)")
    ap.add_argument("--run-id", default=None,
                    help="Custom output folder name. Default: <timestamp>_<model>")
    ap.add_argument("--seed", type=int, default=0,
                    help="Shuffle seed for A/B/C/D assignment. Default: 0")
    ap.add_argument("--circular", action="store_true",
                    help="Circular evaluation: run each item 4x with answer at each position "
                         "(A/B/C/D), average score. Eliminates position bias. "
                         "Recommended for local models only (4x compute).")
    ap.add_argument("--report", action="store_true",
                    help="Auto-generate paper tables + plots after all models finish")
    ap.add_argument("--prompt-version", default="conv", choices=["conv", "disc"],
                    help="Prompt version: 'conv' (full conversation text, default) | "
                         "'disc' (discourse-label based, no raw text)")
    args = ap.parse_args()

    models = _resolve_models(args.model)

    # Step 1: Convert
    if args.skip_convert:
        eval_data_dir = CONVERTED_DATA_DIR
        print(f"[CONVERT] Skipping — using {eval_data_dir}")
    elif args.data_root:
        src = Path(args.data_root)
        print(f"[CONVERT] {src} → {CONVERTED_DATA_DIR}")
        n = convert_benchmark_data(src, CONVERTED_DATA_DIR, limit=args.num)
        if n == 0:
            sys.exit("[error] No items converted — check benchmark_data path")
        eval_data_dir = CONVERTED_DATA_DIR
    else:
        eval_data_dir = CONVERTED_DATA_DIR
        if not eval_data_dir.exists() or not any(eval_data_dir.iterdir()):
            sys.exit(f"[error] No converted data found at {eval_data_dir}\n"
                     f"        Pass --data-root <benchmark_data> to auto-convert.")

    if not eval_data_dir.exists():
        sys.exit(f"[error] Eval data dir not found: {eval_data_dir}")

    # Set prompt version via env (runners read PROMPT_VERSION)
    import os
    os.environ["PROMPT_VERSION"] = args.prompt_version
    print(f"[PROMPT] version={args.prompt_version}")

    # Step 2: Load dataset
    items = load_dataset(eval_data_dir, seed=args.seed)
    if args.uid_list:
        uid_set = set(open(args.uid_list).read().splitlines())
        items = [it for it in items if it.id in uid_set]
        print(f"[UID-LIST] {len(items)} items filtered")
    if args.num is not None:
        items = items[:args.num]
    if not items:
        sys.exit(f"[error] No items found under {eval_data_dir}")
    print(f"[LOAD] {len(items)} items  seed={args.seed}")

    # 아이템별 info.txt 미리보기 (--num 지정 시)
    if items and args.num:
        print(f"\n{'─'*60}")
        print(f"[INFO.TXT PREVIEW — {len(items)} items]")
        for it in items:
            print(f"\n▶ {it.id}")
            print(it.conversation_text.strip())
            print("─" * 60)
        print()

    # Step 3: Run models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries: list[dict] = []
    for model_key in models:
        try:
            run_one_model._circular = args.circular
            summary = run_one_model(
                model_key=model_key,
                items=items,
                timestamp=timestamp,
                run_id_override=args.run_id,
                multi_model=len(models) > 1,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"\n[!] {model_key} aborted: {type(e).__name__}: {e}", file=sys.stderr)
            summaries.append({
                "model": model_key, "run_id": f"{timestamp}_{model_key}",
                "accuracy": 0.0, "correct": 0, "total_valid": 0,
                "error_count": len(items), "wall_seconds": 0.0,
                "fatal_error": f"{type(e).__name__}: {e}",
            })

    if len(summaries) > 1:
        _print_cross_model_table(summaries)

    if args.report:
        _run_reporter(summaries, run_id_override=args.run_id, timestamp=timestamp)

    return 0 if any(s.get("error_count", 0) == 0 for s in summaries) else 2


if __name__ == "__main__":
    sys.exit(main())
