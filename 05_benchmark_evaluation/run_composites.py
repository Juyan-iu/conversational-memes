"""Run commercial VLMs on composite-screenshot meme-selection items.

Use this when each test item is a SINGLE pre-rendered PNG containing both the
Bluesky conversation AND the four candidate memes labeled A/B/C/D — i.e. you
took a screenshot of the whole task as a human would see it. (This is
different from pilot.py, which expects per-item folders with separate
info.txt + 4 image files.)

Usage:
  # Run all .png in a folder through GPT-4o, Claude, Gemini:
  python run_composites.py /path/to/data

  # Pick a subset of models:
  python run_composites.py /path/to/data --model gpt-4o gemini-2.5-pro

  # Provide gold answers (enables accuracy + per-item correctness):
  #   - 10-letter string, one per file in sorted order:
  python run_composites.py /path/to/data --gold ABCDABCDAB
  #   - or a JSON file: {"1.png": "A", "2.png": "C", ...}
  python run_composites.py /path/to/data --gold gold.json

Outputs (under reports/<timestamp>_composite/):
  results.csv             one row per (item, model)
  results.md              markdown summary table
  picks_matrix.{png,pdf}  items × models, color-coded by letter
  letter_distribution.{png,pdf}  per-model A/B/C/D pick frequency
  accuracy.{png,pdf}      ONLY when --gold is provided
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    # override=True so .env wins over an empty/stale value already exported in the shell
    # (otherwise `ANTHROPIC_API_KEY=` in a shell rc file silently shadows .env).
    load_dotenv(override=True)
except ImportError:
    pass

from runners.base import extract_letter, image_to_data_url


# ---------------------------------------------------------------------------
# Image prep: downscale + re-encode to match the budget plan's 512 px JPEG
# spec (per Juyoung's review, May 2026). 2940×1912 PNG composites get
# resized to 512 px max-edge, JPEG-88 (~30-60 KB on disk). This:
#   - matches the per-provider token estimates in the budget plan,
#   - stays well under every provider's per-image limit
#     (Anthropic 5 MB, OpenAI 20 MB, Gemini 20 MB total request).
# ---------------------------------------------------------------------------
import base64 as _b64
import io as _io


def prepare_image(path: Path, max_edge: int = 512, jpeg_quality: int = 88) -> tuple[bytes, str, str]:
    """Return (raw_bytes, mime, base64_data_url) suitable for any provider.

    Always downscales to max_edge and re-encodes as JPEG. Composite screenshots
    don't need PNG fidelity for VLM letter-picking, and JPEG keeps payloads
    well under every provider's per-image limit.
    """
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGB")
    w, h = im.size
    scale = max_edge / max(w, h)
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = _io.BytesIO()
    im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    raw = buf.getvalue()
    b64 = _b64.b64encode(raw).decode()
    return raw, "image/jpeg", f"data:image/jpeg;base64,{b64}"

HERE = Path(__file__).parent
REPORTS_DIR = HERE / "reports"

PROMPT = (
    "The image shows a Bluesky conversation on the left and four candidate "
    "memes labeled A, B, C, and D on the right. The original poster used one "
    "of these memes as the reply. Which letter best fits the conversation?\n\n"
    "Respond with ONLY one letter: A, B, C, or D. Do NOT explain."
)


# ---------------------------------------------------------------------------
# Per-provider single-image callers
# ---------------------------------------------------------------------------

def _retry(fn, max_retries: int = 3):
    """Run fn() with exponential backoff on Exception. Returns (result, error)."""
    delay = 1.0
    last: Exception | None = None
    for _ in range(max_retries):
        try:
            return fn(), None
        except Exception as e:  # noqa: BLE001 — provider exceptions vary
            last = e
            time.sleep(delay + random.random())
            delay *= 2
    return None, last


def call_gpt4o(image_path: Path, model: str = "gpt-4o") -> str:
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=key)

    _, _, data_url = prepare_image(image_path)

    def go():
        resp = client.chat.completions.create(
            model=model, max_tokens=16, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        )
        return resp.choices[0].message.content or ""

    text, err = _retry(go)
    if err:
        raise err
    return extract_letter(text) or f"?:{text.strip()[:40]}"


def call_claude(image_path: Path, model: str = "claude-sonnet-4-5-20250929") -> str:
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=key)

    raw, mime, _ = prepare_image(image_path)
    b64 = _b64.b64encode(raw).decode()

    def go():
        resp = client.messages.create(
            model=model,
            max_tokens=5000,
            thinking={"type": "enabled", "budget_tokens": 4000},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": PROMPT},
            ]}],
        )
        # Strip thinking blocks; concatenate text blocks.
        out = "".join(b.text for b in resp.content if b.type == "text")
        return out

    text, err = _retry(go)
    if err:
        raise err
    return extract_letter(text) or f"?:{text.strip()[:40]}"


def call_gemini(image_path: Path, model: str = "gemini-2.5-pro") -> str:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    client = genai.Client(api_key=key)

    img_bytes, mime, _ = prepare_image(image_path)

    def go():
        resp = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime), PROMPT],
        )
        return getattr(resp, "text", "") or ""

    text, err = _retry(go)
    if err:
        raise err
    return extract_letter(text) or f"?:{text.strip()[:40]}"


def call_reallms(image_path: Path, model: str = "llama-4-scout") -> str:
    """REALLMS Llama 4 Scout via the IU school API (OpenAI-compatible endpoint)."""
    from openai import OpenAI
    key = os.environ.get("REALLMS_API_KEY")
    if not key:
        raise RuntimeError("REALLMS_API_KEY is not set.")
    base_url = os.environ.get("REALLMS_BASE_URL",
                              "https://reallms.rescloud.iu.edu/direct/v1")
    client = OpenAI(api_key=key, base_url=base_url)

    _, _, data_url = prepare_image(image_path)

    def go():
        resp = client.chat.completions.create(
            model=model, max_tokens=16, temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
        )
        return resp.choices[0].message.content or ""

    text, err = _retry(go)
    if err:
        raise err
    return extract_letter(text) or f"?:{text.strip()[:40]}"


def call_gemini_flash(image_path: Path) -> str:
    """Free-tier-compatible Gemini fallback when 2.5-pro requires billing."""
    return call_gemini(image_path, model="gemini-2.5-flash")


CALLERS = {
    "gpt-4o": call_gpt4o,
    "claude-sonnet-4-5": call_claude,         # dated: claude-sonnet-4-5-20250929
    "gemini-2.5-pro": call_gemini,
    "gemini-2.5-flash": call_gemini_flash,    # free tier
    "reallms": call_reallms,
}


# ---------------------------------------------------------------------------
# Gold parsing
# ---------------------------------------------------------------------------

def parse_gold(arg: str | None, items: list[Path]) -> dict[str, str]:
    """Returns {filename: letter}. Empty dict if no gold provided."""
    if not arg:
        return {}
    p = Path(arg)
    if p.exists() and p.suffix == ".json":
        return {str(k): str(v).upper() for k, v in json.loads(p.read_text()).items()}
    # Treat as a string of letters, one per item in sorted order.
    letters = arg.strip().upper()
    if len(letters) != len(items) or not all(c in "ABCD" for c in letters):
        sys.exit(f"[error] --gold string must be {len(items)} letters from A-D; got {arg!r}")
    return {it.name: letters[i] for i, it in enumerate(items)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: list[dict], models: list[str], gold: dict[str, str], path: Path) -> None:
    """Per-item picks table + per-model accuracy summary."""
    items = sorted({r["item"] for r in rows})
    by_pair = {(r["item"], r["model"]): r["pick"] for r in rows}

    head = "| Item | " + " | ".join(models) + (" | Gold |" if gold else " |")
    sep = "|---|" + "---|" * (len(models) + (1 if gold else 0))
    lines = [head, sep]
    for it in items:
        row = [it]
        for m in models:
            row.append(by_pair.get((it, m), "—"))
        if gold:
            row.append(gold.get(it, "?"))
        lines.append("| " + " | ".join(row) + " |")

    if gold:
        lines.append("")
        lines.append("**Accuracy:**")
        for m in models:
            picks = [by_pair.get((it, m), "") for it in items]
            n = sum(1 for it in items if it in gold)
            ok = sum(1 for it in items if by_pair.get((it, m)) == gold.get(it))
            acc = ok / n if n else 0.0
            lines.append(f"- `{m}`: {ok}/{n} ({acc:.2%})")

    path.write_text("\n".join(lines) + "\n")


def plot_picks_matrix(rows: list[dict], models: list[str], items: list[str], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    by_pair = {(r["item"], r["model"]): r["pick"] for r in rows}
    letter_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    grid = np.full((len(items), len(models)), -1, dtype=int)
    for i, it in enumerate(items):
        for j, m in enumerate(models):
            p = by_pair.get((it, m), "")
            if p in letter_idx:
                grid[i, j] = letter_idx[p]

    cmap = plt.cm.get_cmap("tab10", 4) if hasattr(plt.cm, "get_cmap") else plt.get_cmap("tab10", 4)
    fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(models) + 1), max(3, 0.5 * len(items) + 1.5)))
    masked = np.ma.masked_where(grid < 0, grid)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(items)
    for i in range(len(items)):
        for j in range(len(models)):
            v = grid[i, j]
            ax.text(j, i, "ABCD"[v] if v >= 0 else "—", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
    ax.set_title("Per-item picks (rows = items, cols = models)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(4)]
    ax.legend(handles, list("ABCD"), title="Letter", loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def plot_letter_distribution(rows: list[dict], models: list[str], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(models) + 2), 4))
    width = 0.18
    x = np.arange(len(models))
    by_model = defaultdict(Counter)
    for r in rows:
        if r["pick"] in "ABCD":
            by_model[r["model"]][r["pick"]] += 1
    for i, letter in enumerate("ABCD"):
        counts = [by_model[m][letter] for m in models]
        ax.bar(x + (i - 1.5) * width, counts, width, label=letter)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Number of items picked")
    ax.set_title("Letter pick distribution by model (position bias check)")
    ax.legend(title="Letter")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return True


def plot_accuracy(rows: list[dict], models: list[str], gold: dict[str, str], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not gold:
        return False

    by_pair = {(r["item"], r["model"]): r["pick"] for r in rows}
    items = sorted({r["item"] for r in rows if r["item"] in gold})
    n = len(items)

    accs, labels = [], []
    for m in models:
        ok = sum(1 for it in items if by_pair.get((it, m)) == gold.get(it))
        accs.append(ok / n if n else 0.0)
        labels.append(f"{m}\n{ok}/{n}")

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(models) + 2), 4))
    bars = ax.bar(models, accs, color="#4c72b0", edgecolor="black", linewidth=0.4)
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=0.9, label="Random (25%)")
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.02,
                f"{a:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Composite-screenshot accuracy (n = {n})")
    plt.xticks(rotation=20, ha="right")
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

def _natural_key(p: Path) -> tuple:
    """Sort '1.png' before '10.png' (numeric prefix order)."""
    stem = p.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run commercial VLMs on composite-screenshot meme items.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_composites.py /path/to/data\n"
            "  python run_composites.py /path/to/data --model gpt-4o\n"
            "  python run_composites.py /path/to/data --gold ABCDABCDAB\n"
            "  python run_composites.py /path/to/data --gold gold.json\n"
        ),
    )
    ap.add_argument("data_root", help="Folder containing one PNG per item.")
    ap.add_argument("--model", nargs="+",
                    default=["gpt-4o", "gemini-2.5-flash", "claude-sonnet-4-5", "reallms"],
                    metavar="MODEL", help=f"Choices: {', '.join(CALLERS)}.")
    ap.add_argument("--gold", default=None,
                    help="Either a string of letters (one per file in sorted order) "
                         "or path to a JSON file mapping filename → letter. "
                         "If omitted, no accuracy is computed.")
    ap.add_argument("--out", default=None, help="Output dir. Default: reports/<timestamp>_composite/")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        sys.exit(f"[error] data_root not found: {data_root}")

    images = sorted(
        [p for p in data_root.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
        key=_natural_key,
    )
    if not images:
        sys.exit(f"[error] no PNG/JPG images found under {data_root}")

    # Validate models
    for m in args.model:
        if m not in CALLERS:
            sys.exit(f"[error] unknown --model {m!r}; choices: {', '.join(CALLERS)}")

    gold = parse_gold(args.gold, images)

    out_dir = Path(args.out) if args.out else (
        REPORTS_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_composite"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nComposite run: n_items={len(images)}  models={args.model}  out={out_dir.relative_to(HERE) if out_dir.is_relative_to(HERE) else out_dir}")
    print("=" * 72)

    rows: list[dict] = []
    for it in images:
        print(f"\n[{it.name}]")
        for m in args.model:
            t0 = time.time()
            try:
                pick = CALLERS[m](it)
                err = None
            except Exception as e:
                pick, err = "ERR", f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            ok_str = ""
            if gold and it.name in gold:
                ok = pick == gold[it.name]
                ok_str = "  ✓" if ok else f"  ✗ (gold={gold[it.name]})"
            print(f"  {m:<22} → {pick}{ok_str}    [{dt:.1f}s]"
                  + (f"  ERR: {err}" if err else ""))
            rows.append({"item": it.name, "model": m, "pick": pick,
                         "gold": gold.get(it.name, ""),
                         "correct": (pick == gold.get(it.name)) if gold and it.name in gold else "",
                         "wall_seconds": round(dt, 2),
                         "error": err or ""})

    write_csv(rows, out_dir / "results.csv")
    write_markdown(rows, args.model, gold, out_dir / "results.md")
    items = sorted({r["item"] for r in rows}, key=lambda s: _natural_key(Path(s)))
    plotted_pm = plot_picks_matrix(rows, args.model, items, out_dir / "picks_matrix.png")
    plotted_ld = plot_letter_distribution(rows, args.model, out_dir / "letter_distribution.png")
    plotted_ac = plot_accuracy(rows, args.model, gold, out_dir / "accuracy.png")

    print(f"\n→ {out_dir.relative_to(HERE) if out_dir.is_relative_to(HERE) else out_dir}/")
    print("  results.csv   results.md")
    if plotted_pm: print("  picks_matrix.{png,pdf}")
    if plotted_ld: print("  letter_distribution.{png,pdf}")
    if plotted_ac: print("  accuracy.{png,pdf}")
    if not (plotted_pm and plotted_ld):
        print("  [!] matplotlib not installed — plots skipped.")

    print("\n" + (out_dir / "results.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
