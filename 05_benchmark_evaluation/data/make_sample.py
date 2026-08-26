"""Generate one placeholder MC item to smoke-test the pipeline.

The real 1,000-item dataset will live elsewhere (point --data-root at it).
This script only produces a single fake item so `python pilot.py --num 1`
has something to run against.

Output layout (matches the data schema in README.md):
  data/placeholder/
    post_example/
      info.txt                     # Bluesky-style conversation (real bsky link)
      post_example.jpeg            # "correct" meme (filename stem = folder name)
      distractor_lex.jpeg          # lexical distractor
      distractor_vis.jpeg          # visual distractor
      distractor_rand.jpeg         # random distractor
      labels.json                  # maps distractor filenames to type tags
  data/samples/test_letter_B.png   # unchanged — used by smoke_test.py

Run once:
  python data/make_sample.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
PLACEHOLDER_ROOT = HERE / "placeholder"
SMOKE_SAMPLES = HERE / "samples"

# A real Bluesky post — lifted from the data.txt example Juyan shared. Using a
# real URL so the format matches exactly; swap any content in info.txt when
# building the real 1,000-item dataset.
PLACEHOLDER_ITEM_ID = "post_example"
PLACEHOLDER_INFO_TEXT = """===== ROOT POST =====
Translation: Pete may well go down for this. But I'm not going with him.

Quote:
Reporter: If there were a second strike that killed wounded people, would that be legal?

Trump: I don't know that happened and Pete said he did not even know what people were talking about. I wouldn't have wanted a second strike. The first strike was very lethal. It was fine.

===== REPLY POST =====
That's what he's there for

Parent Comment:
Pete is taking the heat off of the Epstein Files.

===== URL =====
https://bsky.app/profile/did:plc:rjr6nfdzrfngjgc34jjyty6a/post/3m6v2kgsuj223
"""

# Four synthetic memes. Colors + captions are arbitrary — the point is just to
# have visually distinct files the model can choose between.
MEMES = [
    # (filename, role, top_caption, bottom_caption, background_color, swatch_color)
    (f"{PLACEHOLDER_ITEM_ID}.jpeg",  "correct",  "TAKING THE HEAT",   "(for the Files)",  (40, 40, 40),   (220, 80, 80)),
    ("distractor_lex.jpeg",           "lexical",  "PETE HOT MIC",      "PETE'S MISTAKE",   (40, 40, 40),   (100, 180, 220)),
    ("distractor_vis.jpeg",           "visual",   "LITERAL FIRE HEAT", "HOT STUFF",        (40, 40, 40),   (180, 220, 100)),
    ("distractor_rand.jpeg",          "random",   "CAT AT 3AM",        "ZOOMIES",          (40, 40, 40),   (200, 140, 220)),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _meme(path: Path, top: str, bottom: str, bg, swatch) -> None:
    img = Image.new("RGB", (512, 512), bg)
    draw = ImageDraw.Draw(img)
    font = _font(40)
    draw.rectangle([40, 160, 472, 352], fill=swatch)
    for y, text in ((30, top), (420, bottom)):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((512 - w) / 2, y), text, fill="white", font=font)
    img.save(path)
    print(f"  wrote {path.relative_to(HERE.parent)}")


def _letter_image(letter: str, path: Path) -> None:
    img = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(img)
    font = _font(320)
    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((512 - w) / 2 - bbox[0], (512 - h) / 2 - bbox[1]),
              letter, fill="black", font=font)
    img.save(path)
    print(f"  wrote {path.relative_to(HERE.parent)}")


def make_placeholder_item() -> None:
    item_dir = PLACEHOLDER_ROOT / PLACEHOLDER_ITEM_ID
    item_dir.mkdir(parents=True, exist_ok=True)

    (item_dir / "info.txt").write_text(PLACEHOLDER_INFO_TEXT)
    print(f"  wrote {(item_dir / 'info.txt').relative_to(HERE.parent)}")

    labels: dict[str, str] = {}
    for fname, role, top, bottom, bg, swatch in MEMES:
        _meme(item_dir / fname, top, bottom, bg, swatch)
        if role != "correct":
            labels[fname] = role

    labels_path = item_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"  wrote {labels_path.relative_to(HERE.parent)}")


def make_smoke_test_image() -> None:
    SMOKE_SAMPLES.mkdir(parents=True, exist_ok=True)
    _letter_image("B", SMOKE_SAMPLES / "test_letter_B.png")


def main() -> None:
    print("Generating one placeholder MC item...")
    make_placeholder_item()
    print("Generating smoke-test image (for `python smoke_test.py`)...")
    make_smoke_test_image()
    print("\nDone.")
    print("  Pilot smoke test:   python pilot.py --num 1")
    print("  REALLMS sanity:     python smoke_test.py")


if __name__ == "__main__":
    main()
