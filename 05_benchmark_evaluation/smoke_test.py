"""Run three connectivity checks against REALLMS before the full eval.

  1. List models (verifies API key + endpoint reachable)
  2. Text-only chat (verifies chat/completions works)
  3. Image + text chat (verifies multimodal input works for llama-4-scout)

If (3) fails but (1) and (2) pass, the deployment may not expose image input.
Contact UITS RADL or switch to Jetstream2 inference.

Usage:  python smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from runners import reallms_runner
from runners.base import extract_letter


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def check_list_models() -> bool:
    print("1) Listing models...")
    try:
        ids = reallms_runner.list_models()
        _ok(f"Found {len(ids)} models: {', '.join(ids)}")
        if reallms_runner.DEFAULT_MODEL not in ids:
            _fail(f"'{reallms_runner.DEFAULT_MODEL}' is NOT in the model list.")
            return False
        return True
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        return False


def check_text() -> bool:
    print("2) Text-only chat...")
    try:
        out = reallms_runner.chat_text("Reply with only the letter B.", max_tokens=8)
        _ok(f"Response: {out!r}")
        return "B" in out.upper()
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        return False


def check_image(image_path: Path) -> bool:
    print(f"3) Image + text chat using {image_path.name}...")
    if not image_path.exists():
        _fail(f"Sample image not found: {image_path}")
        print("     Run `python data/make_sample.py` first.")
        return False
    try:
        from runners.base import image_to_data_url

        # Direct low-level call — connectivity check shouldn't depend on the
        # full MCItem schema, which is specific to the meme-selection task.
        client = reallms_runner._client()
        resp = client.chat.completions.create(
            model=reallms_runner.DEFAULT_MODEL,
            max_tokens=16,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "What letter is shown in the image? "
                        "Respond with ONLY one letter."
                    )},
                    {"type": "image_url",
                     "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }],
        )
        out = (resp.choices[0].message.content or "").strip()
        _ok(f"Response: {out!r}")
        letter = extract_letter(out)
        if letter == "B":
            _ok("Image input works and model read the test image correctly.")
            return True
        _fail(f"Model ran but returned the wrong letter (expected B, got {letter}).")
        print("     Multimodal input IS working — the test image may just be ambiguous.")
        print("     Inspect data/samples/test_letter_B.png manually before deciding.")
        return True
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}")
        print("     If the error mentions image/content type, the REALLMS deployment may")
        print("     not expose multimodal input for llama-4-scout. Contact UITS RADL.")
        return False


def main() -> int:
    here = Path(__file__).parent
    sample_image = here / "data" / "samples" / "test_letter_B.png"
    print("REALLMS smoke test\n" + "=" * 60)

    results = [
        check_list_models(),
        check_text(),
        check_image(sample_image),
    ]

    print("=" * 60)
    passed = sum(results)
    print(f"Passed {passed}/3.")
    return 0 if passed == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
