"""Runner for Google Gemini 2.5 Pro via the unified `google-genai` SDK."""
from __future__ import annotations

import os
import random
import time

from google import genai
from google.genai import types

from .base import MCItem, build_prompt, extract_letter, prepare_image_bytes, split_conv_by_images

DEFAULT_MODEL = "gemini-2.5-pro"


def _client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=key)


def _image_part(path) -> types.Part:
    raw, mime = prepare_image_bytes(path)
    return types.Part.from_bytes(data=raw, mime_type=mime)


def _url_image_part(url: str) -> types.Part | None:
    """Fetch CDN URL and return Gemini Part."""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            raw = r.read()
        return types.Part.from_bytes(data=raw, mime_type="image/jpeg")
    except Exception:
        return None


def _build_contents(item: MCItem) -> list:
    """Prompt with inline context images, then 4 candidate meme images."""
    import re as _re
    prompt_text = build_prompt(item)
    conv_blocks = split_conv_by_images(item.conversation_text, item.context_images)

    contents = []

    if "Conversation:" in prompt_text:
        preamble, rest = prompt_text.split("Conversation:", 1)
        contents.append(preamble.strip() + "\n\nConversation:")

        for block in conv_blocks:
            if block["type"] == "text":
                contents.append(block["text"])
            elif block["type"] == "image_ref":
                part = _url_image_part(block["url"])
                if part:
                    contents.append(part)
                else:
                    contents.append(f"[{block.get('label','image')} unavailable]")

        if "The four candidate meme images" in rest:
            tail = _re.sub(r".*?\n\nThe four", "\n\nThe four", rest, flags=_re.DOTALL)
            contents.append(tail.strip())
    else:
        contents.append(prompt_text)

    contents.append("\n--- Candidate meme images ---")
    for letter in "ABCD":
        contents.append(f"Image {letter}:")
        contents.append(_image_part(item.images[letter]))
    return contents


def run(
    item: MCItem,
    model: str = DEFAULT_MODEL,
    max_retries: int = 4,
    max_tokens: int = 256,  # thinking + answer
) -> str:
    client = _client()
    delay = 1.0
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=_build_contents(item),
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=128),  # disable thinking
                ),
            )
            text = (resp.text or "").strip()
            return extract_letter(text) or f"?:{text[:40]}"
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = ("rate", "timeout", "deadline", "429", "500", "502", "503", "504")
            if any(s in msg for s in transient):
                time.sleep(delay + random.random())
                delay *= 2
                continue
            raise
    raise RuntimeError(f"Gemini call failed after {max_retries} retries: {last_err!r}")
