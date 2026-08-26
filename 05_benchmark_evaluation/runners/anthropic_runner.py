"""Runner for Anthropic Claude Sonnet 4.5.

Expects ANTHROPIC_API_KEY in env or .env. Uses the messages endpoint with
multimodal input. Sends the same 4-labeled-image structure as the other runners.

Model history note:
  Originally targeted claude-3-7-sonnet-20250219 (the model used in MemeReaCon
  CMI-C reporting). That snapshot was deprecated when Anthropic rolled out the
  Claude 4 family; the current default is claude-sonnet-4-5-20250929.

Default config (per Juyoung's review, May 2026):
  - **Extended thinking is OFF by default.** Enabling it triples the cost
    (output tokens dominate the bill at $15/M) and we don't need it for the
    meme-selection comparison. Pass `use_thinking=True` to opt in if you want
    to reproduce the published "Claude w/ thinking" numbers from MemeReaCon.
  - When `use_thinking=True`, the API forces temperature=1 and requires
    max_tokens > thinking budget. We strip thinking blocks from the response
    and only return the final text answer.
"""
from __future__ import annotations

import os
import random
import time

import anthropic
from anthropic import APIConnectionError, APIStatusError, RateLimitError

from .base import MCItem, build_prompt, extract_letter, prepare_image_bytes, split_conv_by_images

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
THINKING_BUDGET_TOKENS = 4000  # Tokens reserved for hidden reasoning.


def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file."
        )
    return anthropic.Anthropic(api_key=key)


def _image_block(path) -> dict:
    """Resize to 512 px JPEG (per budget plan), then base64-encode for Anthropic."""
    import base64

    raw, mime = prepare_image_bytes(path)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": base64.b64encode(raw).decode(),
        },
    }


def _url_image_block(url: str) -> dict:
    """Fetch a CDN image URL and encode as base64 for Anthropic."""
    import base64, urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            raw = r.read()
        # 실제 포맷 감지 후 webp는 jpeg로 변환
        import imghdr
        fmt = imghdr.what(None, h=raw)
        if fmt == "webp":
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            raw = buf.getvalue()
            fmt = "jpeg"
        mime = f"image/{fmt}" if fmt else "image/jpeg"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime,
                       "data": base64.b64encode(raw).decode()},
        }
    except Exception:
        return {"type": "text", "text": f"[context image unavailable: {url}]"}


def _build_content(item: MCItem) -> list[dict]:
    """Build content with inline conversation images at [IMAGE:N] positions."""
    prompt_text = build_prompt(item)
    conv_blocks = split_conv_by_images(item.conversation_text, item.context_images)

    content: list[dict] = []

    if "Conversation:" in prompt_text:
        preamble, _ = prompt_text.split("Conversation:", 1)
        content.append({"type": "text", "text": preamble.strip() + "\n\nConversation:"})
    else:
        content.append({"type": "text", "text": prompt_text})
        for letter in "ABCD":
            content.append({"type": "text", "text": f"Image {letter}:"})
            content.append(_image_block(item.images[letter]))
        return content

    # Inline conversation blocks
    for block in conv_blocks:
        if block["type"] == "text":
            content.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image_ref":
            content.append(_url_image_block(block["url"]))

    # Guidelines tail
    if "The four candidate meme images" in prompt_text:
        _, tail = prompt_text.split("Conversation:", 1)
        import re as _re
        conv_stripped = _re.sub(r".*?\n\nThe four", "\n\nThe four",
                                tail, flags=_re.DOTALL)
        content.append({"type": "text", "text": conv_stripped.strip()})

    # Candidate meme images
    content.append({"type": "text", "text": "\n--- Candidate meme images ---"})
    for letter in "ABCD":
        content.append({"type": "text", "text": f"Image {letter}:"})
        content.append(_image_block(item.images[letter]))
    return content


def run(
    item: MCItem,
    model: str = DEFAULT_MODEL,
    max_retries: int = 4,
    max_tokens: int = 5000,  # must exceed THINKING_BUDGET_TOKENS when thinking on
    use_thinking: bool = False,  # OFF by default per Juyoung (May 2026)
) -> str:
    messages = [{"role": "user", "content": _build_content(item)}]
    client = _client()
    delay = 1.0
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            kwargs: dict = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if use_thinking:
                # Extended thinking forces temperature=1 internally.
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": THINKING_BUDGET_TOKENS,
                }
            else:
                kwargs["temperature"] = 0
            resp = client.messages.create(**kwargs)
            # Response may contain "thinking" blocks before the final "text" block.
            text = ""
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    text = block.text
                    break
            return extract_letter(text) or f"?:{text[:40]}"
        except (RateLimitError, APIConnectionError) as e:
            last_err = e
            time.sleep(delay + random.random())
            delay *= 2
        except APIStatusError as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status in (408, 500, 502, 503, 504):
                time.sleep(delay + random.random())
                delay *= 2
            else:
                raise
    raise RuntimeError(f"Anthropic call failed after {max_retries} retries: {last_err!r}")
