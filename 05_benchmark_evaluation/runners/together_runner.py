"""Runner for Llama-4-Scout via OpenRouter API.

Expects OPENROUTER_API_KEY in env or .env.
OpenRouter uses an OpenAI-compatible API endpoint.

Install: pip install openai

Model: meta-llama/llama-4-scout
"""
from __future__ import annotations

import os
import random
import time
import base64

from .base import MCItem, build_prompt, extract_letter, prepare_image_bytes, split_conv_by_images

DEFAULT_MODEL = "meta-llama/llama-4-scout"


def _client():
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
    )


def _image_data_url(path) -> str:
    raw, mime = prepare_image_bytes(path)
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def _fetch_url_b64(url: str) -> str | None:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            raw = r.read()
        return f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"
    except Exception:
        return None


def _build_content(item: MCItem) -> list[dict]:
    prompt_text = build_prompt(item)
    conv_blocks = split_conv_by_images(item.conversation_text, item.context_images)

    content: list[dict] = []

    if "Conversation:" in prompt_text:
        preamble, rest = prompt_text.split("Conversation:", 1)
        content.append({"type": "text", "text": preamble.strip() + "\n\nConversation:"})

        ctx_n = 0
        for block in conv_blocks:
            if block["type"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image_ref" and ctx_n < 2:
                data_url = _fetch_url_b64(block["url"])
                if data_url:
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                    ctx_n += 1
                else:
                    content.append({"type": "text",
                                    "text": f"[{block.get('label','image')} unavailable]"})

        import re as _re
        if "The four candidate" in rest:
            tail = _re.sub(r".*?\n\nThe four", "\n\nThe four", rest, flags=_re.DOTALL)
            content.append({"type": "text", "text": tail.strip()})
    else:
        content.append({"type": "text", "text": prompt_text})

    content.append({"type": "text", "text": "\n--- Candidate meme images ---"})
    for letter in "ABCD":
        content.append({"type": "text", "text": f"Image {letter}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_data_url(item.images[letter])}
        })
    return content


def run(
    item: MCItem,
    model: str = DEFAULT_MODEL,
    max_retries: int = 4,
    max_tokens: int = 16,
) -> str:
    client = _client()
    messages = [{"role": "user", "content": _build_content(item)}]

    delay = 1.0
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            return extract_letter(text) or f"?:{text[:40]}"
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = ("rate", "timeout", "429", "500", "502", "503", "504")
            if any(s in msg for s in transient):
                time.sleep(delay + random.random())
                delay *= 2
                continue
            raise
    raise RuntimeError(f"OpenRouter call failed after {max_retries} retries: {last_err!r}")
