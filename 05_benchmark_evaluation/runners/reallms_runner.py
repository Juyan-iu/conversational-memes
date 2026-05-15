"""Runner for IU REALLMS (Llama 4 Scout and other REALLMS-hosted models).

Endpoint is OpenAI-compatible. Set REALLMS_API_KEY in env or in a .env file.

Sends 4 labeled image blocks (one per option A/B/C/D) + the conversation text
as a single user message. Retries transient errors with exponential backoff.
"""
from __future__ import annotations

import os
import random
import time

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from .base import MCItem, build_prompt, extract_letter, prepare_image_data_url

BASE_URL = "https://reallms.rescloud.iu.edu/direct/v1"
DEFAULT_MODEL = "llama-4-scout"


def _client() -> OpenAI:
    key = os.environ.get("REALLMS_API_KEY")
    if not key:
        raise RuntimeError(
            "REALLMS_API_KEY is not set. Get a key at https://one.iu.edu/launch-task/iu/rt-projects "
            "and add it to your .env file."
        )
    return OpenAI(api_key=key, base_url=BASE_URL)


def list_models() -> list[str]:
    return [m.id for m in _client().models.list().data]


def chat_text(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 64) -> str:
    """Text-only convenience helper (used by smoke_test.py)."""
    resp = _client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _build_content(item: MCItem) -> list[dict]:
    """Prompt text followed by 4 labeled image blocks."""
    content: list[dict] = [{"type": "text", "text": build_prompt(item)}]
    for letter in "ABCD":
        content.append({"type": "text", "text": f"Image {letter}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": prepare_image_data_url(item.images[letter])}})
    return content


def run(
    item: MCItem,
    model: str = DEFAULT_MODEL,
    max_retries: int = 4,
    max_tokens: int = 16,
) -> str:
    """Run one item through REALLMS and return a single letter A/B/C/D.

    Raises after `max_retries` transient failures.
    """
    messages = [{"role": "user", "content": _build_content(item)}]
    client = _client()
    delay = 1.0
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                messages=messages,
            )
            text = (resp.choices[0].message.content or "").strip()
            letter = extract_letter(text)
            return letter or f"?:{text[:40]}"
        except (RateLimitError, APIConnectionError) as e:
            last_err = e
            time.sleep(delay + random.random())
            delay *= 2
        except APIStatusError as e:
            last_err = e
            if e.status_code in (408, 500, 502, 503, 504):
                time.sleep(delay + random.random())
                delay *= 2
            else:
                raise
    raise RuntimeError(f"REALLMS call failed after {max_retries} retries: {last_err!r}")
