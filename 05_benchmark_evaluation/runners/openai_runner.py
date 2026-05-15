"""Runner for OpenAI GPT-4o.

Expects OPENAI_API_KEY in env or .env. Uses the standard chat-completions
endpoint with multimodal input. Sends the same 4-labeled-image structure as
the other runners. Retries transient errors with exponential backoff.

Notes from the model-selection review:
  - GPT-4o was the BEST zero-shot model in MemeQA (~59.6 macro avg).
  - In MemeReaCon it beats every open VLM but ranks below the four other VRMs
    (QvQ, Grok-3, Claude-3.7, Gemini-2.5-Pro on CMI-C accuracy). Treat as a
    strong upper-bound for closed models, not the single best frontier model.
"""
from __future__ import annotations

import os
import random
import time

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from .base import MCItem, build_prompt, extract_letter, prepare_image_data_url, split_conv_by_images

DEFAULT_MODEL = "gpt-4o"


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )
    return OpenAI(api_key=key)


def _build_content(item: MCItem) -> list[dict]:
    """
    Build multimodal content with inline conversation images.
    Context images appear at their exact position in the conversation
    via [IMAGE:N] placeholders. Candidate meme images follow after.
    """
    prompt_text = build_prompt(item)

    # Split conversation by [IMAGE:N] placeholders → inline images
    conv_blocks = split_conv_by_images(item.conversation_text, item.context_images)

    # Rebuild prompt with inline images
    content: list[dict] = []

    # Preamble (everything before "Conversation:" in the prompt)
    if "Conversation:" in prompt_text:
        preamble, _ = prompt_text.split("Conversation:", 1)
        content.append({"type": "text", "text": preamble.strip() + "\n\nConversation:"})
    else:
        content.append({"type": "text", "text": prompt_text})
        # No inline image splitting needed
        for letter in "ABCD":
            content.append({"type": "text", "text": f"Image {letter}:"})
            content.append({"type": "image_url",
                            "image_url": {"url": prepare_image_data_url(item.images[letter])}})
        return content

    # Inline conversation blocks
    for block in conv_blocks:
        if block["type"] == "text":
            content.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image_ref":
                import base64 as _b64, urllib.request as _ur
                try:
                    with _ur.urlopen(block["url"], timeout=10) as r:
                        raw = r.read()
                    b64 = _b64.b64encode(raw).decode()
                    data_url = f"data:image/jpeg;base64,{b64}"
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
                except Exception:
                    content.append({"type": "text", "text": f"[{block.get('label','image')} unavailable]"})

    # Guidelines + final instruction (everything after conversation in prompt)
    if "The four candidate meme images" in prompt_text:
        _, tail = prompt_text.split("Conversation:", 1)
        # Find where conversation ends and guidelines begin
        import re as _re
        # Remove the conv part (already rendered inline), keep guidelines
        conv_stripped = _re.sub(r".*?\n\nThe four", "\n\nThe four",
                                tail, flags=_re.DOTALL)
        content.append({"type": "text", "text": conv_stripped.strip()})

    # Candidate meme images
    content.append({"type": "text", "text": "\n--- Candidate meme images ---"})
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
            return extract_letter(text) or f"?:{text[:40]}"
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
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last_err!r}")
