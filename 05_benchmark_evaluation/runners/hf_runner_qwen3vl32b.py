"""Runner for Qwen3-VL-32B-Instruct (Thinking variant) — local GPU inference.

Qwen3-VL-32B-Thinking is the visual reasoning successor to QVQ-72B-Preview,
featuring chain-of-thought multimodal reasoning. Fits comfortably on A100 40GB
in 4-bit quantization (~16GB VRAM).

Set QWEN3VL_THINKING=0 to use the non-thinking (Instruct) variant instead.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from .base import MCItem, build_prompt, prepare_image_pil, split_conv_by_images

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
USE_THINKING = os.environ.get("QWEN3VL_THINKING", "1") == "1"

# Thinking mode needs more tokens for CoT chain
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("QWEN3VL_MAX_TOKENS", "2048"))


def _pick_device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def _load():
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
    )
    try:
        from transformers import Qwen3VLForConditionalGeneration as _ModelClass
    except ImportError:
        from transformers import AutoModel as _ModelClass
    import torch

    device, dtype = _pick_device_and_dtype()
    print(f"[qwen3-vl-32b] loading {MODEL_ID} on {device} ({dtype}, 4bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    kwargs = {"quantization_config": bnb, "torch_dtype": dtype}
    if device == "cuda":
        kwargs["device_map"] = "auto"

    model = _ModelClass.from_pretrained(MODEL_ID, trust_remote_code=True, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"[qwen3-vl-32b] ready (thinking={USE_THINKING}).")
    return model, processor, device


def _fetch_url_image(url: str):
    import urllib.request, io
    from PIL import Image
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            img = Image.open(io.BytesIO(r.read())).convert("RGB")
        return prepare_image_pil(img)  # resize to 512px
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
                img = _fetch_url_image(block["url"])
                if img:
                    content.append({"type": "image", "image": img})
                    ctx_n += 1
                else:
                    content.append({"type": "text",
                                    "text": f"[{block.get('label','image')} unavailable]"})
        if "The four candidate" in rest:
            tail = re.sub(r".*?\n\nThe four", "\n\nThe four", rest, flags=re.DOTALL)
            content.append({"type": "text", "text": tail.strip()})
    else:
        content.append({"type": "text", "text": prompt_text})

    content.append({"type": "text", "text": "\n--- Candidate meme images ---"})
    for letter in "ABCD":
        content.append({"type": "text", "text": f"Image {letter}:"})
        content.append({"type": "image", "image": prepare_image_pil(item.images[letter])})
    return content


# Reasoning models scatter letters through CoT — take the LAST A/B/C/D
_LAST_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_final_letter(text: str) -> str | None:
    if not text:
        return None
    matches = _LAST_LETTER_RE.findall(text.upper())
    return matches[-1] if matches else None


def run(item: MCItem) -> str:
    import torch
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise RuntimeError("qwen-vl-utils required: pip install qwen-vl-utils") from e

    model, processor, device = _load()

    # Thinking mode: enable via chat template
    enable_thinking = USE_THINKING
    messages = [{"role": "user", "content": _build_content(item)}]

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older processor versions don't have enable_thinking
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            do_sample=False,
        )
    trimmed = out[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    torch.cuda.empty_cache()

    # For thinking mode, extract last letter after </think>
    if enable_thinking and "</think>" in decoded:
        answer_part = decoded.split("</think>")[-1]
    else:
        answer_part = decoded

    pick = _extract_final_letter(answer_part) or _extract_final_letter(decoded)
    return pick or f"?:{decoded[:60]}"
