"""Runner for QvQ-72B-Preview (Qwen reasoning VLM, local GPU inference)."""
from __future__ import annotations

import os
import re
from functools import lru_cache

from .base import MCItem, build_prompt, prepare_image_pil, split_conv_by_images

MODEL_ID = "Qwen/QVQ-72B-Preview"
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("QVQ_MAX_NEW_TOKENS", "4096"))


def _pick_device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    device, dtype = _pick_device_and_dtype()
    use_4bit = os.environ.get("QVQ_USE_4BIT", "0") == "1"
    print(f"[qvq] loading {MODEL_ID} on {device} ({dtype}, 4bit={use_4bit})...")
    kwargs: dict = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    if use_4bit:
        from transformers import BitsAndBytesConfig
        import torch as _torch
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=_torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    if device != "cuda" and not use_4bit:
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"[qvq] ready.")
    return model, processor, device


def _fetch_url_image(url: str):
    import urllib.request, io
    from PIL import Image
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception:
        return None


def _build_content(item: MCItem) -> list[dict]:
    """Prompt with inline context images, then 4 candidate meme images."""
    prompt_text = build_prompt(item)
    conv_blocks = split_conv_by_images(item.conversation_text, item.context_images)

    content: list[dict] = []

    if "Conversation:" in prompt_text:
        preamble, rest = prompt_text.split("Conversation:", 1)
        content.append({"type": "text", "text": preamble.strip() + "\n\nConversation:"})
        for block in conv_blocks:
            if block["type"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image_ref":
                img = _fetch_url_image(block["url"])
                if img:
                    content.append({"type": "image", "image": img})
                else:
                    content.append({"type": "text",
                                    "text": f"[{block.get('label','image')} unavailable]"})
        if "The four candidate meme images" in rest:
            tail = re.sub(r".*?\n\nThe four", "\n\nThe four", rest, flags=re.DOTALL)
            content.append({"type": "text", "text": tail.strip()})
    else:
        content.append({"type": "text", "text": prompt_text})

    content.append({"type": "text", "text": "\n--- Candidate meme images ---"})
    for letter in "ABCD":
        content.append({"type": "text", "text": f"Image {letter}:"})
        content.append({"type": "image", "image": prepare_image_pil(item.images[letter])})
    return content


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
        raise RuntimeError("qwen-vl-utils is required.") from e

    model, processor, device = _load()
    messages = [{"role": "user", "content": _build_content(item)}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    pick = _extract_final_letter(decoded)
    return pick or f"?:{decoded[:60]}"
