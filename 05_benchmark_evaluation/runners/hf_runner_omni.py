"""Runner for Qwen2.5-Omni-7B (local GPU inference)."""
from __future__ import annotations

import os
from collections import Counter
from functools import lru_cache

from .base import MCItem, build_prompt, extract_letter, prepare_image_pil, split_conv_by_images

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
SC_SAMPLES_DEFAULT = 5
SC_TEMPERATURE = 0.7


def _pick_device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def _load():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    device, dtype = _pick_device_and_dtype()
    print(f"[qwen25-omni] loading {MODEL_ID} on {device} ({dtype})...")
    kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    print("[qwen25-omni] ready.")
    return model, processor, device


def _fetch_url_image(url: str):
    import urllib.request, io
    from PIL import Image
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception:
        return None


def _build_conversation(item: MCItem) -> list[dict]:
    import re as _re
    prompt_text = build_prompt(item)
    conv_blocks = split_conv_by_images(item.conversation_text, item.context_images)

    user_content: list[dict] = []

    if "Conversation:" in prompt_text:
        preamble, rest = prompt_text.split("Conversation:", 1)
        user_content.append({"type": "text", "text": preamble.strip() + "\n\nConversation:"})
        for block in conv_blocks:
            if block["type"] == "text":
                user_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image_ref":
                img = _fetch_url_image(block["url"])
                if img:
                    user_content.append({"type": "image", "image": img})
                else:
                    user_content.append({"type": "text",
                                         "text": f"[{block.get('label','image')} unavailable]"})
        if "The four candidate meme images" in rest:
            tail = _re.sub(r".*?\n\nThe four", "\n\nThe four", rest, flags=_re.DOTALL)
            user_content.append({"type": "text", "text": tail.strip()})
    else:
        user_content.append({"type": "text", "text": prompt_text})

    user_content.append({"type": "text", "text": "\n--- Candidate meme images ---"})
    for letter in "ABCD":
        user_content.append({"type": "text", "text": f"Image {letter}:"})
        user_content.append({"type": "image", "image": prepare_image_pil(item.images[letter])})

    return [
        {"role": "system",
         "content": [{"type": "text",
                      "text": "You are a careful meme-selection assistant. "
                              "Respond with only one letter."}]},
        {"role": "user", "content": user_content},
    ]


def _generate_one(item: MCItem, do_sample: bool) -> str:
    import torch
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as e:
        raise RuntimeError("qwen-omni-utils is required.") from e

    model, processor, device = _load()
    conversation = _build_conversation(item)
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    audio_inputs, image_inputs, video_inputs = process_mm_info(
        conversation, use_audio_in_video=False
    )
    inputs = processor(
        text=[text], audio=audio_inputs, images=image_inputs,
        videos=video_inputs, padding=True, return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        gen_kwargs: dict = {"max_new_tokens": 16, "return_audio": False}
        if do_sample:
            gen_kwargs.update({"do_sample": True, "temperature": SC_TEMPERATURE})
        else:
            gen_kwargs["do_sample"] = False
        out = model.generate(**inputs, **gen_kwargs)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def run(item: MCItem, use_self_consistency: bool | None = None,
        n_samples: int = SC_SAMPLES_DEFAULT) -> str:
    if use_self_consistency is None:
        use_self_consistency = os.environ.get("QWEN_OMNI_USE_SC", "0") == "1"
    if not use_self_consistency:
        decoded = _generate_one(item, do_sample=False)
        return extract_letter(decoded) or f"?:{decoded[:40]}"
    votes, raw_outputs = [], []
    for _ in range(n_samples):
        decoded = _generate_one(item, do_sample=True)
        raw_outputs.append(decoded)
        letter = extract_letter(decoded)
        if letter:
            votes.append(letter)
    if not votes:
        return f"?:{raw_outputs[0][:40] if raw_outputs else 'no_output'}"
    return Counter(votes).most_common(1)[0][0]
