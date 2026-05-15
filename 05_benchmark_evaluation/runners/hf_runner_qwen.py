"""Runner for Qwen2.5-VL-7B-Instruct (local GPU inference)."""
from __future__ import annotations

from functools import lru_cache

from .base import MCItem, build_prompt, extract_letter, prepare_image_pil, split_conv_by_images

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def _pick_device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device, dtype = _pick_device_and_dtype()
    print(f"[qwen25-vl] loading {MODEL_ID} on {device} ({dtype})...")
    kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"[qwen25-vl] ready.")
    return model, processor, device


def _build_content(item: MCItem) -> list[dict]:
    """Prompt text with inline context images, then 4 labeled candidate meme images."""
    prompt_text = build_prompt(item)
    conv_blocks = split_conv_by_images(item.conversation_text, item.context_images)

    content: list[dict] = []

    # Split prompt into preamble + conversation + guidelines
    if "Conversation:" in prompt_text:
        preamble, rest = prompt_text.split("Conversation:", 1)
        content.append({"type": "text", "text": preamble.strip() + "\n\nConversation:"})

        # Inline conversation blocks (text + context images)
        for block in conv_blocks:
            if block["type"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image_ref":
                ctx_img_count = sum(1 for b in content if b.get("type") == "image")
                if False:  # cap removed; images resized to 512px instead
                    content.append({"type": "text",
                                    "text": f"[{block.get('label','image')}: omitted]"})
                else:
                    try:
                        import urllib.request
                        from PIL import Image
                        import io
                        with urllib.request.urlopen(block["url"], timeout=10) as r:
                            img = Image.open(io.BytesIO(r.read())).convert("RGB")
                        img = prepare_image_pil(img)  # resize to 512px to avoid OOM
                        content.append({"type": "image", "image": img})
                    except Exception:
                        content.append({"type": "text",
                                        "text": f"[{block.get('label','image')} unavailable]"})

        # Guidelines tail
        import re as _re
        if "The four candidate meme images" in rest:
            tail = _re.sub(r".*?\n\nThe four", "\n\nThe four", rest, flags=_re.DOTALL)
            content.append({"type": "text", "text": tail.strip()})
    else:
        content.append({"type": "text", "text": prompt_text})

    # Candidate meme images
    content.append({"type": "text", "text": "\n--- Candidate meme images ---"})
    for letter in "ABCD":
        content.append({"type": "text", "text": f"Image {letter}:"})
        content.append({"type": "image", "image": prepare_image_pil(item.images[letter])})
    return content


def run(item: MCItem) -> str:
    import torch
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise RuntimeError(
            "qwen-vl-utils is required. Install: pip install -r requirements-gpu.txt"
        ) from e

    model, processor, device = _load()
    messages = [{"role": "user", "content": _build_content(item)}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    return extract_letter(decoded) or f"?:{decoded[:40]}"
