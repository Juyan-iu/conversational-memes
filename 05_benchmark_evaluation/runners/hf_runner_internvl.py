"""Runner for InternVL3-8B (OpenGVLab) — local GPU inference."""
from __future__ import annotations

from functools import lru_cache

from .base import MCItem, build_prompt, extract_letter, prepare_image_pil, split_conv_by_images

MODEL_ID = "OpenGVLab/InternVL3-8B"
INPUT_SIZE = 448


def _pick_device_and_dtype():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModel, AutoTokenizer

    device, dtype = _pick_device_and_dtype()
    print(f"[internvl3-8b] loading {MODEL_ID} on {device} ({dtype})...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, use_fast=False
    )
    print("[internvl3-8b] ready.")
    return model, tokenizer, device, dtype


def _load_image_patch(img_or_path, device, dtype):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from PIL import Image

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    if isinstance(img_or_path, Image.Image):
        image = img_or_path.convert("RGB")
    else:
        image = prepare_image_pil(img_or_path)
    pixel = transform(image).unsqueeze(0)
    return pixel.to(device).to(dtype)


def _fetch_url_image(url: str):
    """Fetch CDN image URL and return PIL Image."""
    import urllib.request, io
    from PIL import Image
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception:
        return None


def run(item: MCItem) -> str:
    import torch, re as _re

    model, tokenizer, device, dtype = _load()

    pixel_values_list = []
    num_patches_list  = []

    # Build question text with <image> tags for context images
    prompt_text  = build_prompt(item)
    conv_blocks  = split_conv_by_images(item.conversation_text, item.context_images)

    question_parts = []
    if "Conversation:" in prompt_text:
        preamble, rest = prompt_text.split("Conversation:", 1)
        question_parts.append(preamble.strip() + "\n\nConversation:")
        for block in conv_blocks:
            if block["type"] == "text":
                question_parts.append(block["text"])
            elif block["type"] == "image_ref":
                img = _fetch_url_image(block["url"])
                if img:
                    pv = _load_image_patch(img, device, dtype)
                    pixel_values_list.append(pv)
                    num_patches_list.append(pv.size(0))
                    question_parts.append(f"\n<image>\n")
                else:
                    question_parts.append(f"[{block.get('label','image')} unavailable]")
        if "The four candidate meme images" in rest:
            tail = _re.sub(r".*?\n\nThe four", "\n\nThe four", rest, flags=_re.DOTALL)
            question_parts.append(tail.strip())
    else:
        question_parts.append(prompt_text)

    question_parts.append("\n--- Candidate meme images ---")
    for letter in "ABCD":
        pv = _load_image_patch(item.images[letter], device, dtype)
        pixel_values_list.append(pv)
        num_patches_list.append(pv.size(0))
        question_parts.append(f"Image {letter}: <image>")

    pixel_values = torch.cat(pixel_values_list, dim=0)
    question = "\n".join(question_parts)

    generation_config = {"max_new_tokens": 16, "do_sample": False}
    response = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
    )
    return extract_letter(response) or f"?:{response[:40]}"
