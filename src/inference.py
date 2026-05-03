import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.config import MAX_NEW_TOKENS


@torch.no_grad()
def generate_caption(
    image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: torch.device,
    prompt: str | None = None,
) -> str:
    if prompt:
        inputs = processor(image, prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption
