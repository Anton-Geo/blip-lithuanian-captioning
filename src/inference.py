import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from src.config import DEFAULT_PROMPT, MAX_NEW_TOKENS


@torch.no_grad()
def generate_caption(
    image: Image.Image,
    processor: Blip2Processor,
    model: Blip2ForConditionalGeneration,
    prompt: str = DEFAULT_PROMPT,
) -> str:
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=3,
    )

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()
