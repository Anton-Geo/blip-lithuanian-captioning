import torch
from peft import PeftModel
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.config import MODEL_NAME


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_blip_model(model_name: str = MODEL_NAME):
    device = get_device()

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return processor, model, device


def load_blip_lora_model(
    adapter_dir: str,
    base_model_name: str = MODEL_NAME,
):
    device = get_device()

    processor = BlipProcessor.from_pretrained(adapter_dir)

    base_model = BlipForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)

    model.to(device)
    model.eval()

    return processor, model, device
