import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.config import MODEL_NAME


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_blip_model():
    device = get_device()

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    return processor, model, device
