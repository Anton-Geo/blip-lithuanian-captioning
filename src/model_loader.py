import torch
from peft import PeftModel
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from src.config import MODEL_NAME


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mblip_model(model_name: str = MODEL_NAME):
    processor = Blip2Processor.from_pretrained(model_name)

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model.eval()

    return processor, model


def load_mblip_lora_model(
    adapter_dir: str,
    base_model_name: str = MODEL_NAME,
):
    processor = Blip2Processor.from_pretrained(adapter_dir)

    base_model = Blip2ForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    return processor, model
