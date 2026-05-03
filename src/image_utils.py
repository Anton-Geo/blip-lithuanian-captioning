from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image


def load_image_from_path(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
