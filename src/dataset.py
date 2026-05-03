from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.config import DEFAULT_PROMPT, MAX_TEXT_LENGTH


class MBlipCaptionDataset(Dataset):
    def __init__(
        self,
        annotations_path: str | Path,
        images_dir: str | Path,
        processor,
        split: str,
        prompt: str = DEFAULT_PROMPT,
    ):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.prompt = prompt

        df = pd.read_csv(annotations_path, encoding="utf-8")

        df = df.dropna(subset=["caption_lt"])
        df = df[df["caption_lt"].astype(str).str.strip() != ""]
        df = df[df["split"] == split].copy()

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.images_dir / row["filename"]
        image = Image.open(image_path).convert("RGB")
        caption = str(row["caption_lt"]).strip()

        encoding = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors="pt",
        ).input_ids

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        encoding = {key: value.squeeze(0) for key, value in encoding.items()}
        encoding["labels"] = labels.squeeze(0)

        return encoding
