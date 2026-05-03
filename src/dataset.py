from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BlipCaptionDataset(Dataset):
    def __init__(self, annotations_path: str | Path, images_dir: str | Path, processor, split: str):
        self.images_dir = Path(images_dir)
        self.processor = processor

        df = pd.read_csv(annotations_path)

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
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        labels = encoding["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        return encoding
