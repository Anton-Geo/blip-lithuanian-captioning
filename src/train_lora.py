from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BlipForConditionalGeneration, BlipProcessor
from peft import LoraConfig, get_peft_model

from src.dataset import BlipCaptionDataset


def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    model.train()
    return total_loss / max(len(dataloader), 1)


def train_lora(
    annotations_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "Salesforce/blip-image-captioning-base",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor = BlipProcessor.from_pretrained(model_name)

    model = BlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "key", "value"],
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()

    model.print_trainable_parameters()

    train_dataset = BlipCaptionDataset(
        annotations_path=annotations_path,
        images_dir=images_dir,
        processor=processor,
        split="train",
    )

    val_dataset = BlipCaptionDataset(
        annotations_path=annotations_path,
        images_dir=images_dir,
        processor=processor,
        split="val",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    history = []

    for epoch in range(epochs):
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"Epoch {epoch + 1}, "
                    f"step {step}/{len(train_loader)}, "
                    f"loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_val_loss = evaluate_loss(model, val_loader, device)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f}"
        )

    log_path = output_dir / "training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False, encoding="utf-8")
    print(f"Saved training log to: {log_path}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"Saved LoRA weights to: {output_dir}")
