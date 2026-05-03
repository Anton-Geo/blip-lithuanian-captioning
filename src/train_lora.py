from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from src.config import DEFAULT_PROMPT, MODEL_NAME
from src.dataset import MBlipCaptionDataset


def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            device = next(model.parameters()).device
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    model.train()
    return total_loss / max(len(dataloader), 1)


def train_lora(
    annotations_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    model_name: str = MODEL_NAME,
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    prompt: str = DEFAULT_PROMPT,
    patience: int = 5,
    min_delta: float = 1e-3,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processor...")
    processor = Blip2Processor.from_pretrained(model_name)

    print("Loading model...")

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    print(f"Model dtype: {model_dtype}")

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q", "v"],
    )

    model = get_peft_model(model, lora_config)
    model.train()
    model.print_trainable_parameters()

    train_dataset = MBlipCaptionDataset(
        annotations_path=annotations_path,
        images_dir=images_dir,
        processor=processor,
        split="train",
        prompt=prompt,
    )

    val_dataset = MBlipCaptionDataset(
        annotations_path=annotations_path,
        images_dir=images_dir,
        processor=processor,
        split="val",
        prompt=prompt,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Prompt: {prompt}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        threshold=1e-3,
        min_lr=5e-6,
    )

    history = []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            device = next(model.parameters()).device
            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Non-finite loss detected: {loss.item()}")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0,
            )

            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

            # if step % 20 == 0:
            #     print(
            #         f"Epoch {epoch + 1}, "
            #         f"step {step}/{len(train_loader)}, "
            #         f"loss: {loss.item():.4f}"
            #     )

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_val_loss = evaluate_loss(model, val_loader)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = avg_val_loss < best_val_loss - min_delta

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": current_lr,
                "improved": improved,
            }
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.8f}"
        )

        if improved:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

            print("Saved best model")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    log_path = output_dir / "training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False, encoding="utf-8")
    print(f"Saved training log to: {log_path}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"Saved LoRA adapter to: {output_dir}")
