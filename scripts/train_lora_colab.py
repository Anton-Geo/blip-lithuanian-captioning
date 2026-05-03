import argparse
import torch

from src.config import DEFAULT_PROMPT
from src.train_lora import train_lora


def parse_args():
    parser = argparse.ArgumentParser(description="Train mBLIP with LoRA.")

    parser.add_argument("--annotations-path", type=str, default="/content/dataset/annotations.csv")
    parser.add_argument("--images-dir", type=str, default="/content/dataset/images")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/dl_task2/models/mblip-lora-lithuanian",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)

    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting mBLIP LoRA training...")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_lora(
        annotations_path=args.annotations_path,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
