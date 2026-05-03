import argparse
import torch

from src.train_lora import train_lora


def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP with LoRA on Lithuanian captions.")

    parser.add_argument("--annotations-path", type=str, default="/content/dataset/annotations.csv")
    parser.add_argument("--images-dir", type=str, default="/content/dataset/images")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/dl_task2/models/blip-lora-lithuanian",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)

    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting BLIP LoRA training...")
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
    )


if __name__ == "__main__":
    main()
