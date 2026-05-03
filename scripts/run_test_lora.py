import argparse
from pathlib import Path

import pandas as pd

from src.image_utils import load_image_from_path
from src.inference import generate_caption
from src.model_loader import load_blip_lora_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LoRA fine-tuned BLIP")

    parser.add_argument("--annotations-path", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--adapter-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    annotations_path = Path(args.annotations_path)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(annotations_path, encoding="utf-8")

    df = df.dropna(subset=["caption_lt"])
    df = df[df["caption_lt"].astype(str).str.strip() != ""]

    test_df = df[df["split"] == "test"].copy()

    print(f"Annotated test samples: {len(test_df)}")
    print(f"Loading LoRA adapter from: {args.adapter_dir}")

    processor, model, device = load_blip_lora_model(
        adapter_dir=args.adapter_dir,
    )

    print(f"Device: {device}")

    rows = []

    for _, row in test_df.iterrows():
        filename = row["filename"]
        image_path = images_dir / filename

        image = load_image_from_path(image_path)

        prediction = generate_caption(
            image=image,
            processor=processor,
            model=model,
            device=device,
        )

        rows.append(
            {
                "filename": filename,
                "reference": row["caption_lt"],
                "prediction_after": prediction,
            }
        )

        print(filename)
        print("REF:", row["caption_lt"])
        print("PRED:", prediction)
        print()

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
