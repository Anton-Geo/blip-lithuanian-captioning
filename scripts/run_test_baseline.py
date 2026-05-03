import argparse
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_PROMPT
from src.image_utils import load_image_from_path
from src.inference import generate_caption
from src.model_loader import load_mblip_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline mBLIP inference.")

    parser.add_argument("--annotations-path", type=str, default="data/annotations.csv")
    parser.add_argument("--images-dir", type=str, default="data/raw/images")
    parser.add_argument("--output-path", type=str, default="outputs/predictions_before.csv")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)

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
    print(f"Prompt: {args.prompt}")

    processor, model = load_mblip_model()

    rows = []

    for _, row in test_df.iterrows():
        filename = row["filename"]
        image_path = images_dir / filename

        image = load_image_from_path(image_path)

        prediction = generate_caption(
            image=image,
            processor=processor,
            model=model,
            prompt=args.prompt,
        )

        rows.append(
            {
                "filename": filename,
                "reference": row["caption_lt"],
                "prediction_before": prediction,
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
