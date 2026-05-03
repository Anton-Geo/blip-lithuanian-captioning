from pathlib import Path
import pandas as pd

from src.image_utils import load_image_from_path
from src.model_loader import load_blip_model
from src.inference import generate_caption


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "annotations.csv"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "predictions_before.csv"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANNOTATIONS_PATH)

    df = df.dropna(subset=["caption_lt"])
    df = df[df["caption_lt"].astype(str).str.strip() != ""]

    test_df = df[df["split"] == "test"].copy()

    print(f"Annotated test samples: {len(test_df)}")

    processor, model, device = load_blip_model()

    rows = []

    for _, row in test_df.iterrows():
        filename = row["filename"]
        image_path = IMAGES_DIR / filename

        image = load_image_from_path(image_path)
        prediction = generate_caption(image, processor, model, device)

        rows.append({
            "filename": filename,
            "reference": row["caption_lt"],
            "prediction_before": prediction,
        })

        print(filename)
        print("REF:", row["caption_lt"])
        print("PRED:", prediction)
        print()

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
