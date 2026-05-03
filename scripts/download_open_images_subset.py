from pathlib import Path
import random

import fiftyone.zoo as foz
import pandas as pd
from tqdm import tqdm
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "annotations.csv"

TARGET_TOTAL_IMAGES = 1200
RANDOM_SEED = 1707

OPEN_IMAGES_CLASSES = [
    "Person",
    "Dog",
    "Cat",
    "Car",
    "Bicycle",
    "Bus",
    "Train",
    "Boat",
    "Bird",
    "Horse",
    "Pizza",
    "Sandwich",
    "Coffee cup",
    "Chair",
    "Table",
    "Laptop",
    "Mobile phone",
    "Book",
    "Building",
    "Tree",
]


def make_split(index: int, total: int) -> str:
    """
    70/15/15 split: train/val/test.
    """
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def ensure_rgb_jpeg(src_path: Path, dst_path: Path) -> bool:
    """
    Converts image to RGB JPEG and saves it with normalized filename.
    Returns False if image cannot be opened.
    """
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img.save(dst_path, format="JPEG", quality=95)
        return True
    except Exception as error:
        print(f"Skipping invalid image {src_path}: {error}")
        return False


def main():
    random.seed(RANDOM_SEED)

    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Open Images V7 subset...")

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=OPEN_IMAGES_CLASSES,
        max_samples=TARGET_TOTAL_IMAGES,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    samples = list(dataset)
    random.shuffle(samples)

    rows = []
    used_source_paths = set()

    print("Copying and normalizing images...")

    for sample in tqdm(samples):
        if len(rows) >= TARGET_TOTAL_IMAGES:
            break

        src_path = Path(sample.filepath)

        if src_path in used_source_paths:
            continue

        used_source_paths.add(src_path)

        filename = f"img_{len(rows) + 1:06d}.jpg"
        dst_path = OUTPUT_IMAGES_DIR / filename

        ok = ensure_rgb_jpeg(src_path, dst_path)
        if not ok:
            continue

        rows.append(
            {
                "filename": filename,
                "caption_lt": "",
                "split": make_split(len(rows), TARGET_TOTAL_IMAGES),
            }
        )

    if len(rows) < TARGET_TOTAL_IMAGES:
        print(f"Warning: only prepared {len(rows)} images.")

    annotations = pd.DataFrame(rows)
    annotations.to_csv(ANNOTATIONS_PATH, index=False, encoding="utf-8")

    print(f"Saved images to: {OUTPUT_IMAGES_DIR}")
    print(f"Saved annotations to: {ANNOTATIONS_PATH}")
    print(f"Total images: {len(rows)}")


if __name__ == "__main__":
    main()
