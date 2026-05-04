import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image-text CLIP similarity.")

    parser.add_argument("--baseline-csv", type=str, required=True)
    parser.add_argument("--finetuned-csv", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)

    parser.add_argument("--filename-column", type=str, default="filename")
    parser.add_argument("--reference-column", type=str, default="reference")
    parser.add_argument("--baseline-column", type=str, default="prediction_before")
    parser.add_argument("--finetuned-column", type=str, default="prediction_after")

    return parser.parse_args()


def summarize(scores: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def cosine_scores(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return torch.sum(a * b, dim=-1).detach().cpu().numpy()


@torch.no_grad()
def encode_images(model, processor, images, device, batch_size=16):
    embeddings = []

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]

        inputs = processor(
            images=batch,
            return_tensors="pt",
            padding=True,
        ).to(device)

        image_features = model.get_image_features(**inputs)
        embeddings.append(image_features.cpu())

    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def encode_texts(model, processor, texts, device, batch_size=32):
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        text_features = model.get_text_features(**inputs)
        embeddings.append(text_features.cpu())

    return torch.cat(embeddings, dim=0)


def main():
    args = parse_args()

    baseline_df = pd.read_csv(args.baseline_csv, encoding="utf-8")
    finetuned_df = pd.read_csv(args.finetuned_csv, encoding="utf-8")

    df = baseline_df[
        [args.filename_column, args.reference_column, args.baseline_column]
    ].merge(
        finetuned_df[[args.filename_column, args.finetuned_column]],
        on=args.filename_column,
        how="inner",
    )

    images_dir = Path(args.images_dir)
    images = [
        Image.open(images_dir / filename).convert("RGB")
        for filename in df[args.filename_column].astype(str).tolist()
    ]

    references = df[args.reference_column].astype(str).tolist()
    baseline_predictions = df[args.baseline_column].astype(str).tolist()
    finetuned_predictions = df[args.finetuned_column].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Samples: {len(df)}")
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    image_emb = encode_images(model, processor, images, device)

    ref_emb = encode_texts(model, processor, references, device)
    baseline_emb = encode_texts(model, processor, baseline_predictions, device)
    finetuned_emb = encode_texts(model, processor, finetuned_predictions, device)

    df["clip_image_reference"] = cosine_scores(image_emb, ref_emb)
    df["clip_image_baseline"] = cosine_scores(image_emb, baseline_emb)
    df["clip_image_finetuned"] = cosine_scores(image_emb, finetuned_emb)

    summary = {
        "samples": int(len(df)),
        "model": MODEL_NAME,
        "clip_image_reference": summarize(df["clip_image_reference"].to_numpy()),
        "clip_image_baseline": summarize(df["clip_image_baseline"].to_numpy()),
        "clip_image_finetuned": summarize(df["clip_image_finetuned"].to_numpy()),
    }

    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False, encoding="utf-8")

    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to: {output_csv}")
    print(f"Saved summary to: {summary_json}")


if __name__ == "__main__":
    main()
