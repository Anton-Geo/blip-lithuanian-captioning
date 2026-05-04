import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate text semantic similarity.")

    parser.add_argument("--baseline-csv", type=str, required=True)
    parser.add_argument("--finetuned-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)

    parser.add_argument("--filename-column", type=str, default="filename")
    parser.add_argument("--reference-column", type=str, default="reference")
    parser.add_argument("--baseline-column", type=str, default="prediction_before")
    parser.add_argument("--finetuned-column", type=str, default="prediction_after")

    return parser.parse_args()


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


def summarize(scores: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


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

    df[args.reference_column] = df[args.reference_column].astype(str)
    df[args.baseline_column] = df[args.baseline_column].astype(str)
    df[args.finetuned_column] = df[args.finetuned_column].astype(str)

    print(f"Samples: {len(df)}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    references = df[args.reference_column].tolist()
    baseline_predictions = df[args.baseline_column].tolist()
    finetuned_predictions = df[args.finetuned_column].tolist()

    ref_emb = model.encode(
        references,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True,
    )

    baseline_emb = model.encode(
        baseline_predictions,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True,
    )

    finetuned_emb = model.encode(
        finetuned_predictions,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True,
    )

    df["semantic_ref_baseline"] = cosine_scores(ref_emb, baseline_emb)
    df["semantic_ref_finetuned"] = cosine_scores(ref_emb, finetuned_emb)

    summary = {
        "samples": int(len(df)),
        "model": MODEL_NAME,
        "semantic_ref_baseline": summarize(df["semantic_ref_baseline"].to_numpy()),
        "semantic_ref_finetuned": summarize(df["semantic_ref_finetuned"].to_numpy()),
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
