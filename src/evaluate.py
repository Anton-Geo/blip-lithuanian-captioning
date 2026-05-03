import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_metrics(
    csv_path: str,
    prediction_column: str,
    reference_column: str = "reference",
):
    df = pd.read_csv(csv_path, encoding="utf-8")

    if reference_column not in df.columns:
        raise ValueError(f"Missing reference column: {reference_column}")

    if prediction_column not in df.columns:
        raise ValueError(f"Missing prediction column: {prediction_column}")

    smoothie = SmoothingFunction().method1

    bleu_scores = []

    for _, row in df.iterrows():
        reference = str(row[reference_column]).lower().split()
        prediction_text = str(row[prediction_column])
        prediction = prediction_text.lower().split()

        if len(prediction) == 0:
            continue

        bleu = sentence_bleu(
            [reference],
            prediction,
            smoothing_function=smoothie,
        )

        bleu_scores.append(bleu)

    samples = len(bleu_scores)

    if samples == 0:
        print("No valid predictions found.")
        return {
            "samples": 0,
            "average_bleu": 0.0,
            "average_lt_ratio": 0.0,
        }

    average_bleu = sum(bleu_scores) / samples

    print(f"Samples: {samples}")
    print(f"Average BLEU: {average_bleu:.4f}")

    return {
        "samples": samples,
        "average_bleu": average_bleu,
    }
