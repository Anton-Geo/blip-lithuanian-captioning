import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def lithuanian_ratio(text: str) -> float:
    lt_chars = set("ąčęėįšųūž")
    words = text.lower().split()

    if not words:
        return 0.0

    count = sum(any(c in lt_chars for c in word) for word in words)
    return count / len(words)


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
    lt_ratios = []

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
        lt_ratios.append(lithuanian_ratio(prediction_text))

    samples = len(bleu_scores)

    if samples == 0:
        print("No valid predictions found.")
        return {
            "samples": 0,
            "average_bleu": 0.0,
            "average_lt_ratio": 0.0,
        }

    average_bleu = sum(bleu_scores) / samples
    average_lt_ratio = sum(lt_ratios) / samples

    print(f"Samples: {samples}")
    print(f"Average BLEU: {average_bleu:.4f}")
    print(f"Average LT ratio: {average_lt_ratio:.4f}")

    return {
        "samples": samples,
        "average_bleu": average_bleu,
        "average_lt_ratio": average_lt_ratio,
    }
