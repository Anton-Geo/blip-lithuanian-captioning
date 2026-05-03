import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def lithuanian_ratio(text: str) -> float:
    lt_chars = set("ąčęėįšųūž")
    words = text.lower().split()

    if not words:
        return 0.0

    count = sum(any(c in lt_chars for c in w) for w in words)
    return count / len(words)


def compute_metrics(csv_path: str):
    df = pd.read_csv(csv_path)

    smoothie = SmoothingFunction().method1

    bleu_scores = []
    lt_ratios = []

    for _, row in df.iterrows():
        ref = str(row["reference"]).lower().split()
        pred = str(row["prediction_before"]).lower().split()

        if len(pred) == 0:
            continue

        bleu = sentence_bleu([ref], pred, smoothing_function=smoothie)
        bleu_scores.append(bleu)

        lt_ratio = lithuanian_ratio(" ".join(pred))
        lt_ratios.append(lt_ratio)

    print(f"Samples: {len(bleu_scores)}")
    print(f"Average BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"Average LT ratio: {sum(lt_ratios)/len(lt_ratios):.4f}")
