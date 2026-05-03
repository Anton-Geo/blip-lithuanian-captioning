import argparse
from src.evaluate import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate captioning results.")

    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to predictions CSV",
    )

    parser.add_argument(
        "--prediction-column",
        type=str,
        required=True,
        help="Column name with model predictions",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    compute_metrics(
        csv_path=args.csv_path,
        prediction_column=args.prediction_column,
    )


if __name__ == "__main__":
    main()
