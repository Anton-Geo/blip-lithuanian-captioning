import argparse
from src.evaluate import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--prediction-column", type=str, default="prediction_after")
    return parser.parse_args()


def main():
    args = parse_args()

    compute_metrics(
        csv_path=args.csv_path,
        prediction_column=args.prediction_column,
    )


if __name__ == "__main__":
    main()
