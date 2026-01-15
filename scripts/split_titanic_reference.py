import argparse
from pathlib import Path

import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a source CSV into train/test by randomly sampling N rows for test and "
            "removing them from train."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset") / "titanic_reference.csv",
        help="Path to source CSV (default: dataset/titanic_reference.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of rows to put into test (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("dataset") / "titanic_train.csv",
        help="Output path for train CSV (default: dataset/titanic_train.csv)",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("dataset") / "titanic_test.csv",
        help="Output path for test CSV (default: dataset/titanic_test.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.test_size <= 0:
        raise SystemExit("--test-size must be > 0")

    df = pd.read_csv(args.input)

    if len(df) < args.test_size:
        raise SystemExit(
            f"Not enough rows in {args.input} (rows={len(df)}) for test-size={args.test_size}"
        )

    test_df = df.sample(n=args.test_size, random_state=args.seed)
    train_df = df.drop(index=test_df.index)

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(args.train_output, index=False)
    test_df.to_csv(args.test_output, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Train rows: {len(train_df)} -> {args.train_output}")
    print(f"Test rows:  {len(test_df)} -> {args.test_output}")


if __name__ == "__main__":
    main()
