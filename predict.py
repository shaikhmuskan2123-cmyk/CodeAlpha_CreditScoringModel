from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from src.config import MODEL_PATH
from src.prediction import batch_predict, load_model_artifact, predict_single


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict credit risk using the trained model.")
    parser.add_argument("--input", type=str, required=True, help="CSV file of records or a single JSON input file.")
    parser.add_argument("--batch", action="store_true", help="Explicitly score a CSV batch.")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Optional model artifact path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    input_type = input_path.suffix.lower()
    if args.batch and input_type != ".csv":
        raise SystemExit("--batch requires a .csv input file.")
    if input_type not in {".csv", ".json"}:
        raise SystemExit("Input file must use the .csv or .json extension.")

    model = load_model_artifact(args.model)

    if input_type == ".csv":
        df = pd.read_csv(input_path)
        result = batch_predict(df, model=model)
        sys.stdout.write(result.to_csv(index=False))
        return

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(predict_single(payload, model=model))


if __name__ == "__main__":
    main()
