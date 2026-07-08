"""
train_risk_model.py
====================
CLI to train the Future Risk Prediction classifier (Random Forest) on
every historical fatigue report, evaluate it on a chronological holdout
(the last N% of reports by date - i.e. genuinely "the future" relative
to training), and save the fitted model to disk for
`run_risk_prediction.py` to load.

Example
-------
    python train_risk_model.py
    python train_risk_model.py --test-fraction 0.25 --out risk_prediction_engine/risk_model.joblib
"""

from __future__ import annotations

import argparse

from merge_data import load_merged_data
from risk_prediction_engine.model import train_and_evaluate
from risk_prediction_engine.predictor import DEFAULT_MODEL_PATH, save_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ShiftGuard Future Risk Prediction classifier (Random Forest)"
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing the 4 CSVs")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of reports (chronologically last) held out for evaluation",
    )
    parser.add_argument("--out", default=DEFAULT_MODEL_PATH, help="Where to save the trained model")
    args = parser.parse_args()

    print("Loading and joining data...")
    data = load_merged_data(args.data_dir)

    print(f"Building leak-safe training table and fitting Random Forest "
          f"(chronological {int((1 - args.test_fraction) * 100)}/{int(args.test_fraction * 100)} split)...")
    result = train_and_evaluate(data, test_fraction=args.test_fraction)

    print()
    print(f"Train set : {result.n_train} reports  ({result.train_date_range[0].date()} -> {result.train_date_range[1].date()})")
    print(f"Test set  : {result.n_test} reports  ({result.test_date_range[0].date()} -> {result.test_date_range[1].date()})")
    print()
    print(f"Accuracy               : {result.accuracy:.3f}")
    print(f"Macro F1                : {result.macro_f1:.3f}")
    print(f"Mean class-rank error   : {result.mean_absolute_class_error:.3f}  "
          f"(0 = perfect, avg # of risk bands off by)")
    print()
    print("Classification report (test set, chronological holdout):")
    print(result.classification_report_text)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(result.confusion_matrix.to_string())
    print()
    print("Top 10 feature importances:")
    print(result.feature_importances.head(10).to_string())

    save_model(result.pipeline, args.out)
    print(f"\nSaved trained model to {args.out}")


if __name__ == "__main__":
    main()
