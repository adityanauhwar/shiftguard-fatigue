"""
run_risk_prediction.py
=======================
CLI for the Future Risk Prediction Engine: loads the trained classifier
(train it first with `train_risk_model.py`) and predicts a fatigue-risk
category for a crew member at a future point in time.

Examples
--------
    # Predict risk for crew 101 "now" (latest timestamp on record)
    python run_risk_prediction.py --crew-id 101

    # Predict risk for crew 101 at a specific future moment
    python run_risk_prediction.py --crew-id 101 --as-of "2026-07-10 05:30:00"

    # Predict risk for every crew member, "now"
    python run_risk_prediction.py --all --out risk_predictions.json

    # Score every upcoming open shift as if crew 101 took it - a
    # pre-assignment forward risk check on top of scheduling_engine
    python run_risk_prediction.py --crew-id 101 --shifts data/open_shifts.csv

    # Add a plain-English briefing via the Groq API (requires GROQ_API_KEY
    # and `pip install groq` - see risk_prediction_engine/explain.py)
    python run_risk_prediction.py --crew-id 101 --explain
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from merge_data import load_merged_data
from risk_prediction_engine.model import feature_importances
from risk_prediction_engine.predictor import (
    load_model,
    predict_for_open_shifts,
    predict_future_risk,
)


def _default_as_of(data, crew_id: int) -> pd.Timestamp:
    duty_df = data.duty_for(crew_id)
    sleep_df = data.sleep_for(crew_id)
    candidates = []
    if not duty_df.empty:
        candidates.append(duty_df.duty_end.max())
    if not sleep_df.empty:
        candidates.append(sleep_df.sleep_end.max())
    return max(candidates) if candidates else pd.Timestamp.utcnow()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShiftGuard Future Risk Prediction (Random Forest classifier)"
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing the 4 CSVs")
    parser.add_argument("--model", default=None, help="Path to trained model (default: risk_prediction_engine/risk_model.joblib)")
    parser.add_argument("--crew-id", type=int, help="Predict for a single crew member")
    parser.add_argument("--all", action="store_true", help="Predict for every crew member")
    parser.add_argument("--as-of", help="Future (or past) timestamp to predict at, e.g. '2026-07-10 05:30:00'. Defaults to latest data on record.")
    parser.add_argument("--category", default="Scheduled Check-in", help="Report context: Pre-Flight / Post-Flight / Layover / Scheduled Check-in")
    parser.add_argument("--shifts", help="Path to an open-shifts CSV; scores each shift for --crew-id as a pre-assignment check")
    parser.add_argument("--out", help="Optional path to write JSON output to")
    parser.add_argument("--explain", action="store_true", help="Add a plain-English briefing per prediction via the Groq API (requires GROQ_API_KEY)")
    parser.add_argument("--explain-model", default=None, help="Override the Groq model used for --explain (default: llama-3.3-70b-versatile)")
    args = parser.parse_args()

    data = load_merged_data(args.data_dir)
    pipeline = load_model(args.model) if args.model else load_model()
    importances = feature_importances(pipeline)

    if args.shifts:
        if args.crew_id is None:
            parser.error("--shifts requires --crew-id")
            return
        shifts_df = pd.read_csv(args.shifts, parse_dates=["duty_start", "duty_end"])
        result = predict_for_open_shifts(
            data, shifts_df, crew_id=args.crew_id, pipeline=pipeline, feature_importances=importances
        )
    elif args.crew_id is not None:
        as_of = pd.Timestamp(args.as_of) if args.as_of else _default_as_of(data, args.crew_id)
        prediction = predict_future_risk(
            data, crew_id=args.crew_id, as_of=as_of, pipeline=pipeline,
            category=args.category, feature_importances=importances,
        )
        result = vars(prediction)
    elif args.all:
        result = []
        for crew_id in data.all_crew_ids():
            as_of = pd.Timestamp(args.as_of) if args.as_of else _default_as_of(data, crew_id)
            prediction = predict_future_risk(
                data, crew_id=crew_id, as_of=as_of, pipeline=pipeline,
                category=args.category, feature_importances=importances,
            )
            result.append(vars(prediction))
    else:
        parser.error("Specify --crew-id <id> (optionally with --shifts) or --all")
        return

    if args.explain:
        from risk_prediction_engine.explain import DEFAULT_MODEL, explain_prediction

        explain_model = args.explain_model or DEFAULT_MODEL
        items = result if isinstance(result, list) else [result]
        try:
            for item in items:
                item["explanation"] = explain_prediction(item, model=explain_model)
        except (ImportError, RuntimeError) as e:
            print(f"[--explain skipped] {e}\n")

    output = json.dumps(result, indent=2, default=str)
    print(output)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
