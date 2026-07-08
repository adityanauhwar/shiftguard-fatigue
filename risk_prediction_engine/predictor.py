"""
predictor.py
============
Loads a trained classifier (see `model.py` / `train_risk_model.py`) and
uses it to answer the forward-looking question the rest of this
project doesn't: "what fatigue-risk category is this crew member
likely to report/experience at a specific FUTURE point in time?"

This is intentionally complementary to, not a replacement for:
  - `fatigue_engine`: a deterministic biomathematical score at a
    moment, with no notion of "risk category" or learned pattern.
  - `recommendation_engine`: unsupervised clustering of *current*
    fatigue patterns into archetypes, plus rule-triggered advice.
  - `scheduling_engine`: rule-based forward assignment; explicitly not
    a predictive model (see its README section).

`risk_prediction_engine` is the one genuinely predictive, supervised
piece: trained on thousands of historical self-reports to classify a
FUTURE moment (e.g. a not-yet-worked open shift's report time) into a
risk category, before anyone self-reports anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .feature_extraction import ALL_FEATURES, RISK_TIERS, build_feature_row

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "risk_model.joblib")


@dataclass
class RiskPrediction:
    crew_id: int
    as_of: str
    predicted_risk: str
    risk_probabilities: dict  # class -> probability
    top_risk_factors: list  # [(feature, value), ...] - highest-importance features for this row


def save_model(pipeline, path: str = DEFAULT_MODEL_PATH) -> None:
    joblib.dump(pipeline, path)


def load_model(path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python train_risk_model.py` first."
        )
    return joblib.load(path)


def _top_factors(features: dict, feature_importances: Optional[pd.Series], n: int = 3) -> list:
    if feature_importances is None:
        return []
    numeric_importances = feature_importances[feature_importances.index.isin(features.keys())]
    top = numeric_importances.sort_values(ascending=False).head(n)
    return [(name, features[name]) for name in top.index]


def predict_future_risk(
    data,
    crew_id: int,
    as_of,
    pipeline=None,
    category: str = "Scheduled Check-in",
    feature_importances: Optional[pd.Series] = None,
) -> RiskPrediction:
    """
    Predict the fatigue-risk category for `crew_id` at a future (or
    past) timestamp `as_of`.

    `as_of` can be any timestamp - including one beyond the latest
    record in the data (e.g. an upcoming open shift's `duty_start`).
    Everything the model conditions on (recent duty/sleep windows, the
    biomathematical engine outputs, the last self-report) is computed
    strictly from history before `as_of`, so this is a genuine forecast
    rather than a lookup.
    """
    pipeline = pipeline or load_model()

    feat_row = build_feature_row(data, crew_id=crew_id, as_of=as_of, category=category)
    X = pd.DataFrame([feat_row.features])[ALL_FEATURES]

    proba = pipeline.predict_proba(X)[0]
    classes = list(pipeline.named_steps["classifier"].classes_)
    proba_by_class = {cls: float(p) for cls, p in zip(classes, proba)}
    # Report in canonical Low->Severe order regardless of sklearn's
    # internal (alphabetical) class ordering.
    proba_by_class = {c: proba_by_class.get(c, 0.0) for c in RISK_TIERS}

    predicted = max(proba_by_class, key=proba_by_class.get)

    return RiskPrediction(
        crew_id=int(crew_id),
        as_of=pd.Timestamp(as_of).isoformat(),
        predicted_risk=predicted,
        risk_probabilities={k: round(v, 4) for k, v in proba_by_class.items()},
        top_risk_factors=_top_factors(feat_row.features, feature_importances),
    )


def predict_for_open_shifts(
    data,
    shifts_df: pd.DataFrame,
    crew_id: int,
    pipeline=None,
    feature_importances: Optional[pd.Series] = None,
) -> list[dict]:
    """
    Score every shift in `shifts_df` (as loaded by
    `scheduling_engine.shift_pool.load_open_shifts`, or any DataFrame
    with a `duty_start` column) as if `crew_id` were assigned to it -
    i.e. predict their fatigue-risk category AT that shift's report
    time, before it's ever worked. Useful as a pre-assignment check
    layered on top of (not instead of) `scheduling_engine`'s hard
    eligibility rules.
    """
    pipeline = pipeline or load_model()
    results = []
    for row in shifts_df.itertuples():
        pred = predict_future_risk(
            data,
            crew_id=crew_id,
            as_of=row.duty_start,
            pipeline=pipeline,
            category="Pre-Flight",
            feature_importances=feature_importances,
        )
        results.append(
            {
                "shift_id": getattr(row, "shift_id", None),
                "flight_no": getattr(row, "flight_no", None),
                "duty_start": pd.Timestamp(row.duty_start).isoformat(),
                "predicted_risk": pred.predicted_risk,
                "risk_probabilities": pred.risk_probabilities,
            }
        )
    return results
