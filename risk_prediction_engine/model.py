"""
model.py
========
The actual "ML task classification algorithm": a supervised classifier
that predicts a crew member's future fatigue-risk TIER (Low-Risk /
Elevated / High-Risk) from the leak-safe features in
`feature_extraction.py`.

The tiers are a deliberate collapse of the raw 5-class `fatigue_level`
(Low/Mild/Moderate/High/Severe) - see `RISK_TIERS` and
`FATIGUE_LEVEL_TO_TIER` in `feature_extraction.py` for the mapping and
the rationale (the 5-class model's errors were almost entirely between
adjacent classes, so merging neighbors removes most of that noise
while keeping a 3-way distinction that's still operationally useful).

Algorithm choice: Random Forest.
  - Handles a mix of numeric (engine outputs, rolling workload/sleep
    aggregates) and one-hot categorical (chronotype, report category)
    features without needing careful scaling per-feature.
  - `class_weight="balanced_subsample"` matters here: "Severe" is ~2%
    of reports (see README below), and an unweighted model would
    happily ignore it and still score well on accuracy alone.
  - Gives inspectable `feature_importances_`, which matters for a
    fatigue-risk tool that has to explain *why* it flagged someone -
    a black-box score alone isn't operationally actionable.
  - Robust to the modest (~6k row) dataset size without heavy tuning,
    unlike e.g. gradient boosting, which tends to need more careful
    regularization to avoid overfitting on data this size.

Evaluation is CHRONOLOGICAL, not a random shuffle-split: the model
trains on the earlier N% of reports (by report_date) across the whole
roster and is evaluated on the later slice. This directly matches the
"future risk prediction" framing - can the model generalize to reports
that happen *after* everything it trained on - rather than the easier
(and less honest) question of whether it memorized interpolated
patterns from a random split of the same time period.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .feature_extraction import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RISK_TIERS,
    build_training_table,
)

RANDOM_STATE = 42
N_ESTIMATORS = 400
MAX_DEPTH = 10

# Fraction of reports (chronologically last) held out as the test set.
TEST_FRACTION = 0.2


@dataclass
class TrainingResult:
    pipeline: Pipeline
    accuracy: float
    macro_f1: float
    mean_absolute_class_error: float  # avg |predicted_rank - true_rank| on RISK_TIERS
    classification_report_text: str
    confusion_matrix: pd.DataFrame
    feature_importances: pd.Series
    n_train: int
    n_test: int
    train_date_range: tuple
    test_date_range: tuple


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("classifier", classifier)])


def _chronological_split(
    X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, test_fraction: float
):
    order = meta.report_date.argsort().values
    X, y, meta = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True), meta.iloc[order].reset_index(drop=True)
    split_idx = int(len(X) * (1 - test_fraction))
    return (
        X.iloc[:split_idx], y.iloc[:split_idx], meta.iloc[:split_idx],
        X.iloc[split_idx:], y.iloc[split_idx:], meta.iloc[split_idx:],
    )


def feature_importances(pipeline: Pipeline) -> pd.Series:
    """Public helper so callers (e.g. the predictor CLI) can explain a
    prediction using the same importances computed during training,
    without needing access to the internal TrainingResult."""
    ohe: OneHotEncoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    names = NUMERIC_FEATURES + cat_names
    importances = pipeline.named_steps["classifier"].feature_importances_
    return pd.Series(importances, index=names).sort_values(ascending=False)


# Backwards-compatible private alias used within this module.
_feature_importances = feature_importances


def _mean_absolute_class_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    rank = {label: i for i, label in enumerate(RISK_TIERS)}
    true_ranks = y_true.map(rank).to_numpy()
    pred_ranks = np.array([rank[p] for p in y_pred])
    return float(np.mean(np.abs(true_ranks - pred_ranks)))


def train_and_evaluate(data, test_fraction: float = TEST_FRACTION) -> TrainingResult:
    """Build the training table, fit on the chronologically-earlier
    slice, and evaluate on the chronologically-later slice."""
    X, y, meta = build_training_table(data)
    X_train, y_train, meta_train, X_test, y_test, meta_test = _chronological_split(
        X, y, meta, test_fraction
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    return TrainingResult(
        pipeline=pipeline,
        accuracy=float((y_pred == y_test.to_numpy()).mean()),
        macro_f1=float(f1_score(y_test, y_pred, average="macro", labels=RISK_TIERS)),
        mean_absolute_class_error=_mean_absolute_class_error(y_test, y_pred),
        classification_report_text=classification_report(
            y_test, y_pred, labels=RISK_TIERS, zero_division=0
        ),
        confusion_matrix=pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=RISK_TIERS),
            index=[f"true_{c}" for c in RISK_TIERS],
            columns=[f"pred_{c}" for c in RISK_TIERS],
        ),
        feature_importances=_feature_importances(pipeline),
        n_train=len(X_train),
        n_test=len(X_test),
        train_date_range=(meta_train.report_date.min(), meta_train.report_date.max()),
        test_date_range=(meta_test.report_date.min(), meta_test.report_date.max()),
    )
