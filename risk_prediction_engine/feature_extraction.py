"""
feature_extraction.py
======================
Builds leak-safe feature rows for the *supervised* future fatigue-risk
classifier, one row per historical self-report in `fatigue_reports.csv`
(the training set) or one row per hypothetical future moment (live
prediction).

This is deliberately a different feature philosophy from
`recommendation_engine/feature_extraction.py`. That module builds one
row *per crew member* summarizing their overall pattern, for
unsupervised clustering. This module builds one row *per point in
time*, using only information available strictly BEFORE that moment,
so a classifier trained on it is answering a genuinely forward-looking
question: "given everything known about this person up to now, what
fatigue level will they report next / experience on their next duty?"

Leak-safety
-----------
For a training row anchored at `report_date` for `crew_id`:
  - The biomathematical engine outputs (process_c/s/w, sleep_debt,
    alertness, base_fatigue_score) are re-run with `as_of=report_date`.
    `run_fatigue_model` internally discards any sleep/duty record at or
    after `as_of` (see fatigue_engine/sleep_homeostat.py), so these are
    safe.
  - Trailing workload/sleep windows below only include duty_logs / sleep
    rows whose end time is <= as_of.
  - The self-reported `samn_perelli_score` and `fatigue_level` of the
    CURRENT report are never used as features - only the PREVIOUS
    report (if any) is used as a lagged autoregressive signal, exactly
    as a real system would only know a person's past self-reports when
    forecasting their next one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fatigue_engine.engine import run_fatigue_model

# Trailing window for workload/sleep aggregates. 7 days matches the
# window already used elsewhere in the project (sleep_debt.py,
# recommendation_engine) for rolling fatigue-relevant signals.
TRAILING_WINDOW_DAYS = 7

# Same short-rest / night-ops thresholds as recommendation_engine, kept
# in sync so the two engines describe "short rest" and "night duty" the
# same way.
SHORT_REST_THRESHOLD_HOURS = 18.0
NIGHT_DUTY_START_HOUR = (0, 6)  # [start, end)

# Ordinal risk classes, low -> high, as they appear natively in
# fatigue_reports.csv. Kept around for documentation and as the basis
# for the tier mapping below, even though the model is no longer
# trained directly on these 5 classes (see RISK_TIERS).
RISK_CLASSES: list[str] = ["Low", "Mild", "Moderate", "High", "Severe"]

# The classifier is trained on 3 broader risk TIERS rather than the 5
# raw fatigue_level classes. Evaluation showed the 5-class model's
# errors were almost entirely between ADJACENT classes (Mild <-> Moderate
# <-> High blur together) while the extremes (Low vs. Severe) were
# rarely confused - i.e. the noise sits at the boundaries between
# neighboring classes, which merging those neighbors directly removes.
# This also matches the tier language recommendation_engine already
# uses for `overall_priority` (Low/Medium/High/Critical), so predictions
# from this engine read consistently with the rest of ShiftGuard.
RISK_TIERS: list[str] = ["Low-Risk", "Elevated", "High-Risk"]

FATIGUE_LEVEL_TO_TIER: dict[str, str] = {
    "Low": "Low-Risk",
    "Mild": "Low-Risk",
    "Moderate": "Elevated",
    "High": "High-Risk",
    "Severe": "High-Risk",
}


def collapse_to_tier(fatigue_level: str) -> str:
    """Map a raw 5-class `fatigue_level` label to its 3-tier bucket."""
    return FATIGUE_LEVEL_TO_TIER[fatigue_level]

NUMERIC_FEATURES: list[str] = [
    "process_c",
    "process_s",
    "process_w",
    "sleep_debt",
    "alertness",
    "base_fatigue_score",
    "n_duties_7d",  # recent duty count - workload volume
    "total_duty_hours_7d",  # recent duty hours - workload intensity
    "mean_sectors_7d",  # sectors/duty - task-switching load
    "short_rest_frequency_7d",  # fraction of recent duties with tight turnaround
    "night_duty_fraction_7d",  # fraction of recent duty starts in WOCL window
    "max_timezone_crossed_7d",  # worst recent jet-lag exposure
    "mean_sleep_hours_7d",  # recent sleep quantity
    "sleep_quality_mean_7d",  # recent self-rated sleep quality
    "hrv_deviation_7d",  # recent physiological strain vs. personal baseline
    "prev_samn_perelli",  # lagged self-report - their own trend
    "days_since_prev_report",  # recency of that lagged signal
    "has_prior_report",  # 0/1 flag - is prev_samn_perelli meaningful or a fallback?
    "sleep_need",
    "fatigue_sensitivity",
]

CATEGORICAL_FEATURES: list[str] = ["chronotype", "category"]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class RiskFeatureRow:
    crew_id: int
    as_of: pd.Timestamp
    features: dict


def _trailing(df: pd.DataFrame, end_col: str, as_of: pd.Timestamp, window_days: int = TRAILING_WINDOW_DAYS) -> pd.DataFrame:
    if df.empty:
        return df
    start = as_of - timedelta(days=window_days)
    return df.loc[(df[end_col] <= as_of) & (df[end_col] > start)]


def _duty_window_features(duty_df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    window = _trailing(duty_df, "duty_end", as_of)
    if window.empty:
        return {
            "n_duties_7d": 0.0,
            "total_duty_hours_7d": 0.0,
            "mean_sectors_7d": 0.0,
            "short_rest_frequency_7d": 0.0,
            "night_duty_fraction_7d": 0.0,
            "max_timezone_crossed_7d": 0.0,
        }
    duty_hours = (window.duty_end - window.duty_start).dt.total_seconds() / 3600.0
    hours = window.duty_start.dt.hour
    lo, hi = NIGHT_DUTY_START_HOUR
    return {
        "n_duties_7d": float(len(window)),
        "total_duty_hours_7d": float(duty_hours.sum()),
        "mean_sectors_7d": float(window.sectors.mean()),
        "short_rest_frequency_7d": float((window.rest_before < SHORT_REST_THRESHOLD_HOURS).mean()),
        "night_duty_fraction_7d": float(((hours >= lo) & (hours < hi)).mean()),
        "max_timezone_crossed_7d": float(window.timezone_crossed.max()),
    }


def _sleep_window_features(sleep_df: pd.DataFrame, as_of: pd.Timestamp, baseline_hrv: float) -> dict:
    window = _trailing(sleep_df, "sleep_end", as_of)
    if window.empty:
        return {
            "mean_sleep_hours_7d": 0.0,
            "sleep_quality_mean_7d": 3.0,
            "hrv_deviation_7d": 0.0,
        }
    hrv_dev = float(window.hrv.mean() - baseline_hrv) if not pd.isna(baseline_hrv) else 0.0
    return {
        "mean_sleep_hours_7d": float(window.sleep_hours.mean()),
        "sleep_quality_mean_7d": float(window.sleep_quality.mean()),
        "hrv_deviation_7d": hrv_dev,
    }


def _prior_report_features(reports_df: pd.DataFrame, as_of: pd.Timestamp, exclude_report_id: int | None = None) -> dict:
    prior = reports_df.loc[reports_df.report_date < as_of]
    if exclude_report_id is not None:
        prior = prior.loc[prior.report_id != exclude_report_id]
    if prior.empty:
        return {
            "prev_samn_perelli": 2.0,  # neutral fallback (~"Low" band midpoint)
            "days_since_prev_report": 14.0,  # neutral fallback: "a while ago"
            "has_prior_report": 0.0,
        }
    last = prior.sort_values("report_date").iloc[-1]
    days_since = max(0.0, (as_of - last.report_date).total_seconds() / 86400.0)
    return {
        "prev_samn_perelli": float(last.samn_perelli_score),
        "days_since_prev_report": days_since,
        "has_prior_report": 1.0,
    }


def build_feature_row(
    data,
    crew_id: int,
    as_of: pd.Timestamp,
    category: str = "Scheduled Check-in",
    exclude_report_id: int | None = None,
) -> RiskFeatureRow:
    """
    Build one leak-safe feature row for `crew_id` at time `as_of`.

    `category` is the report/check context ("Pre-Flight", "Post-Flight",
    "Layover", "Scheduled Check-in") - known ahead of time when you're
    the one scheduling the check, so it's a legitimate feature rather
    than a leak. `exclude_report_id` excludes a specific report from the
    "previous report" lookup - used during training so a report never
    sees itself as its own lag feature.
    """
    as_of = pd.Timestamp(as_of)
    profile = data.crew_profile(crew_id)
    duty_df = data.duty_for(crew_id)
    sleep_df = data.sleep_for(crew_id)
    reports_df = data.fatigue_reports_for(crew_id)

    engine_result = run_fatigue_model(data, crew_id=crew_id, as_of=as_of)

    features: dict = {
        "process_c": engine_result["process_c"],
        "process_s": engine_result["process_s"],
        "process_w": engine_result["process_w"],
        "sleep_debt": engine_result["sleep_debt"],
        "alertness": engine_result["alertness"],
        "base_fatigue_score": engine_result["base_fatigue_score"],
        **_duty_window_features(duty_df, as_of),
        **_sleep_window_features(sleep_df, as_of, float(profile.baseline_hrv)),
        **_prior_report_features(reports_df, as_of, exclude_report_id=exclude_report_id),
        "sleep_need": float(profile.sleep_need),
        "fatigue_sensitivity": float(profile.fatigue_sensitivity),
        "chronotype": str(profile.chronotype),
        "category": category,
    }
    return RiskFeatureRow(crew_id=int(crew_id), as_of=as_of, features=features)


def build_training_table(data) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build the full supervised training set: one row per historical
    fatigue report across every crew member.

    Returns
    -------
    X: DataFrame, ALL_FEATURES columns
    y: Series of 3-tier risk labels (RISK_TIERS: "Low-Risk" / "Elevated"
       / "High-Risk") - the collapsed target the classifier is actually
       trained on.
    meta: DataFrame with crew_id / report_date / report_id / the
          ORIGINAL 5-class fatigue_level, aligned with X and y, used
          for chronological train/test splitting and for tracing
          predictions back to the source report.
    """
    reports = data.fatigue_reports.sort_values("report_date").reset_index(drop=True)

    rows = []
    labels = []
    meta_rows = []
    for row in reports.itertuples():
        feat_row = build_feature_row(
            data,
            crew_id=row.crew_id,
            as_of=row.report_date,
            category=row.category,
            exclude_report_id=row.report_id,
        )
        rows.append(feat_row.features)
        labels.append(collapse_to_tier(row.fatigue_level))
        meta_rows.append(
            {
                "report_id": row.report_id,
                "crew_id": row.crew_id,
                "report_date": row.report_date,
                "fatigue_level": row.fatigue_level,
            }
        )

    X = pd.DataFrame(rows)[ALL_FEATURES]
    y = pd.Series(labels, name="risk_tier")
    meta = pd.DataFrame(meta_rows)
    return X, y, meta
