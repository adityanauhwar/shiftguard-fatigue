"""
feature_extraction.py
======================
Builds one fatigue-pattern feature vector per crew member, combining:

  (a) the biomathematical fatigue engine's outputs (Process C/S/W,
      sleep debt, alertness, base fatigue score) re-run at each crew
      member's most recent duties, so the features reflect *operational*
      fatigue exposure rather than a single snapshot in time, and
  (b) raw workload / physiological / subjective signals straight from
      duty_logs.csv, sleep.csv and fatigue_reports.csv that the engine
      itself doesn't score (workload, rest buffers, HRV drift,
      self-reported Samn-Perelli fatigue).

This is deliberately a separate, additive layer on top of
`fatigue_engine` rather than a modification of it: the engine's job is
to score fatigue *at a moment*; this module's job is to characterize a
crew member's *pattern* over recent history for clustering.

Each feature is documented inline with why it's included and what a
high value means operationally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fatigue_engine.engine import run_fatigue_model

# How many of each crew member's most recent duties to sample when
# re-running the fatigue engine. Sampling (rather than scoring every
# duty) keeps this cheap while still capturing a representative recent
# window; recent duties matter most for "what should change next".
MAX_DUTY_SAMPLES = 12

# A duty is flagged "short rest" if the rest period beforehand falls
# below this many hours. Chosen below the sample data's 25th percentile
# (~19h) so the flag picks out the genuinely tight-turnaround tail
# rather than a large fraction of normal operations.
SHORT_REST_THRESHOLD_HOURS = 18.0

# A duty starting inside this local-clock hour window is treated as
# "night ops" - i.e. reporting for duty during (or just before) the
# window of circadian low (WOCL), rather than merely operating through
# it later in the duty.
NIGHT_DUTY_START_HOUR = (0, 6)  # [start, end)

FEATURE_NAMES: list[str] = [
    "mean_alertness",  # lower = chronically less alert across recent duties
    "mean_base_fatigue_score",  # higher = higher operational fatigue risk
    "mean_sleep_debt_hours",  # higher = chronically under-recovered
    "mean_process_w",  # higher = more residual sleep inertia at duty time
    "circadian_variability",  # higher = duty timing swings across the body clock
    "short_rest_frequency",  # fraction of duties with a tight turnaround
    "mean_sectors_per_duty",  # workload density per duty
    "mean_sleep_hours",  # lower = short-sleeping as a pattern
    "sleep_quality_mean",  # lower = poor self-rated sleep quality
    "sleep_quality_variability",  # higher = erratic/unpredictable sleep quality
    "hrv_deviation",  # negative = physiological stress vs. personal baseline
    "mean_samn_perelli",  # higher = more subjective fatigue at self-report time
    "night_duty_fraction",  # higher = frequent WOCL-window duty starts
]


@dataclass
class CrewFeatureRow:
    crew_id: int
    features: dict[str, float]
    sample_count: int  # how many duties the engine was re-run against


def _severity_flags(duty_df: pd.DataFrame) -> tuple[float, float]:
    if duty_df.empty:
        return 0.0, 0.0
    short_rest_frac = float((duty_df.rest_before < SHORT_REST_THRESHOLD_HOURS).mean())
    mean_sectors = float(duty_df.sectors.mean())
    return short_rest_frac, mean_sectors


def _night_duty_fraction(duty_df: pd.DataFrame) -> float:
    if duty_df.empty:
        return 0.0
    hours = duty_df.duty_start.dt.hour
    lo, hi = NIGHT_DUTY_START_HOUR
    return float(((hours >= lo) & (hours < hi)).mean())


def _hrv_deviation(sleep_df: pd.DataFrame, baseline_hrv: float, n_recent: int = 14) -> float:
    if sleep_df.empty or pd.isna(baseline_hrv):
        return 0.0
    recent = sleep_df.sort_values("sleep_end").tail(n_recent)
    return float(recent.hrv.mean() - baseline_hrv)


def extract_features_for_crew(data, crew_id: int, max_samples: int = MAX_DUTY_SAMPLES) -> CrewFeatureRow:
    """
    Build a `CrewFeatureRow` for one crew member from `MergedData`.

    Re-runs the fatigue engine at each of the crew member's most recent
    `max_samples` duty-start timestamps, then aggregates engine outputs
    (mean/variability) alongside raw workload, sleep and self-report
    signals pulled directly from the merged tables.
    """
    profile = data.crew_profile(crew_id)
    duty_df = data.duty_for(crew_id)
    sleep_df = data.sleep_for(crew_id)
    reports_df = data.fatigue_reports_for(crew_id)

    recent_duties = duty_df.tail(max_samples)

    alertness_vals: list[float] = []
    fatigue_vals: list[float] = []
    debt_vals: list[float] = []
    inertia_vals: list[float] = []
    circadian_vals: list[float] = []

    for row in recent_duties.itertuples():
        result = run_fatigue_model(data, crew_id=crew_id, as_of=row.duty_start)
        alertness_vals.append(result["alertness"])
        fatigue_vals.append(result["base_fatigue_score"])
        debt_vals.append(result["sleep_debt"])
        inertia_vals.append(result["process_w"])
        circadian_vals.append(result["process_c"])

    def _mean(vals: list[float], default: float = 0.0) -> float:
        return float(np.mean(vals)) if vals else default

    def _std(vals: list[float], default: float = 0.0) -> float:
        return float(np.std(vals)) if len(vals) > 1 else default

    short_rest_freq, mean_sectors = _severity_flags(duty_df)

    features = {
        "mean_alertness": _mean(alertness_vals, default=70.0),
        "mean_base_fatigue_score": _mean(fatigue_vals, default=30.0),
        "mean_sleep_debt_hours": _mean(debt_vals),
        "mean_process_w": _mean(inertia_vals),
        "circadian_variability": _std(circadian_vals),
        "short_rest_frequency": short_rest_freq,
        "mean_sectors_per_duty": mean_sectors,
        "mean_sleep_hours": float(sleep_df.sleep_hours.mean()) if not sleep_df.empty else float(profile.sleep_need),
        "sleep_quality_mean": float(sleep_df.sleep_quality.mean()) if not sleep_df.empty else 3.0,
        "sleep_quality_variability": float(sleep_df.sleep_quality.std()) if len(sleep_df) > 1 else 0.0,
        "hrv_deviation": _hrv_deviation(sleep_df, float(profile.baseline_hrv)),
        "mean_samn_perelli": float(reports_df.samn_perelli_score.mean()) if not reports_df.empty else 2.0,
        "night_duty_fraction": _night_duty_fraction(duty_df),
    }

    return CrewFeatureRow(crew_id=int(crew_id), features=features, sample_count=len(recent_duties))


def build_feature_table(data, crew_ids: list[int] | None = None) -> pd.DataFrame:
    """
    Build the full crew x feature DataFrame used as clustering input.

    Returns a DataFrame indexed by `crew_id` with one column per entry in
    `FEATURE_NAMES`.
    """
    ids = crew_ids if crew_ids is not None else data.all_crew_ids()
    rows = [extract_features_for_crew(data, cid) for cid in ids]
    df = pd.DataFrame(
        {row.crew_id: row.features for row in rows}
    ).T
    df.index.name = "crew_id"
    return df[FEATURE_NAMES]
