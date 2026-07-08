"""
engine.py
=========
Runs the full Three-Process Fatigue Engine for one crew member (or every
crew member) at a given point in time, using data joined by merge_data.py.

Pipeline for a single crew member at timestamp `as_of`:

    1. Pull their sleep history and static profile (chronotype,
       sleep_need, fatigue_sensitivity, timezone).
    2. sleep_homeostat.simulate_process_s(...)  -> Process S
    3. sleep_debt.compute_sleep_debt(...)       -> Sleep Debt (hours)
    4. sleep_inertia.compute_process_w(...)     -> Process W  (uses debt)
    5. circadian.circadian_process(...)         -> Process C
    6. alertness.compute_alertness(...)         -> Alertness Score
    7. fatigue_score.compute_fatigue_score(...) -> Base Fatigue Score

Output matches the required schema:

    {
        "crew_id": 101,
        "process_c": 82.4,
        "process_s": 61.3,
        "process_w": 9.7,
        "sleep_debt": 2.4,
        "alertness": 74.5,
        "base_fatigue_score": 68.2
    }
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from .alertness import compute_alertness
from .circadian import circadian_process
from .fatigue_score import compute_fatigue_score
from .sleep_debt import compute_sleep_debt
from .sleep_homeostat import SleepSession, simulate_process_s
from .sleep_inertia import compute_process_w

try:  # merge_data.py lives one directory up (project root)
    from merge_data import MergedData
except ImportError:  # pragma: no cover - allows the package to be imported
    MergedData = None  # type: ignore


def _sleep_sessions_from_df(sleep_df: pd.DataFrame) -> list[SleepSession]:
    return [
        SleepSession(start=row.sleep_start, end=row.sleep_end, quality=row.sleep_quality)
        for row in sleep_df.itertuples()
    ]


def _timezone_adaptation_shift(duty_df: pd.DataFrame, as_of: datetime) -> float:
    """
    Rough estimate of residual circadian misalignment (hours) from the
    most recent duty prior to `as_of`, based on `timezone_crossed`.
    Assumes partial (50%) adaptation for time elapsed since that duty,
    at a conservative ~1 body-clock-hour of adaptation per day - the
    commonly cited rule of thumb for circadian realignment after
    transmeridian travel.
    """
    prior = duty_df.loc[duty_df.duty_end <= as_of]
    if prior.empty:
        return 0.0

    last_duty = prior.sort_values("duty_end").iloc[-1]
    crossed = float(last_duty.timezone_crossed or 0)
    if crossed == 0:
        return 0.0

    days_since = max(0.0, (as_of - last_duty.duty_end).total_seconds() / 86400.0)
    adapted = min(crossed, days_since * 1.0)  # ~1h adaptation per day
    remaining_shift = crossed - adapted
    return remaining_shift


def run_fatigue_model(
    data,  # MergedData
    crew_id: int,
    as_of: Optional[datetime] = None,
) -> dict:
    """
    Run the Three-Process Fatigue Engine for a single crew member.

    Parameters
    ----------
    data: MergedData
        Output of `merge_data.load_merged_data(...)`.
    crew_id: int
        Crew member to evaluate.
    as_of: datetime, optional
        The moment to evaluate fatigue at. Defaults to the latest
        timestamp on record for this crew member (last sleep_end or
        duty_end, whichever is later) - i.e. "now" relative to their data.

    Returns
    -------
    dict matching the required output schema.
    """
    profile = data.crew_profile(crew_id)
    sleep_df = data.sleep_for(crew_id)
    duty_df = data.duty_for(crew_id)

    if as_of is None:
        candidates = []
        if not sleep_df.empty:
            candidates.append(sleep_df.sleep_end.max())
        if not duty_df.empty:
            candidates.append(duty_df.duty_end.max())
        as_of = max(candidates) if candidates else pd.Timestamp.utcnow()

    sessions = _sleep_sessions_from_df(sleep_df)

    sleep_need = float(profile.sleep_need)
    fatigue_sensitivity = float(profile.fatigue_sensitivity)
    chronotype = str(profile.chronotype)

    # --- Process S (sleep pressure) ------------------------------------
    process_s = simulate_process_s(sessions, as_of)

    # --- Sleep Debt ------------------------------------------------------
    sleep_debt = compute_sleep_debt(sessions, as_of, sleep_need)

    # --- Process W (sleep inertia) --------------------------------------
    process_w = compute_process_w(sessions, as_of, sleep_debt)

    # --- Process C (circadian) ------------------------------------------
    tz_shift = _timezone_adaptation_shift(duty_df, as_of)
    process_c = round(
        circadian_process(
            as_of.to_pydatetime() if hasattr(as_of, "to_pydatetime") else as_of,
            chronotype=chronotype,
            timezone_adaptation_shift_hours=tz_shift,
        ),
        2,
    )

    # --- Alertness & Base Fatigue Score -----------------------------------
    alertness = compute_alertness(process_c, process_s, process_w, sleep_debt, fatigue_sensitivity)
    base_fatigue_score = compute_fatigue_score(
        process_c, process_s, process_w, sleep_debt, fatigue_sensitivity
    )

    return {
        "crew_id": int(crew_id),
        "as_of": pd.Timestamp(as_of).isoformat(),
        "process_c": process_c,
        "process_s": process_s,
        "process_w": process_w,
        "sleep_debt": sleep_debt,
        "alertness": alertness,
        "base_fatigue_score": base_fatigue_score,
    }


def run_for_all_crew(data, as_of: Optional[datetime] = None) -> list[dict]:
    """Run the fatigue engine for every crew member in `data`."""
    return [run_fatigue_model(data, crew_id, as_of) for crew_id in data.all_crew_ids()]
