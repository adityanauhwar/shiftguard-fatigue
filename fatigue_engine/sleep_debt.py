"""
sleep_debt.py
=============
Accumulated sleep debt: how far behind a crew member's own biological
sleep need (`sleep_need` from crew.csv) their recent actual sleep has
been, in hours.

We use an exponentially-weighted trailing window (most recent nights
count more than older ones - a night of catch-up sleep starts paying
down debt immediately, while an old deficit fades in relevance) rather
than a flat rolling sum, which better matches how cumulative fatigue
research (e.g. multi-day partial-sleep-restriction studies) treats debt:
recent restriction dominates, and debt only partially "resets" without a
recovery night.

Output: hours of accumulated debt (>= 0). 0 means fully caught up or
ahead of personal sleep need.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

from .sleep_homeostat import SleepSession

WINDOW_DAYS = 7
DECAY_PER_DAY = 0.85  # older nights count for progressively less
CREDIT_RETENTION = 0.5  # a surplus night only offsets 50% of prior debt


def compute_sleep_debt(
    sleep_sessions: Iterable[SleepSession],
    as_of: datetime,
    sleep_need_hours: float,
    window_days: int = WINDOW_DAYS,
) -> float:
    """
    Compute accumulated sleep debt (hours) as of `as_of`.

    Walks the crew member's sleep sessions inside the trailing
    `window_days` window (ending at `as_of`), oldest to newest, and
    accumulates an exponentially-weighted deficit: each night's shortfall
    (sleep_need - sleep_hours) adds to the running debt; a surplus night
    only pays down a fraction of existing debt (`CREDIT_RETENTION`)
    reflecting how recovery sleep helps but doesn't instantly erase
    accumulated pressure.
    """
    window_start = as_of - timedelta(days=window_days)
    sessions = sorted(
        (s for s in sleep_sessions if window_start <= s.end <= as_of),
        key=lambda s: s.end,
    )

    if not sessions:
        return 0.0

    debt = 0.0
    for session in sessions:
        nightly_hours = (session.end - session.start).total_seconds() / 3600.0
        age_days = (as_of - session.end).total_seconds() / 86400.0
        weight = DECAY_PER_DAY ** age_days

        shortfall = sleep_need_hours - nightly_hours
        if shortfall > 0:
            debt += shortfall * weight
        else:
            # Surplus sleep pays down existing debt, but not 1:1.
            debt = max(0.0, debt + shortfall * weight * CREDIT_RETENTION)

    return round(max(0.0, debt), 2)
