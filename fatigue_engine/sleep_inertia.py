"""
sleep_inertia.py
================
Process W - sleep inertia: the transient grogginess / impaired alertness
immediately after waking, on top of (and separate from) the circadian and
homeostatic processes. Physiologically it fades fast - the bulk of
inertia dissipates within about 15-30 minutes, with a longer low-level
tail out to roughly 2-4 hours, especially after waking from deep sleep or
from a short/fragmented sleep (higher "sleep debt" at wake -> deeper,
longer inertia - this mirrors how forced awakenings during slow-wave sleep
produce worse inertia than natural wake-ups).

Output range: 0-100 where 0 = no inertia (fully clear-headed) and higher
values = stronger grogginess. In practice this rarely exceeds ~40 even
right at the moment of waking, because it is a short-lived add-on effect,
not a dominant driver of fatigue on its own.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable

from .sleep_homeostat import SleepSession, hours_since_last_wake

TAU_INERTIA_HOURS = 0.75  # most of the effect decays within ~45 minutes
INERTIA_TAIL_HOURS = 4.0  # beyond this, inertia is considered fully resolved

BASE_INERTIA = 22.0  # peak inertia right at wake after an "average" sleep


def sleep_inertia(
    hours_since_wake: float | None,
    prior_sleep_hours: float | None,
    sleep_debt_hours: float = 0.0,
) -> float:
    """
    Compute Process W at a point `hours_since_wake` hours after the crew
    member's last recorded wake-up.

    Parameters
    ----------
    hours_since_wake:
        Hours elapsed since the last sleep_end on record. If None (no
        prior sleep on record) inertia is treated as negligible (0) since
        we cannot characterize the preceding sleep bout.
    prior_sleep_hours:
        Duration (hours) of the sleep bout that was just woken from.
        Short sleeps produce sharper inertia at wake.
    sleep_debt_hours:
        Accumulated sleep debt at the time of waking; higher debt deepens
        and slightly prolongs inertia.

    Returns
    -------
    float in [0, 100].
    """
    if hours_since_wake is None or hours_since_wake > INERTIA_TAIL_HOURS:
        return 0.0
    if hours_since_wake < 0:
        return 0.0

    peak = BASE_INERTIA

    # Short prior sleep -> sharper inertia at wake.
    if prior_sleep_hours is not None:
        if prior_sleep_hours < 5.0:
            peak += (5.0 - prior_sleep_hours) * 4.0
        elif prior_sleep_hours > 8.5:
            peak -= 3.0  # long, likely recovery sleep -> slightly gentler wake

    # Accumulated debt deepens inertia (capped contribution).
    peak += min(15.0, sleep_debt_hours * 1.5)

    value = peak * math.exp(-hours_since_wake / TAU_INERTIA_HOURS)
    return round(max(0.0, min(100.0, value)), 2)


def compute_process_w(
    sleep_sessions: Iterable[SleepSession],
    as_of: datetime,
    sleep_debt_hours: float = 0.0,
) -> float:
    """Convenience wrapper: derive hours-since-wake and the duration of
    the most recent sleep bout directly from a crew member's sleep
    history, then compute Process W at `as_of`."""
    sessions = sorted(sleep_sessions, key=lambda s: s.end)
    prior = [s for s in sessions if s.end <= as_of]

    if not prior:
        return 0.0

    last_session = prior[-1]
    h_since_wake = hours_since_last_wake(sessions, as_of)
    prior_hours = (last_session.end - last_session.start).total_seconds() / 3600.0

    return sleep_inertia(h_since_wake, prior_hours, sleep_debt_hours)
