"""
sleep_homeostat.py
===================
Process S - the homeostatic sleep-pressure process from Borbély's
two-process model of sleep regulation:

    * While AWAKE, sleep pressure rises and saturates exponentially
      toward a maximum.
    * While ASLEEP, sleep pressure decays exponentially toward a
      (non-zero) asymptote - one sleep episode rarely erases pressure
      completely, especially if it is short or low quality.

We simulate this as a trajectory across a crew member's actual sleep
history (rather than a single fixed formula) so that back-to-back short
sleeps compound realistically, exactly like real sleep-pressure dynamics.

Output range: 0-100, where 0 = fully rested / no sleep pressure and
100 = maximal homeostatic sleep pressure.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, NamedTuple

# Time constants (hours) - in line with the values commonly used in
# biomathematical fatigue models (e.g. Daan, Beersma & Borbély / SAFTE):
# pressure builds slowly across a ~16-18h wake period and unwinds faster
# across a full sleep period.
TAU_RISE_HOURS = 18.2
TAU_DECAY_HOURS = 4.2

S_MAX = 100.0          # asymptotic ceiling of sleep pressure while awake
S_FLOOR = 3.0           # pressure never fully reaches zero even after great sleep
DEFAULT_START_PRESSURE = 35.0  # assumed pressure before we have any history


class SleepSession(NamedTuple):
    start: datetime
    end: datetime
    quality: float  # 1-5 scale (sleep_quality column)


def _hours_between(a: datetime, b: datetime) -> float:
    return max(0.0, (b - a).total_seconds() / 3600.0)


def _rise(s0: float, hours: float, tau: float = TAU_RISE_HOURS) -> float:
    """Exponential saturating build-up of pressure across `hours` of wake."""
    return S_MAX - (S_MAX - s0) * pow(2.718281828, -hours / tau)


def _decay(s0: float, hours: float, quality: float, tau: float = TAU_DECAY_HOURS) -> float:
    """
    Exponential decay of pressure across `hours` of sleep.

    Sleep quality (1-5, from the wearable/self-report `sleep_quality`
    column) modulates how effectively pressure is dissipated: poor quality
    sleep (fragmented, low HRV-consistency) unwinds pressure more slowly,
    so we stretch the effective time constant for low-quality sleep.
    """
    quality = max(1.0, min(5.0, quality))
    quality_factor = 0.6 + 0.2 * quality  # quality=1 -> 0.8x speed, quality=5 -> 1.6x speed
    effective_tau = tau / quality_factor
    asymptote = S_FLOOR + (5.0 - quality) * 3.0  # poor sleep leaves a higher floor
    return asymptote + (s0 - asymptote) * pow(2.718281828, -hours / effective_tau)


def simulate_process_s(
    sleep_sessions: Iterable[SleepSession],
    as_of: datetime,
    start_pressure: float = DEFAULT_START_PRESSURE,
) -> float:
    """
    Walk chronologically through `sleep_sessions` (sorted ascending by
    start time) and simulate Process S up to `as_of`.

    Any sessions starting after `as_of` are ignored. If `as_of` falls
    inside a sleep session, we simulate the decay only up to `as_of`
    (partial night's sleep so far).
    """
    pressure = start_pressure
    cursor = None  # end of the last processed sleep bout

    sessions = sorted(sleep_sessions, key=lambda s: s.start)

    for session in sessions:
        if session.start >= as_of:
            break

        # Wake period between previous sleep end and this sleep's start.
        if cursor is not None:
            wake_hours = _hours_between(cursor, session.start)
            pressure = _rise(pressure, wake_hours)

        # This sleep bout: decay pressure across it (clipped to as_of).
        sleep_end = min(session.end, as_of)
        sleep_hours = _hours_between(session.start, sleep_end)
        pressure = _decay(pressure, sleep_hours, session.quality)
        cursor = sleep_end

        if session.end >= as_of:
            return round(max(0.0, min(100.0, pressure)), 2)

    # Final wake stretch from the last processed point up to as_of.
    if cursor is not None:
        wake_hours = _hours_between(cursor, as_of)
        pressure = _rise(pressure, wake_hours)

    return round(max(0.0, min(100.0, pressure)), 2)


def hours_since_last_wake(
    sleep_sessions: Iterable[SleepSession], as_of: datetime
) -> float | None:
    """Hours elapsed since the crew member's most recent sleep_end at or
    before `as_of`. Returns None if there's no prior sleep on record."""
    prior_ends = [s.end for s in sleep_sessions if s.end <= as_of]
    if not prior_ends:
        return None
    return _hours_between(max(prior_ends), as_of)
