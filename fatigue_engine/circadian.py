"""
circadian.py
============
Process C - the circadian ("body clock") contribution to alertness.

Modeled as a two-harmonic cosine oscillation across the 24h day, which is
the standard way the circadian component is represented in biomathematical
fatigue models (e.g. the SAFTE model and Åkerstedt's three-process model).
A single cosine captures the main day/night swing; a second, weaker
harmonic captures the well-documented post-lunch dip (the "circadian
trough" felt around 2-4pm even though it's not the main sleepiness peak).

The acrophase (time of peak alertness) and bathyphase (time of peak
sleepiness, typically ~04:00-06:00 body clock time) are shifted according
to the crew member's chronotype and, in a rostering context, however many
hours their body clock has adapted across the timezones they've crossed.

Output range: 0-100, where 100 = circadian peak alertness, 0 = circadian
trough (deepest body-clock sleepiness, ~04:00-06:00 for a neutral
chronotype).
"""

from __future__ import annotations

import math
from datetime import datetime

# Base circadian trough (hours, 24h clock) for a "Neutral" chronotype.
# This is the well-established window of maximum sleepiness / minimum
# core body temperature (WOCL - Window of Circadian Low).
BASE_TROUGH_HOUR = 4.5  # 04:30

# How many hours the trough shifts for morning ("larks") / evening ("owls")
# chronotypes.
CHRONOTYPE_SHIFT_HOURS = {
    "Morning": -1.0,   # larks trough earlier, e.g. ~03:30
    "Neutral": 0.0,
    "Evening": +1.5,   # owls trough later, e.g. ~06:00
}

# Amplitude of the primary 24h circadian rhythm and the secondary
# ~12h post-lunch-dip harmonic, in alertness points.
PRIMARY_AMPLITUDE = 42.0
SECONDARY_AMPLITUDE = 8.0
SECONDARY_PEAK_OFFSET_HOURS = 14.0  # post-lunch dip centered ~14:00-15:00

MIDLINE = 50.0  # process oscillates around this baseline (0-100 scale)


def _decimal_hour(ts: datetime) -> float:
    """Convert a timestamp to a fractional hour-of-day in [0, 24)."""
    return ts.hour + ts.minute / 60.0 + ts.second / 3600.0


def circadian_trough_hour(chronotype: str) -> float:
    """The body-clock hour (0-24) of deepest circadian sleepiness for a
    given chronotype."""
    shift = CHRONOTYPE_SHIFT_HOURS.get(chronotype, 0.0)
    return (BASE_TROUGH_HOUR + shift) % 24.0


def circadian_process(
    timestamp: datetime,
    chronotype: str = "Neutral",
    timezone_adaptation_shift_hours: float = 0.0,
) -> float:
    """
    Compute Process C: the circadian alertness contribution at `timestamp`.

    Parameters
    ----------
    timestamp:
        The moment we're evaluating alertness for (local/body-clock time).
    chronotype:
        One of "Morning", "Neutral", "Evening" (from crew.csv).
    timezone_adaptation_shift_hours:
        Net hours the body clock is still offset from local time due to
        recent timezone crossings (positive = body clock running behind
        local time, e.g. jet lag after eastward travel). Defaults to 0
        (fully adapted / no jet lag modeled).

    Returns
    -------
    float in [~8, ~92] representing the circadian contribution to
    alertness on a 0-100 scale (higher = more alert).
    """
    hour = _decimal_hour(timestamp) - timezone_adaptation_shift_hours
    hour = hour % 24.0

    trough_hour = circadian_trough_hour(chronotype)
    # Primary rhythm: cosine dips to its minimum exactly at trough_hour and
    # peaks exactly 12h later.
    primary = -PRIMARY_AMPLITUDE * math.cos(
        2 * math.pi * (hour - trough_hour) / 24.0
    )

    # Secondary post-lunch-dip harmonic (12h period).
    secondary = -SECONDARY_AMPLITUDE * math.cos(
        2 * math.pi * (hour - SECONDARY_PEAK_OFFSET_HOURS) / 12.0
    )

    value = MIDLINE + primary + secondary
    return max(0.0, min(100.0, value))
