"""
alertness.py
============
Combines Process C, Process S, Process W and accumulated Sleep Debt into
a single 0-100 Alertness Score (100 = fully alert, 0 = maximally
impaired) - i.e. the "output" side of the three-process model, as
opposed to fatigue_score.py which produces the fatigue-facing view of the
same underlying processes.

Personalization: crew.csv provides a per-person `fatigue_sensitivity`
(roughly 0.8-1.2 in the sample data, 1.0 = average). We apply this as an
ADDITIVE bias rather than a multiplicative slope on the final score.
Earlier calibration work on this project found that scaling the whole
curve by an individual slope over-corrected and over-fit on sparse
per-person history; a small additive shift proportional to sensitivity is
more stable and keeps every score inside a sane, comparable 0-100 range
regardless of how much history a given crew member has.
"""

from __future__ import annotations

# Relative contribution of each process to overall alertness.
WEIGHT_CIRCADIAN = 0.45
WEIGHT_PRESSURE = 0.35   # applied to (100 - process_s)
WEIGHT_INERTIA = 0.20    # applied to process_w (subtracted)

# How strongly accumulated sleep debt drags alertness down (points per hour
# of debt), on top of the process-level effects above.
DEBT_PENALTY_PER_HOUR = 2.5
DEBT_PENALTY_CAP = 25.0

# Additive personalization bias: for each 0.1 of fatigue_sensitivity above
# 1.0, shift the final score down by this many points (and up for below).
SENSITIVITY_BIAS_PER_TENTH = 1.2


def compute_alertness(
    process_c: float,
    process_s: float,
    process_w: float,
    sleep_debt_hours: float,
    fatigue_sensitivity: float = 1.0,
) -> float:
    """
    Compute the 0-100 Alertness Score.

    Parameters
    ----------
    process_c: Circadian contribution (0-100, higher = more alert).
    process_s: Homeostatic sleep pressure (0-100, higher = more pressure).
    process_w: Sleep inertia (0-100, higher = more grogginess).
    sleep_debt_hours: Accumulated sleep debt, in hours.
    fatigue_sensitivity: Per-person sensitivity multiplier from crew.csv
        (1.0 = average).

    Returns
    -------
    float in [0, 100].
    """
    base = (
        WEIGHT_CIRCADIAN * process_c
        + WEIGHT_PRESSURE * (100.0 - process_s)
        - WEIGHT_INERTIA * process_w
    )

    debt_penalty = min(DEBT_PENALTY_CAP, sleep_debt_hours * DEBT_PENALTY_PER_HOUR)
    score = base - debt_penalty

    # Additive personalization bias (see module docstring for rationale).
    sensitivity_delta = fatigue_sensitivity - 1.0
    score += -sensitivity_delta * 10 * SENSITIVITY_BIAS_PER_TENTH

    return round(max(0.0, min(100.0, score)), 2)
