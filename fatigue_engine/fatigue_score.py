"""
fatigue_score.py
================
Combines Process C, Process S, Process W and Sleep Debt into a single
0-100 Base Fatigue Score.

This is deliberately NOT simply `100 - alertness`. Alertness (in
alertness.py) is weighted to reflect moment-to-moment performance
capacity, while the Base Fatigue Score is weighted to better reflect
operational/safety risk: it leans more heavily on sleep pressure, sleep
inertia and accumulated debt (the things a fatigue risk management
system needs to flag) and gives the circadian process comparatively less
say, since circadian dips are predictable/universal and less of a
distinguishing safety signal than a crew member's individual pressure and
debt build-up.

Output: 0-100, where 0 = no fatigue, 100 = maximal fatigue.
"""

from __future__ import annotations

WEIGHT_PRESSURE = 0.35        # process_s
WEIGHT_INERTIA = 0.20          # process_w
WEIGHT_DEBT = 0.25             # sleep debt, hours -> 0-100 scale
WEIGHT_CIRCADIAN = 0.20        # applied to (100 - process_c)

# Debt hours are converted to a 0-100 scale for blending; debt above this
# many hours is treated as maximally fatiguing.
DEBT_SATURATION_HOURS = 10.0

# Additive personalization bias, mirroring alertness.py: small, bounded,
# and additive rather than a multiplicative slope (see alertness.py for
# why the slope approach was rejected during calibration).
SENSITIVITY_BIAS_PER_TENTH = 1.2


def compute_fatigue_score(
    process_c: float,
    process_s: float,
    process_w: float,
    sleep_debt_hours: float,
    fatigue_sensitivity: float = 1.0,
) -> float:
    """
    Compute the 0-100 Base Fatigue Score.

    Parameters
    ----------
    process_c: Circadian contribution (0-100, higher = more alert).
    process_s: Homeostatic sleep pressure (0-100, higher = more pressure).
    process_w: Sleep inertia (0-100, higher = more grogginess).
    sleep_debt_hours: Accumulated sleep debt, in hours.
    fatigue_sensitivity: Per-person sensitivity multiplier from crew.csv.

    Returns
    -------
    float in [0, 100].
    """
    debt_scaled = min(100.0, (sleep_debt_hours / DEBT_SATURATION_HOURS) * 100.0)

    score = (
        WEIGHT_PRESSURE * process_s
        + WEIGHT_INERTIA * process_w
        + WEIGHT_DEBT * debt_scaled
        + WEIGHT_CIRCADIAN * (100.0 - process_c)
    )

    sensitivity_delta = fatigue_sensitivity - 1.0
    score += sensitivity_delta * 10 * SENSITIVITY_BIAS_PER_TENTH

    return round(max(0.0, min(100.0, score)), 2)
