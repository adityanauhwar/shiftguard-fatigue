"""
fatigue_engine
==============
A biomathematical fatigue-risk model based on the classic "Three Process
Model" of alertness (Åkerstedt & Folkard) built on top of Borbély's
two-process model of sleep regulation:

    Process C  - Circadian process   (body-clock oscillation)
    Process S  - Homeostatic process (sleep pressure, builds awake/decays asleep)
    Process W  - Sleep inertia       (grogginess in the minutes/hours after waking)

These three processes are combined into a single 0-100 Alertness Score and
its complement, the Base Fatigue Score, and are further adjusted by an
individual's accumulated Sleep Debt relative to their personal sleep need
and a per-person `fatigue_sensitivity` multiplier (from crew.csv).

Public entry point: `fatigue_engine.engine.run_fatigue_model`
"""

from .engine import run_fatigue_model, run_for_all_crew  # noqa: F401
