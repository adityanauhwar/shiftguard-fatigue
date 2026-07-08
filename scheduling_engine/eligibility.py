"""
eligibility.py
==============
Hard rules deciding who is *allowed* to take a shift - never a matter of
optimization score. Same convention as `recommendation_engine/rules.py`:
every threshold is named and documented so it can be re-tuned without
hunting through the logic.

These are non-negotiable, regulator-style constraints. A candidate who
fails any one of them is dropped from consideration entirely, no matter
how good their fatigue score looks - a scheduler that lets a great
optimization cost overrule a rest-period violation isn't safe to use.

The one soft-in-name-but-effectively-hard rule is the fitness-for-duty
sleep-debt gate: it exists here (not in `scoring.py`) because "already
critically sleep-debted" is a go/no-go safety call, not a preference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .shift_pool import OpenShift, base_to_iata

# Minimum turnaround (hours) required between the end of a crew member's
# last duty (real or already assigned earlier in this scheduling run) and
# the start of a new one. Set below the sample data's median `rest_before`
# (~24h) but above a bare-minimum regulatory-style floor, so the rule
# bites on genuinely tight turnarounds rather than routine ones.
MIN_REST_HOURS = 12.0

# Rolling flight-duty-period caps, mirroring how real rostering rules
# bound cumulative exposure over a trailing window rather than just
# per-duty limits.
ROLLING_WINDOW_DAYS = 7
MAX_DUTY_HOURS_ROLLING = 60.0
MAX_SECTORS_ROLLING = 12

# Fitness-for-duty gate (hours). Reuses the same sleep-debt tiers as
# `recommendation_engine/rules.py` (SLEEP_DEBT_CRITICAL_HOURS = 6.0) but
# adds a harder ceiling: beyond this, a crew member is not eligible for
# *any* new shift, full stop, regardless of role or how few alternatives
# exist.
SLEEP_DEBT_HARD_LOCKOUT_HOURS = 8.0
# Between the critical threshold and the hard lockout, a crew member is
# still barred from duties that *start* inside the window of circadian
# low (WOCL) specifically - the highest-risk combination of already
# under-recovered plus a low-alertness report time.
SLEEP_DEBT_WOCL_LOCKOUT_HOURS = 6.0
WOCL_START_HOUR, WOCL_END_HOUR = 0, 6


@dataclass
class CrewLedgerEntry:
    """Mutable per-crew running state, updated as the scheduler assigns
    shifts in duty_start order, so rest/rolling-window rules see
    provisional assignments from earlier in the *same* run - not just
    historical duty_logs."""
    last_duty_end: pd.Timestamp | None
    recent_duties: list[tuple[pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)
    # recent_duties: (duty_start, duty_end, sectors) within the rolling window,
    # combining real history and assignments made earlier this run.

    def register(self, duty_start: pd.Timestamp, duty_end: pd.Timestamp, sectors: int) -> None:
        self.recent_duties.append((duty_start, duty_end, sectors))
        if self.last_duty_end is None or duty_end > self.last_duty_end:
            self.last_duty_end = duty_end

    def rolling_hours_and_sectors(self, as_of: pd.Timestamp) -> tuple[float, int]:
        window_start = as_of - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
        hours = 0.0
        sectors = 0
        for start, end, sec in self.recent_duties:
            if start >= window_start and start <= as_of:
                hours += (end - start).total_seconds() / 3600.0
                sectors += sec
        return hours, sectors


def build_initial_ledger(data, crew_ids: list[int], horizon_start: pd.Timestamp) -> dict[int, CrewLedgerEntry]:
    """Seed each crew member's ledger from their real `duty_logs.csv`
    history, so rest and rolling-window checks for the *first* shift they
    might be assigned already reflect their real recent workload."""
    ledger: dict[int, CrewLedgerEntry] = {}
    lookback_start = horizon_start - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
    for crew_id in crew_ids:
        duty_df = data.duty_for(crew_id)
        past = duty_df.loc[duty_df.duty_end <= horizon_start]
        entry = CrewLedgerEntry(last_duty_end=past.duty_end.max() if not past.empty else None)
        recent = past.loc[past.duty_start >= lookback_start]
        for row in recent.itertuples():
            entry.recent_duties.append((row.duty_start, row.duty_end, int(row.sectors)))
        ledger[crew_id] = entry
    return ledger


def _is_wocl_start(duty_start: pd.Timestamp) -> bool:
    return WOCL_START_HOUR <= duty_start.hour < WOCL_END_HOUR


def check_eligibility(
    crew_profile: pd.Series,
    shift: OpenShift,
    ledger_entry: CrewLedgerEntry,
    current_sleep_debt: float,
) -> tuple[bool, list[str]]:
    """
    Returns (is_eligible, reasons). `reasons` is always populated - either
    the single "why excluded" reason(s), or, for an eligible candidate, an
    empty list.
    """
    reasons: list[str] = []

    # --- Qualification match -------------------------------------------
    if str(crew_profile["rank"]) != shift.required_rank:
        reasons.append(f"rank mismatch ({crew_profile['rank']} != {shift.required_rank})")
    if str(crew_profile["fleet"]) != shift.required_fleet:
        reasons.append(f"fleet mismatch ({crew_profile['fleet']} != {shift.required_fleet})")
    if base_to_iata(crew_profile["base"]) != shift.required_base_iata:
        reasons.append(f"base mismatch ({base_to_iata(crew_profile['base'])} != {shift.required_base_iata})")
    if reasons:
        return False, reasons  # no point checking rest/FTL for an unqualified candidate

    # --- Minimum rest ----------------------------------------------------
    if ledger_entry.last_duty_end is not None:
        gap_hours = (shift.duty_start - ledger_entry.last_duty_end).total_seconds() / 3600.0
        if gap_hours < MIN_REST_HOURS:
            reasons.append(f"insufficient rest ({gap_hours:.1f}h < {MIN_REST_HOURS:.0f}h minimum)")

    # --- Rolling flight-duty-period caps ----------------------------------
    hours_before, sectors_before = ledger_entry.rolling_hours_and_sectors(shift.duty_start)
    projected_hours = hours_before + shift.duty_hours
    projected_sectors = sectors_before + shift.sectors
    if projected_hours > MAX_DUTY_HOURS_ROLLING:
        reasons.append(
            f"rolling {ROLLING_WINDOW_DAYS}-day duty hours would reach {projected_hours:.1f}h "
            f"(> {MAX_DUTY_HOURS_ROLLING:.0f}h cap)"
        )
    if projected_sectors > MAX_SECTORS_ROLLING:
        reasons.append(
            f"rolling {ROLLING_WINDOW_DAYS}-day sectors would reach {projected_sectors} "
            f"(> {MAX_SECTORS_ROLLING} cap)"
        )

    # --- Fitness-for-duty sleep-debt gate ---------------------------------
    if current_sleep_debt >= SLEEP_DEBT_HARD_LOCKOUT_HOURS:
        reasons.append(
            f"fitness-for-duty lockout: sleep debt {current_sleep_debt:.1f}h "
            f">= {SLEEP_DEBT_HARD_LOCKOUT_HOURS:.0f}h hard ceiling"
        )
    elif current_sleep_debt >= SLEEP_DEBT_WOCL_LOCKOUT_HOURS and _is_wocl_start(shift.duty_start):
        reasons.append(
            f"barred from a WOCL-window report time with sleep debt {current_sleep_debt:.1f}h "
            f">= {SLEEP_DEBT_WOCL_LOCKOUT_HOURS:.0f}h"
        )

    return (len(reasons) == 0), reasons
