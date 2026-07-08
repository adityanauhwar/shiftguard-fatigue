"""
shift_pool.py
=============
The "open shifts" that need a crew member assigned - the demand side of
the scheduling problem (crew are the supply side, from `merge_data`).

An `OpenShift` mirrors the shape of a row in `duty_logs.csv` (so the same
fatigue engine that scores historical duties can score a hypothetical
future one) plus the *qualification requirements* a real roster line
would carry: which rank, fleet and home base this shift needs.

Two ways to get a shift pool:
  - `load_open_shifts(path)`      - read a real `open_shifts.csv`.
  - `generate_demo_open_shifts()` - synthesize a plausible batch of near-
    future shifts by resampling realistic route/duration/timezone
    patterns straight out of the project's own `duty_logs.csv`, so the
    scheduler can be demoed end-to-end without hand-authoring shift data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

IATA_RE = re.compile(r"\(([A-Z]{3})\)")

REQUIRED_COLUMNS = [
    "shift_id", "flight_no", "departure", "arrival", "duty_start", "duty_end",
    "sectors", "timezone_crossed", "required_rank", "required_fleet", "required_base",
]


def base_to_iata(base: str) -> str:
    """'Delhi (DEL)' -> 'DEL'. Falls back to the raw string if unparsed."""
    match = IATA_RE.search(str(base))
    return match.group(1) if match else str(base)


@dataclass
class OpenShift:
    shift_id: int
    flight_no: str
    departure: str
    arrival: str
    duty_start: pd.Timestamp
    duty_end: pd.Timestamp
    sectors: int
    timezone_crossed: float
    required_rank: str
    required_fleet: str
    required_base_iata: str  # e.g. "DEL", already normalized to IATA

    @property
    def duty_hours(self) -> float:
        return (self.duty_end - self.duty_start).total_seconds() / 3600.0


def load_open_shifts(path: str) -> list[OpenShift]:
    """Load a shift pool from CSV. See REQUIRED_COLUMNS for the schema."""
    df = pd.read_csv(path, parse_dates=["duty_start", "duty_end"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"open_shifts file is missing columns: {missing}")

    shifts = []
    for row in df.itertuples():
        shifts.append(OpenShift(
            shift_id=int(row.shift_id),
            flight_no=str(row.flight_no),
            departure=str(row.departure),
            arrival=str(row.arrival),
            duty_start=pd.Timestamp(row.duty_start),
            duty_end=pd.Timestamp(row.duty_end),
            sectors=int(row.sectors),
            timezone_crossed=float(row.timezone_crossed),
            required_rank=str(row.required_rank),
            required_fleet=str(row.required_fleet),
            required_base_iata=base_to_iata(row.required_base),
        ))
    return shifts


def generate_demo_open_shifts(
    data,  # MergedData
    n_shifts: int = 30,
    days_ahead: int = 7,
    random_state: int = 7,
) -> pd.DataFrame:
    """
    Synthesize a plausible near-future shift pool by resampling real
    (route, duration, sectors, timezone_crossed) combinations out of
    `duty_logs.csv`, scheduled to start after the latest timestamp in the
    dataset. Each sampled historical duty already belongs to some crew
    member, so we borrow *their* rank/fleet/base as the shift's
    qualification requirement - this keeps every synthetic shift
    staffable by at least one real crew profile, while still needing the
    scheduler to find the *right* one.

    Returns a DataFrame in the `REQUIRED_COLUMNS` schema (ready to write
    to CSV via `.to_csv(..., index=False)` or feed straight into
    `load_open_shifts` after a round-trip).
    """
    rng = np.random.default_rng(random_state)

    duty = data.duty.copy()
    anchor = duty.duty_end.max() + pd.Timedelta(days=1)
    window_end = anchor + pd.Timedelta(days=days_ahead)

    sample_idx = rng.integers(0, len(duty), size=n_shifts)
    sampled = duty.iloc[sample_idx].reset_index(drop=True)

    rows = []
    for i, row in enumerate(sampled.itertuples(), start=1):
        duration = row.duty_end - row.duty_start
        offset_seconds = rng.uniform(0, (window_end - anchor).total_seconds())
        new_start = anchor + pd.Timedelta(seconds=offset_seconds)
        rows.append({
            "shift_id": i,
            "flight_no": row.flight_no,
            "departure": row.departure,
            "arrival": row.arrival,
            "duty_start": new_start,
            "duty_end": new_start + duration,
            "sectors": row.sectors,
            "timezone_crossed": row.timezone_crossed,
            "required_rank": row.rank,
            "required_fleet": row.fleet,
            "required_base": row.base,
        })

    out = pd.DataFrame(rows).sort_values("duty_start").reset_index(drop=True)
    out["shift_id"] = range(1, len(out) + 1)
    return out
