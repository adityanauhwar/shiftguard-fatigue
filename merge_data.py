"""
merge_data.py
=============
Joins crew.csv, sleep.csv, duty_logs.csv and fatigue_reports.csv on crew_id
into a single set of enriched, time-sorted DataFrames that the fatigue
engine can query.

We do NOT flatten everything into one giant row-per-event table (sleep,
duty and self-reports happen at different timestamps and different
frequencies per crew member). Instead we return a `MergedData` bundle
containing:

    crew            -> one row per crew member (their static profile)
    sleep           -> every sleep session, with crew profile columns attached
    duty            -> every duty/flight, with crew profile columns attached
    fatigue_reports -> every self-reported fatigue entry, with crew profile
                        columns attached

All datetime columns are parsed to pandas Timestamps and each table is
sorted by crew_id + time so the fatigue engine can do fast "as of" lookups
(e.g. "last sleep before this duty").

Usage
-----
    from merge_data import load_merged_data

    data = load_merged_data("data")
    data.sleep_for(crew_id=101)
    data.duty_for(crew_id=101)
    data.crew_profile(crew_id=101)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

DATE_COLUMNS = {
    "sleep": ["sleep_start", "sleep_end", "created_at"],
    "duty": ["duty_start", "duty_end"],
    "fatigue_reports": ["report_date", "created_at"],
}

CREW_PROFILE_COLS = [
    "crew_id",
    "employee_id",
    "first_name",
    "last_name",
    "rank",
    "fleet",
    "base",
    "timezone",
    "chronotype",
    "sleep_need",
    "fatigue_sensitivity",
    "baseline_hrv",
]


def _read_csv(path: str, date_cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in date_cols or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


@dataclass
class MergedData:
    crew: pd.DataFrame
    sleep: pd.DataFrame
    duty: pd.DataFrame
    fatigue_reports: pd.DataFrame

    # ---- convenience accessors used by the fatigue engine ----------

    def crew_profile(self, crew_id: int) -> pd.Series:
        row = self.crew.loc[self.crew.crew_id == crew_id]
        if row.empty:
            raise KeyError(f"No crew profile for crew_id={crew_id}")
        return row.iloc[0]

    def sleep_for(self, crew_id: int) -> pd.DataFrame:
        return self.sleep.loc[self.sleep.crew_id == crew_id].sort_values("sleep_end")

    def duty_for(self, crew_id: int) -> pd.DataFrame:
        return self.duty.loc[self.duty.crew_id == crew_id].sort_values("duty_start")

    def fatigue_reports_for(self, crew_id: int) -> pd.DataFrame:
        return self.fatigue_reports.loc[self.fatigue_reports.crew_id == crew_id].sort_values(
            "report_date"
        )

    def all_crew_ids(self) -> list[int]:
        return sorted(self.crew.crew_id.unique().tolist())


def load_merged_data(data_dir: str = "data") -> MergedData:
    """Load the four CSVs from `data_dir` and join them on crew_id."""

    crew = _read_csv(os.path.join(data_dir, "crew.csv"))
    sleep = _read_csv(os.path.join(data_dir, "sleep.csv"), DATE_COLUMNS["sleep"])
    duty = _read_csv(os.path.join(data_dir, "duty_logs.csv"), DATE_COLUMNS["duty"])
    fatigue_reports = _read_csv(
        os.path.join(data_dir, "fatigue_reports.csv"), DATE_COLUMNS["fatigue_reports"]
    )

    profile_cols = [c for c in CREW_PROFILE_COLS if c in crew.columns]

    sleep = sleep.merge(crew[profile_cols], on="crew_id", how="left")
    duty = duty.merge(crew[profile_cols], on="crew_id", how="left")
    fatigue_reports = fatigue_reports.merge(crew[profile_cols], on="crew_id", how="left")

    sleep = sleep.sort_values(["crew_id", "sleep_end"]).reset_index(drop=True)
    duty = duty.sort_values(["crew_id", "duty_start"]).reset_index(drop=True)
    fatigue_reports = fatigue_reports.sort_values(
        ["crew_id", "report_date"]
    ).reset_index(drop=True)

    return MergedData(
        crew=crew, sleep=sleep, duty=duty, fatigue_reports=fatigue_reports
    )


if __name__ == "__main__":
    data = load_merged_data("data")
    print(f"Crew members : {len(data.crew)}")
    print(f"Sleep records: {len(data.sleep)}")
    print(f"Duty records : {len(data.duty)}")
    print(f"Reports      : {len(data.fatigue_reports)}")
