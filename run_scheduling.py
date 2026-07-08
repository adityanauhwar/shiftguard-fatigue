"""
run_scheduling.py
==================
CLI for the Smart Scheduling Recommendation / Optimization feature
(K-Means fatigue clustering + rule-based eligibility + greedy
lowest-fatigue-cost assignment).

Examples
--------
    # Demo mode: synthesizes a batch of near-future open shifts out of
    # the project's own duty_logs.csv patterns, then schedules them.
    python run_scheduling.py --demo --n-shifts 30 --out schedule.json

    # Real mode: schedule a shift pool you've prepared yourself.
    python run_scheduling.py --shifts data/open_shifts.csv --out schedule.json

    # Write the synthetic demo shifts to disk without running the
    # scheduler, e.g. to hand-edit before a real run.
    python run_scheduling.py --demo --n-shifts 30 --dump-shifts data/open_shifts.csv --no-run
"""

from __future__ import annotations

import argparse
import json

from merge_data import load_merged_data
from scheduling_engine import generate_schedule
from scheduling_engine.shift_pool import generate_demo_open_shifts, load_open_shifts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShiftGuard Smart Scheduling (K-Means fatigue clustering + rule-based scheduling)"
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing the 4 crew/duty/sleep CSVs")
    parser.add_argument("--shifts", help="Path to an open_shifts.csv (see shift_pool.REQUIRED_COLUMNS)")
    parser.add_argument("--demo", action="store_true", help="Synthesize a demo shift pool instead of --shifts")
    parser.add_argument("--n-shifts", type=int, default=30, help="Number of shifts to synthesize with --demo")
    parser.add_argument("--days-ahead", type=int, default=7, help="Scheduling horizon (days) for --demo")
    parser.add_argument("--dump-shifts", help="Write the (demo) shift pool to this CSV path")
    parser.add_argument("--no-run", action="store_true", help="Only dump shifts, don't run the scheduler")
    parser.add_argument("--out", help="Optional path to write the schedule JSON result to")
    args = parser.parse_args()

    if not args.shifts and not args.demo:
        parser.error("Specify --shifts <path> or --demo")

    data = load_merged_data(args.data_dir)

    if args.demo:
        shift_df = generate_demo_open_shifts(data, n_shifts=args.n_shifts, days_ahead=args.days_ahead)
        if args.dump_shifts:
            shift_df.to_csv(args.dump_shifts, index=False)
            print(f"Wrote {len(shift_df)} synthetic shifts to {args.dump_shifts}")
        if args.no_run:
            return
        if args.dump_shifts:
            shifts = load_open_shifts(args.dump_shifts)
        else:
            tmp_path = "_demo_open_shifts_tmp.csv"
            shift_df.to_csv(tmp_path, index=False)
            shifts = load_open_shifts(tmp_path)
    else:
        shifts = load_open_shifts(args.shifts)

    result = generate_schedule(data, shifts)

    output = json.dumps(result, indent=2)
    print(output)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
