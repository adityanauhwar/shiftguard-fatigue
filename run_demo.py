"""
run_demo.py
===========
Small CLI to exercise the fatigue engine end-to-end.

Examples
--------
    # Single crew member, evaluated "now" (latest timestamp on record)
    python run_demo.py --crew-id 101

    # Single crew member at a specific timestamp
    python run_demo.py --crew-id 101 --as-of "2026-01-20 06:58:00"

    # Every crew member, dumped to a JSON file
    python run_demo.py --all --out fatigue_results.json
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from merge_data import load_merged_data
from fatigue_engine.engine import run_fatigue_model, run_for_all_crew


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ShiftGuard Three-Process Fatigue Engine")
    parser.add_argument("--data-dir", default="data", help="Directory containing the 4 CSVs")
    parser.add_argument("--crew-id", type=int, help="Run for a single crew_id")
    parser.add_argument("--all", action="store_true", help="Run for every crew member")
    parser.add_argument("--as-of", help="Timestamp to evaluate at, e.g. '2026-01-20 06:58:00'")
    parser.add_argument("--out", help="Optional path to write JSON output to")
    args = parser.parse_args()

    data = load_merged_data(args.data_dir)
    as_of = pd.Timestamp(args.as_of) if args.as_of else None

    if args.all:
        results = run_for_all_crew(data, as_of=as_of)
    elif args.crew_id is not None:
        results = run_fatigue_model(data, crew_id=args.crew_id, as_of=as_of)
    else:
        parser.error("Specify --crew-id <id> or --all")
        return

    output = json.dumps(results, indent=2)
    print(output)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
