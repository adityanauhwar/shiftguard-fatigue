"""
run_recommendations.py
=======================
CLI for the Personalized Adjustment Recommendation feature (K-Means
clustering + rule engine).

Examples
--------
    # Full roster: clusters + every crew member's recommendations
    python run_recommendations.py --all --out recommendations.json

    # Just one crew member (still clustered against the full roster,
    # so cluster context is meaningful)
    python run_recommendations.py --crew-id 101

    # Just the cluster summary (archetypes, sizes, centroids)
    python run_recommendations.py --clusters-only
"""

from __future__ import annotations

import argparse
import json

from merge_data import load_merged_data
from recommendation_engine import generate_for_crew, generate_recommendations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ShiftGuard Personalized Adjustment Recommendations (K-Means + rule engine)"
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing the 4 CSVs")
    parser.add_argument("--crew-id", type=int, help="Show recommendations for a single crew member")
    parser.add_argument("--all", action="store_true", help="Show the full roster result")
    parser.add_argument("--clusters-only", action="store_true", help="Show only the cluster summary")
    parser.add_argument("--out", help="Optional path to write JSON output to")
    args = parser.parse_args()

    data = load_merged_data(args.data_dir)

    if args.crew_id is not None:
        result = generate_for_crew(data, crew_id=args.crew_id)
    else:
        result = generate_recommendations(data)
        if args.clusters_only:
            result = {"k": result["k"], "silhouette_score": result["silhouette_score"], "clusters": result["clusters"]}
        elif not args.all:
            parser.error("Specify --crew-id <id>, --all, or --clusters-only")
            return

    output = json.dumps(result, indent=2)
    print(output)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
