"""
scheduler.py
============
Orchestrates the Smart Scheduling Recommendation pipeline:

    1. recommendation_engine.feature_extraction + clustering  -> reuse the
       existing fatigue-pattern clusters as context (no re-implementation).
    2. shift_pool                                              -> the batch
       of open shifts needing a crew member.
    3. eligibility.check_eligibility(...)                      -> hard
       rules filter each shift's candidate pool down to who's allowed.
    4. scoring.score_candidate(...)                            -> cost
       each eligible candidate; lowest cost wins the shift.

This is a **sequential resource-allocation** problem, not a static
bipartite matching: assigning shift A to someone changes their rest
clock and rolling-hour totals for shift B. So shifts are processed in
chronological (duty_start) order with a mutable per-crew ledger
(`eligibility.CrewLedgerEntry`) carried forward - a classic greedy
heuristic for this class of scheduling problem, and one whose reasoning
stays fully auditable (each assignment records exactly why it won and
who the runner-up was), which matters more here than squeezing out the
last fraction of global optimality a full ILP solver might find.

Public entry point: `generate_schedule`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from fatigue_engine.engine import run_fatigue_model
from recommendation_engine.clustering import ClusteringResult, cluster_crew
from recommendation_engine.feature_extraction import build_feature_table

from .eligibility import CrewLedgerEntry, build_initial_ledger, check_eligibility
from .scoring import CandidateScore, score_candidate
from .shift_pool import OpenShift


@dataclass
class Assignment:
    shift_id: int
    flight_no: str
    duty_start: str
    duty_end: str
    crew_id: int
    predicted_fatigue: float
    cluster_label: str
    rationale: str
    runner_up: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "shift_id": self.shift_id,
            "flight_no": self.flight_no,
            "duty_start": self.duty_start,
            "duty_end": self.duty_end,
            "assigned_crew_id": self.crew_id,
            "predicted_fatigue_score": round(self.predicted_fatigue, 2),
            "cluster_label": self.cluster_label,
            "rationale": self.rationale,
            "runner_up": self.runner_up,
        }


@dataclass
class UnfilledShift:
    shift_id: int
    flight_no: str
    duty_start: str
    reasons: dict[int, list[str]]  # crew_id -> why they were excluded

    def to_dict(self) -> dict:
        # Keep this readable: summarize the most common exclusion reasons
        # rather than dumping every candidate's full reason list.
        reason_counts: dict[str, int] = {}
        for reasons in self.reasons.values():
            for r in reasons:
                key = r.split(" (")[0]  # group "insufficient rest (11.2h < 12h)" under one bucket
                reason_counts[key] = reason_counts.get(key, 0) + 1
        return {
            "shift_id": self.shift_id,
            "flight_no": self.flight_no,
            "duty_start": self.duty_start,
            "candidates_considered": len(self.reasons),
            "exclusion_reason_summary": reason_counts,
        }


def _rationale(winner: CandidateScore, runner_up: Optional[CandidateScore]) -> str:
    text = (
        f"Crew {winner.crew_id} had the lowest projected cost "
        f"({winner.cost:.1f}: fatigue {winner.predicted_fatigue:.1f} + cluster/workload context) "
        f"among eligible candidates, cluster '{winner.cluster_label}'."
    )
    if runner_up is not None:
        margin = runner_up.cost - winner.cost
        text += (
            f" Runner-up was crew {runner_up.crew_id} (cost {runner_up.cost:.1f}, "
            f"fatigue {runner_up.predicted_fatigue:.1f}), margin {margin:.1f}."
        )
    else:
        text += " No other eligible candidate existed for this shift."
    return text


def generate_schedule(data, shifts: list[OpenShift]) -> dict:
    """
    Run the full Smart Scheduling pipeline over a batch of open shifts.

    Parameters
    ----------
    data: MergedData (from `merge_data.load_merged_data`)
    shifts: list[OpenShift] (from `shift_pool.load_open_shifts` or
        `shift_pool.generate_demo_open_shifts`)

    Returns
    -------
    {
      "k": 4, "silhouette_score": 0.31,
      "clusters": [...],                       # same shape as recommendation_engine
      "assignments": [ {shift_id, assigned_crew_id, predicted_fatigue_score,
                         cluster_label, rationale, runner_up}, ... ],
      "unfilled_shifts": [ {shift_id, candidates_considered,
                             exclusion_reason_summary}, ... ],
      "summary": {"total_shifts", "filled", "unfilled", "fill_rate",
                  "mean_assigned_fatigue"},
    }
    """
    if not shifts:
        raise ValueError("shifts is empty - nothing to schedule")

    crew_ids = data.all_crew_ids()
    feature_df = build_feature_table(data, crew_ids=crew_ids)
    clustering: ClusteringResult = cluster_crew(feature_df)

    ordered_shifts = sorted(shifts, key=lambda s: s.duty_start)
    horizon_start = ordered_shifts[0].duty_start
    ledger: dict[int, CrewLedgerEntry] = build_initial_ledger(data, crew_ids, horizon_start)
    assigned_count: dict[int, int] = {cid: 0 for cid in crew_ids}

    crew_by_id = {int(row.crew_id): row for row in data.crew.itertuples()}

    assignments: list[Assignment] = []
    unfilled: list[UnfilledShift] = []

    for shift in ordered_shifts:
        exclusion_reasons: dict[int, list[str]] = {}
        candidate_scores: list[CandidateScore] = []

        for crew_id in crew_ids:
            profile = data.crew_profile(crew_id)
            fatigue_result = run_fatigue_model(data, crew_id=crew_id, as_of=shift.duty_start)
            is_eligible, reasons = check_eligibility(
                crew_profile=profile,
                shift=shift,
                ledger_entry=ledger[crew_id],
                current_sleep_debt=fatigue_result["sleep_debt"],
            )
            if not is_eligible:
                exclusion_reasons[crew_id] = reasons
                continue

            cluster_id = int(clustering.labels.loc[crew_id])
            candidate_scores.append(score_candidate(
                crew_id=crew_id,
                predicted_fatigue=fatigue_result["base_fatigue_score"],
                cluster_id=cluster_id,
                cluster_label=clustering.archetypes[cluster_id],
                cluster_risk_rank=clustering.risk_rank[cluster_id],
                num_clusters=clustering.k,
                assigned_count_before=assigned_count[crew_id],
            ))

        if not candidate_scores:
            unfilled.append(UnfilledShift(
                shift_id=shift.shift_id, flight_no=shift.flight_no,
                duty_start=shift.duty_start.isoformat(), reasons=exclusion_reasons,
            ))
            continue

        candidate_scores.sort(key=lambda c: c.cost)
        winner = candidate_scores[0]
        runner_up = candidate_scores[1] if len(candidate_scores) > 1 else None

        assignments.append(Assignment(
            shift_id=shift.shift_id, flight_no=shift.flight_no,
            duty_start=shift.duty_start.isoformat(), duty_end=shift.duty_end.isoformat(),
            crew_id=winner.crew_id, predicted_fatigue=winner.predicted_fatigue,
            cluster_label=winner.cluster_label,
            rationale=_rationale(winner, runner_up),
            runner_up=({
                "crew_id": runner_up.crew_id,
                "cost": round(runner_up.cost, 2),
                "predicted_fatigue_score": round(runner_up.predicted_fatigue, 2),
            } if runner_up is not None else None),
        ))

        ledger[winner.crew_id].register(shift.duty_start, shift.duty_end, shift.sectors)
        assigned_count[winner.crew_id] += 1

    total = len(ordered_shifts)
    filled = len(assignments)
    mean_fatigue = (
        sum(a.predicted_fatigue for a in assignments) / filled if filled else 0.0
    )

    from .scheduler_reporting import cluster_summary  # local import avoids a cycle at module load

    return {
        "k": clustering.k,
        "silhouette_score": round(clustering.silhouette, 3),
        "clusters": cluster_summary(clustering, feature_df),
        "assignments": [a.to_dict() for a in assignments],
        "unfilled_shifts": [u.to_dict() for u in unfilled],
        "summary": {
            "total_shifts": total,
            "filled": filled,
            "unfilled": total - filled,
            "fill_rate": round(filled / total, 3) if total else 0.0,
            "mean_assigned_fatigue": round(mean_fatigue, 2),
        },
    }
