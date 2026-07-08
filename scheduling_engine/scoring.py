"""
scoring.py
==========
Turns each *eligible* candidate (everyone who already passed
`eligibility.check_eligibility`) into a single cost - lower is a better
fit for the shift. This is where the optimization happens; eligibility
already handled everything that must never be violated.

cost = predicted_fatigue                 (biomathematical, 0-100 scale)
     + cluster_risk_penalty              (context nudge, not a trigger)
     + workload_fairness_penalty         (spread shifts across the roster)

Each term is named and weighted separately so the trade-off is
inspectable and tunable, same convention as the rest of the project.
"""

from __future__ import annotations

from dataclasses import dataclass

# How much one cluster-risk-rank step is worth, in the same units as the
# 0-100 fatigue score. Small relative to the fatigue score itself - the
# cluster is context, not the primary driver (mirrors the same principle
# in recommendation_engine/rules.py: cluster escalates, never triggers).
CLUSTER_RISK_WEIGHT = 3.0

# Cost added per shift a candidate has already been assigned earlier in
# this same scheduling run, so an optimizer that only minimized fatigue
# wouldn't keep dumping every open shift onto the single most-rested
# person on the roster.
WORKLOAD_FAIRNESS_WEIGHT = 8.0


@dataclass
class CandidateScore:
    crew_id: int
    predicted_fatigue: float
    cluster_id: int
    cluster_label: str
    cluster_risk_rank: int
    assigned_count_before: int
    cost: float


def score_candidate(
    crew_id: int,
    predicted_fatigue: float,
    cluster_id: int,
    cluster_label: str,
    cluster_risk_rank: int,
    num_clusters: int,
    assigned_count_before: int,
) -> CandidateScore:
    cluster_penalty = max(0, num_clusters - cluster_risk_rank) * CLUSTER_RISK_WEIGHT
    workload_penalty = assigned_count_before * WORKLOAD_FAIRNESS_WEIGHT
    cost = predicted_fatigue + cluster_penalty + workload_penalty
    return CandidateScore(
        crew_id=crew_id,
        predicted_fatigue=predicted_fatigue,
        cluster_id=cluster_id,
        cluster_label=cluster_label,
        cluster_risk_rank=cluster_risk_rank,
        assigned_count_before=assigned_count_before,
        cost=cost,
    )
