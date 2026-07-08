"""
rules.py
========
The rule engine half of the Personalized Adjustment Recommendation
feature: turns one crew member's own feature values into specific,
explainable rest / schedule adjustment recommendations.

Design principle: clustering tells you *which population a crew member
resembles*; it must never be the sole trigger for a recommendation,
because two people in the same cluster can still differ meaningfully
from each other. Every rule below fires on the individual's *own*
metric against a named, tunable threshold. The cluster archetype is
used only for two secondary things:

  1. Context - each recommendation is machine-readable and carries the
     rule that fired, so a human reviewer can audit *why*.
  2. Priority escalation - if a crew member's own value on the metric
     that fired is *worse than their own cluster's centroid* (i.e. they
     are an above-average-risk member even within an already elevated
     group), the recommendation is bumped up one priority level.

Priority scale: "Low" < "Medium" < "High" < "Critical".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]

# --- Rule thresholds ---------------------------------------------------
# Each threshold is named and documented so it can be re-tuned without
# hunting through the logic below (same convention as fatigue_engine).

# Sleep debt (hours) - tiers loosely follow how multi-day partial sleep
# restriction studies characterize meaningful cumulative deficits.
SLEEP_DEBT_CRITICAL_HOURS = 6.0
SLEEP_DEBT_HIGH_HOURS = 3.0
SLEEP_DEBT_MEDIUM_HOURS = 1.5

# Fraction of recent duties preceded by a short rest turnaround.
SHORT_REST_HIGH_FREQ = 0.40
SHORT_REST_MEDIUM_FREQ = 0.20

# Circadian variability (std of Process C across recent duties, 0-100
# scale) above which duty timing is swinging widely across the body
# clock rather than settling into a stable pattern.
CIRCADIAN_VARIABILITY_HIGH = 18.0

# Fraction of duties starting inside the WOCL window.
NIGHT_DUTY_HIGH_FREQ = 0.25
NIGHT_DUTY_MEDIUM_FREQ = 0.10

# Mean Process W (sleep inertia, 0-100 scale) at duty time.
PROCESS_W_MEDIUM = 12.0
PROCESS_W_HIGH = 20.0

# Self-rated sleep quality is on a 1-5 scale in sleep.csv.
SLEEP_QUALITY_LOW = 3.0
SLEEP_QUALITY_VARIABILITY_HIGH = 1.1

# HRV deviation (bpm-equivalent units, recent mean minus personal
# baseline_hrv). Negative = below one's own baseline, a physiological
# strain marker independent of self-report.
HRV_DEVIATION_HIGH_STRAIN = -8.0
HRV_DEVIATION_MEDIUM_STRAIN = -4.0

# Samn-Perelli is a validated 1 (fully alert) - 7 (completely exhausted)
# subjective fatigue scale.
SAMN_PERELLI_HIGH = 5.0
SAMN_PERELLI_MEDIUM = 4.0

# Workload: sectors per duty, used only in combination with an already
# elevated fatigue score (workload alone isn't a fatigue signal).
SECTORS_HIGH = 2.5
BASE_FATIGUE_SCORE_ELEVATED = 55.0


@dataclass
class Recommendation:
    code: str
    priority: str  # one of PRIORITY_ORDER
    title: str
    rationale: str
    action: str
    metric: str
    value: float

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "priority": self.priority,
            "title": self.title,
            "rationale": self.rationale,
            "action": self.action,
            "trigger_metric": self.metric,
            "trigger_value": round(float(self.value), 2),
        }


def _escalate(priority: str, individual_value: float, cluster_centroid_value: float, worse_is_higher: bool = True) -> str:
    """
    Bump `priority` up one level if the individual is worse than their
    own cluster's centroid on the metric that triggered the rule -
    i.e. they're an above-average-risk member even within an already
    elevated-risk group.
    """
    is_worse = (
        individual_value > cluster_centroid_value
        if worse_is_higher
        else individual_value < cluster_centroid_value
    )
    if not is_worse:
        return priority
    idx = PRIORITY_ORDER.index(priority)
    return PRIORITY_ORDER[min(idx + 1, len(PRIORITY_ORDER) - 1)]


def generate_recommendations(
    features: pd.Series,
    cluster_centroid: pd.Series,
) -> list[Recommendation]:
    """
    Generate the list of personalized recommendations for one crew
    member.

    Parameters
    ----------
    features: this crew member's own feature values (see
        `feature_extraction.FEATURE_NAMES`).
    cluster_centroid: the mean feature values of the cluster this crew
        member was assigned to - used only for priority escalation
        (see module docstring), never as a trigger by itself.
    """
    recs: list[Recommendation] = []

    # --- Sleep debt ------------------------------------------------------
    debt = features["mean_sleep_debt_hours"]
    if debt >= SLEEP_DEBT_CRITICAL_HOURS:
        priority = _escalate("Critical", debt, cluster_centroid["mean_sleep_debt_hours"])
        recs.append(Recommendation(
            code="sleep_debt_critical",
            priority=priority,
            title="Mandate a protected multi-night recovery break",
            rationale=f"Recent accumulated sleep debt averages {debt:.1f}h, well beyond what a single "
                      "night of recovery sleep can pay down (recovery nights only offset debt partially).",
            action="Schedule at least two consecutive full local nights off duty before the next assignment, "
                   "and re-check sleep debt before returning this crew member to the line.",
            metric="mean_sleep_debt_hours", value=debt,
        ))
    elif debt >= SLEEP_DEBT_HIGH_HOURS:
        priority = _escalate("High", debt, cluster_centroid["mean_sleep_debt_hours"])
        recs.append(Recommendation(
            code="sleep_debt_high",
            priority=priority,
            title="Schedule a full recovery night before next duty",
            rationale=f"Recent accumulated sleep debt averages {debt:.1f}h, indicating recovery is "
                      "consistently lagging behind personal sleep need.",
            action="Ensure the next rostered rest period includes one full uninterrupted local night "
                   "(not just the regulatory minimum turnaround).",
            metric="mean_sleep_debt_hours", value=debt,
        ))
    elif debt >= SLEEP_DEBT_MEDIUM_HOURS:
        priority = _escalate("Medium", debt, cluster_centroid["mean_sleep_debt_hours"])
        recs.append(Recommendation(
            code="sleep_debt_medium",
            priority=priority,
            title="Add a modest rest buffer to the next turnaround",
            rationale=f"Recent accumulated sleep debt averages {debt:.1f}h - a mild but persistent shortfall.",
            action="Add roughly 2 extra hours to the next scheduled rest period where roster flexibility allows.",
            metric="mean_sleep_debt_hours", value=debt,
        ))

    # --- Short-rest turnarounds -------------------------------------------
    short_rest_freq = features["short_rest_frequency"]
    if short_rest_freq >= SHORT_REST_HIGH_FREQ:
        priority = _escalate("High", short_rest_freq, cluster_centroid["short_rest_frequency"])
        recs.append(Recommendation(
            code="short_rest_high",
            priority=priority,
            title="Raise this crew member's personal minimum rest buffer",
            rationale=f"{short_rest_freq:.0%} of recent duties were preceded by a short rest turnaround.",
            action="Set a personal minimum rest floor a few hours above the regulatory minimum for the "
                   "next roster cycle, rather than relying on the fleet-wide default.",
            metric="short_rest_frequency", value=short_rest_freq,
        ))
    elif short_rest_freq >= SHORT_REST_MEDIUM_FREQ:
        priority = _escalate("Medium", short_rest_freq, cluster_centroid["short_rest_frequency"])
        recs.append(Recommendation(
            code="short_rest_medium",
            priority=priority,
            title="Monitor rest turnaround frequency",
            rationale=f"{short_rest_freq:.0%} of recent duties were preceded by a short rest turnaround.",
            action="Where crew-pairing flexibility exists, avoid stacking another short turnaround "
                   "immediately after this one.",
            metric="short_rest_frequency", value=short_rest_freq,
        ))

    # --- Circadian disruption / night ops ----------------------------------
    circadian_var = features["circadian_variability"]
    night_freq = features["night_duty_fraction"]
    if circadian_var >= CIRCADIAN_VARIABILITY_HIGH and night_freq >= NIGHT_DUTY_MEDIUM_FREQ:
        priority = _escalate("High", circadian_var, cluster_centroid["circadian_variability"])
        recs.append(Recommendation(
            code="circadian_disruption_high",
            priority=priority,
            title="Reduce alternating day/night duty starts",
            rationale=f"Duty-time circadian exposure varies widely across recent duties (variability "
                      f"score {circadian_var:.1f}), combined with {night_freq:.0%} of duties starting "
                      "inside the window of circadian low - the body clock has little chance to settle.",
            action="Where operationally possible, group upcoming night-starting duties into blocks of "
                   "2-3 consecutive nights rather than isolated single night duties, to allow partial "
                   "circadian adaptation instead of repeated re-adjustment.",
            metric="circadian_variability", value=circadian_var,
        ))
    elif night_freq >= NIGHT_DUTY_HIGH_FREQ:
        priority = _escalate("Medium", night_freq, cluster_centroid["night_duty_fraction"])
        recs.append(Recommendation(
            code="night_duty_medium",
            priority=priority,
            title="Support upcoming night-starting duties",
            rationale=f"{night_freq:.0%} of recent duties started inside the window of circadian low.",
            action="Offer a pre-duty nap opportunity or a protected anchor-sleep window before night "
                   "departures.",
            metric="night_duty_fraction", value=night_freq,
        ))

    # --- Sleep inertia (post-wake grogginess) ------------------------------
    process_w = features["mean_process_w"]
    if process_w >= PROCESS_W_HIGH:
        priority = _escalate("High", process_w, cluster_centroid["mean_process_w"])
        recs.append(Recommendation(
            code="sleep_inertia_high",
            priority=priority,
            title="Add a buffer before safety-critical tasks after waking",
            rationale=f"Mean residual sleep inertia at duty time is {process_w:.1f} (0-100 scale), "
                      "indicating duty is often starting too soon after waking.",
            action="Shift report time later where possible, or build in a wake-to-duty buffer of at "
                   "least an hour before safety-critical tasks.",
            metric="mean_process_w", value=process_w,
        ))
    elif process_w >= PROCESS_W_MEDIUM:
        priority = _escalate("Medium", process_w, cluster_centroid["mean_process_w"])
        recs.append(Recommendation(
            code="sleep_inertia_medium",
            priority=priority,
            title="Watch wake-to-duty buffer",
            rationale=f"Mean residual sleep inertia at duty time is {process_w:.1f} (0-100 scale).",
            action="Where roster flexibility allows, avoid the earliest report times for this crew member.",
            metric="mean_process_w", value=process_w,
        ))

    # --- Sleep quality ------------------------------------------------------
    quality = features["sleep_quality_mean"]
    quality_var = features["sleep_quality_variability"]
    if quality <= SLEEP_QUALITY_LOW or quality_var >= SLEEP_QUALITY_VARIABILITY_HIGH:
        priority = _escalate("Medium", -quality, -cluster_centroid["sleep_quality_mean"])
        recs.append(Recommendation(
            code="sleep_quality_low",
            priority=priority,
            title="Flag for sleep-hygiene support",
            rationale=f"Self-rated sleep quality averages {quality:.1f}/5 with variability {quality_var:.1f}, "
                      "suggesting recovery sleep isn't as effective as the hours logged would suggest.",
            action="Offer sleep-hygiene coaching and, where roster patterns allow, aim for a more "
                   "consistent sleep-wake window rather than a fixed number of rest hours.",
            metric="sleep_quality_mean", value=quality,
        ))

    # --- Physiological strain (HRV) -----------------------------------------
    hrv_dev = features["hrv_deviation"]
    if hrv_dev <= HRV_DEVIATION_HIGH_STRAIN:
        priority = _escalate("High", -hrv_dev, -cluster_centroid["hrv_deviation"])
        recs.append(Recommendation(
            code="hrv_strain_high",
            priority=priority,
            title="Recommend a fatigue risk medical review",
            rationale=f"Recent HRV runs {abs(hrv_dev):.1f} below this crew member's personal baseline - "
                      "a physiological strain marker independent of self-report.",
            action="Route to fatigue risk / occupational health review before further high-workload "
                   "assignment, in parallel with the schedule adjustments above.",
            metric="hrv_deviation", value=hrv_dev,
        ))
    elif hrv_dev <= HRV_DEVIATION_MEDIUM_STRAIN:
        priority = _escalate("Medium", -hrv_dev, -cluster_centroid["hrv_deviation"])
        recs.append(Recommendation(
            code="hrv_strain_medium",
            priority=priority,
            title="Monitor physiological recovery trend",
            rationale=f"Recent HRV runs {abs(hrv_dev):.1f} below this crew member's personal baseline.",
            action="Keep tracking; escalate to medical review if the trend continues or worsens.",
            metric="hrv_deviation", value=hrv_dev,
        ))

    # --- Subjective fatigue --------------------------------------------------
    samn_perelli = features["mean_samn_perelli"]
    if samn_perelli >= SAMN_PERELLI_HIGH:
        priority = _escalate("High", samn_perelli, cluster_centroid["mean_samn_perelli"])
        recs.append(Recommendation(
            code="self_reported_fatigue_high",
            priority=priority,
            title="Prioritize for the next available schedule adjustment",
            rationale=f"Self-reported Samn-Perelli fatigue averages {samn_perelli:.1f}/7, "
                      "corroborating the objective flags above.",
            action="Prioritize this crew member for the next available roster adjustment slot rather "
                   "than deferring to the next planning cycle.",
            metric="mean_samn_perelli", value=samn_perelli,
        ))
    elif samn_perelli >= SAMN_PERELLI_MEDIUM:
        priority = _escalate("Medium", samn_perelli, cluster_centroid["mean_samn_perelli"])
        recs.append(Recommendation(
            code="self_reported_fatigue_medium",
            priority=priority,
            title="Cross-check self-reports against objective flags",
            rationale=f"Self-reported Samn-Perelli fatigue averages {samn_perelli:.1f}/7.",
            action="Keep an eye on upcoming self-reports; combine with objective flags before adjusting "
                   "the roster.",
            metric="mean_samn_perelli", value=samn_perelli,
        ))

    # --- Workload, only in combination with elevated fatigue score ---------
    sectors = features["mean_sectors_per_duty"]
    fatigue_score = features["mean_base_fatigue_score"]
    if sectors >= SECTORS_HIGH and fatigue_score >= BASE_FATIGUE_SCORE_ELEVATED:
        priority = _escalate("Medium", sectors, cluster_centroid["mean_sectors_per_duty"])
        recs.append(Recommendation(
            code="workload_high",
            priority=priority,
            title="Consider reducing sectors per duty",
            rationale=f"Averaging {sectors:.1f} sectors per duty alongside an elevated base fatigue "
                      f"score ({fatigue_score:.1f}/100) - workload is compounding the fatigue signals above.",
            action="Where crew-pairing flexibility allows, reduce sector count per duty for the next "
                   "few rotations rather than only adjusting rest.",
            metric="mean_sectors_per_duty", value=sectors,
        ))

    if not recs:
        recs.append(Recommendation(
            code="no_action_needed",
            priority="Low",
            title="No elevated fatigue-pattern flags detected",
            rationale="Recent sleep debt, rest turnarounds, circadian exposure, sleep inertia, sleep "
                      "quality, physiological and self-reported signals are all within a sustainable range.",
            action="Maintain the current roster pattern; continue routine monitoring.",
            metric="mean_base_fatigue_score", value=fatigue_score,
        ))

    # Highest-priority recommendation first.
    recs.sort(key=lambda r: PRIORITY_ORDER.index(r.priority), reverse=True)
    return recs


def overall_priority(recommendations: list[Recommendation]) -> str:
    """The single most urgent priority across a crew member's recommendations."""
    if not recommendations:
        return "Low"
    return max(recommendations, key=lambda r: PRIORITY_ORDER.index(r.priority)).priority
