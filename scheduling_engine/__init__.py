"""
scheduling_engine
==================
Smart Scheduling Recommendation / Optimization module for ShiftGuard.

Where `recommendation_engine/` answers "what should we change about this
person's *existing* roster pattern?", this module answers a forward-looking
question: "given a batch of open shifts that need a crew member, who
should be assigned to each one?"

Scheduling is fundamentally an **optimization / constraint-satisfaction**
problem, not a prediction problem - there's no single "correct" label to
predict, only a search over feasible assignments for the one that best
balances competing goals (fatigue risk, legality, fairness). So this
module follows the same two-stage philosophy as `recommendation_engine`,
applied to assignment instead of advice:

1. **Clustering (unsupervised, reused as-is from `recommendation_engine`)**
   groups crew by fatigue-pattern archetype ("Chronic Sleep-Debt",
   "Circadian-Disrupted", "Well-Recovered", ...). Here it's used purely as
   *context* - a crew member's cluster nudges their assignment cost up or
   down, it never disqualifies them by itself.

2. **Hard rule-based eligibility** (`eligibility.py`) decides who is even
   *allowed* to take a shift: qualification match (rank/fleet/base),
   minimum rest since their last duty, rolling flight-duty-period caps on
   hours and sectors, and a fitness-for-duty gate on extreme sleep debt.
   These are non-negotiable, regulator-style constraints - never
   overridden by a "good enough" optimization score.

3. **Rule-weighted optimization** (`scoring.py` + `scheduler.py`) then
   picks the *best* among everyone who passed stage 2: a greedy, most-
   urgent-shift-first search that scores each eligible candidate by their
   biomathematically predicted fatigue at the shift's report time
   (re-running `fatigue_engine`), nudged by cluster risk context and a
   workload-fairness penalty so shifts don't all pile onto the single
   least-fatigued person.

Public entry point: `scheduler.generate_schedule`
"""

from .scheduler import generate_schedule  # noqa: F401
