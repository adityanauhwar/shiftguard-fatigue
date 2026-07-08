# ShiftGuard Fatigue Engine

A biomathematical fatigue-risk model built on the classic **Three Process
Model of Alertness** (Åkerstedt & Folkard), layered on **Borbély's
two-process model of sleep regulation**. It joins your `crew.csv`,
`sleep.csv`, `duty_logs.csv`, and `fatigue_reports.csv` datasets and turns
raw sleep/duty history into a per-crew fatigue and alertness score.

## Project layout

```
merge_data.py              # joins all 4 CSVs on crew_id
fatigue_engine/
  __init__.py
  circadian.py              # Process C - body-clock oscillation
  sleep_homeostat.py         # Process S - sleep pressure (build/decay)
  sleep_inertia.py            # Process W - grogginess after waking
  sleep_debt.py               # accumulated sleep debt (hours)
  alertness.py                # combines C, S, W, debt -> Alertness Score
  fatigue_score.py            # combines C, S, W, debt -> Base Fatigue Score
  engine.py                    # orchestrates the full pipeline
run_demo.py                 # CLI entry point
data/                        # your 4 CSVs
```

## The science, in brief

| Process | File | What it models |
|---|---|---|
| **Process C** | `circadian.py` | Two-harmonic cosine body-clock oscillation. Trough (~04:00-06:00) and peak shift with chronotype (`Morning` / `Neutral` / `Evening`) and residual jet lag from recent timezone crossings. |
| **Process S** | `sleep_homeostat.py` | Exponential-saturating build-up while awake, exponential decay while asleep (classic Borbély time constants: rise τ≈18.2h, decay τ≈4.2h). Simulated across each crew member's actual sleep history, so back-to-back short sleeps compound realistically. Sleep quality modulates how effectively pressure dissipates. |
| **Process W** | `sleep_inertia.py` | Sharp exponential decay from wake (τ≈45 min), fully resolved by ~4h. Deepened by short prior sleep and by accumulated sleep debt. |
| **Sleep Debt** | `sleep_debt.py` | Exponentially-weighted trailing 7-day deficit vs. each crew member's personal `sleep_need`. Recent nights count more; a recovery night only partially pays down existing debt. |
| **Alertness** | `alertness.py` | Weighted blend of C, S, W and debt, tuned for moment-to-moment performance capacity. |
| **Base Fatigue Score** | `fatigue_score.py` | A *separate* weighted blend of the same four inputs, tuned to emphasize operational/safety risk (leans more on pressure, inertia and debt than on the predictable circadian dip). |

Both `alertness.py` and `fatigue_score.py` apply an individual
**`fatigue_sensitivity`** (from `crew.csv`) as a small, capped **additive**
bias rather than a multiplicative slope on the whole curve — multiplicative
per-person scaling was tried and found to over-fit on sparse individual
history, so an additive nudge is the safer default.

## Output schema

```json
{
  "crew_id": 101,
  "process_c": 82.4,
  "process_s": 61.3,
  "process_w": 9.7,
  "sleep_debt": 2.4,
  "alertness": 74.5,
  "base_fatigue_score": 68.2
}
```

All process scores are on a 0-100 scale. `sleep_debt` is in hours.

## Usage

```bash
pip install -r requirements.txt

# Single crew member, evaluated "now" (latest timestamp in their data)
python run_demo.py --crew-id 101

# Single crew member at a specific moment (e.g. right before a flight)
python run_demo.py --crew-id 101 --as-of "2026-01-20 06:58:00"

# Every crew member at once, dumped to JSON
python run_demo.py --all --out fatigue_results.json
```

Or from Python / inside your Flask app:

```python
from merge_data import load_merged_data
from fatigue_engine.engine import run_fatigue_model, run_for_all_crew

data = load_merged_data("data")          # loads + joins the 4 CSVs once
result = run_fatigue_model(data, crew_id=101)   # -> dict, JSON-serializable
all_results = run_for_all_crew(data)             # -> list[dict], one per crew member
```

`data` (the `MergedData` object) is cheap to reuse across calls — load it
once at app startup (or per-request if your CSVs change) rather than
reloading per crew member.

## Personalized Adjustment Recommendation (K-Means + rule engine)

A second, additive layer on top of the fatigue engine: `recommendation_engine/`
groups crew members with similar fatigue *patterns* and turns each
individual's own metrics into specific, explainable rest/schedule
adjustment recommendations.

```
recommendation_engine/
  feature_extraction.py   # per-crew feature vector (re-runs the fatigue
                            # engine over recent duties + raw workload,
                            # physiological and self-report signals)
  clustering.py             # standardizes features, K-Means with
                            # silhouette-based k selection, derives a
                            # human-readable archetype label per cluster
  rules.py                  # named, tunable thresholds -> specific
                            # recommendations with priority + rationale
  recommender.py            # orchestrates the three stages above
run_recommendations.py     # CLI entry point
```

**Two-stage design.** Clustering (unsupervised) answers "what kind of
fatigue profile does this person resemble?" - it groups crew on
features like sleep debt, circadian variability, night-duty frequency,
sleep inertia, sleep quality, HRV deviation vs. personal baseline, and
self-reported (Samn-Perelli) fatigue, then labels each cluster from its
centroid (e.g. *"Chronic Sleep-Debt"*, *"Circadian-Disrupted"*,
*"Well-Recovered"*). The rule engine (domain-knowledge-driven) then
answers "what should we do about *this specific person*?" - every
recommendation is triggered by the individual's own value against a
named threshold, never by cluster membership alone; the cluster is
used only to add context and to escalate priority when someone is
above-average-risk even within an already elevated group.

```bash
python run_recommendations.py --all --out recommendations.json   # full roster
python run_recommendations.py --crew-id 101                       # one crew member
python run_recommendations.py --clusters-only                     # just the archetypes
```

Or from Python:

```python
from merge_data import load_merged_data
from recommendation_engine import generate_recommendations, generate_for_crew

data = load_merged_data("data")
result = generate_recommendations(data)      # {"k", "silhouette_score", "clusters", "crew"}
one = generate_for_crew(data, crew_id=101)    # single crew member + their cluster's summary
```

Each crew member's result includes their assigned cluster and archetype
label, an `overall_priority` ("Low"/"Medium"/"High"/"Critical"), and a
list of recommendations, each with the triggering metric/value, a
plain-language rationale, and a concrete action.

## Smart Scheduling Recommendation / Optimization (K-Means + rule-based assignment)

A third layer, built on top of both the fatigue engine and the
Personalized Adjustment Recommendation clusters: `scheduling_engine/`
takes a batch of **open shifts** that need a crew member and decides who
to assign to each one. Where the recommendation engine looks backward
("what should change about this person's existing pattern?"), this
module looks forward - it's the actual scheduling/optimization step,
which is why it's a clustering + rule engine rather than a predictive
model: there's no ground-truth label to predict here, only a search over
feasible assignments for the one that best balances fatigue risk,
legality and fairness.

```
scheduling_engine/
  shift_pool.py               # OpenShift model, CSV loader, and a demo
                                # generator that resamples realistic
                                # route/duration patterns from duty_logs.csv
  eligibility.py                # hard rules: qualification match, minimum
                                # rest, rolling 7-day duty-hour/sector caps,
                                # fitness-for-duty sleep-debt gate
  scoring.py                     # cost function for everyone who passed
                                # eligibility: predicted fatigue + cluster
                                # risk context + workload-fairness penalty
  scheduler.py                   # orchestrates clustering + eligibility +
                                # scoring into a full schedule
  scheduler_reporting.py        # small shared formatting helper
run_scheduling.py               # CLI entry point
data/open_shifts.csv            # a demo batch of synthetic near-future
                                # shifts (regenerate any time with --demo)
```

**Two-stage design, same philosophy as `recommendation_engine`.**
K-Means clustering (reused as-is, not re-implemented) supplies *context*
- a crew member's fatigue archetype nudges their assignment cost, but
never disqualifies them by itself. Hard, non-negotiable rules
(`eligibility.py`) decide who's even allowed to take a shift at all:
rank/fleet/base qualification match, a minimum rest turnaround, rolling
flight-duty-period caps on cumulative hours and sectors, and a
fitness-for-duty gate that locks out anyone with extreme sleep debt
(fully) or moderate sleep debt during a WOCL-window report time
(partially). Only among candidates who pass every hard rule does the
optimizer step in and rank by cost - each crew member's own
biomathematically *predicted* fatigue at that shift's report time
(re-running the fatigue engine forward), plus a small cluster-risk nudge
and a workload-fairness penalty so shifts don't all pile onto the single
most-rested person on the roster.

**Why greedy instead of a global matching solver (e.g. Hungarian
algorithm).** Assigning shift A to someone changes their rest clock and
rolling-hour totals for shift B - this is a *sequential* resource
allocation problem, not a static one-shot bipartite match. The scheduler
processes shifts in chronological order with a running per-crew ledger
carried forward, which correctly captures that dependency; a plain
assignment-problem solver would need every cost known up front and can't
express "this pairing is only legal if that earlier one didn't happen."
Every assignment records its winning cost, the runner-up, and the margin
between them, so the result is auditable rather than a black box.

```bash
# Demo mode: synthesizes a batch of near-future open shifts by
# resampling realistic patterns out of duty_logs.csv, then schedules them
python run_scheduling.py --demo --n-shifts 30 --out schedule.json

# Real mode: schedule a shift pool you've prepared yourself
python run_scheduling.py --shifts data/open_shifts.csv --out schedule.json
```

Or from Python:

```python
from merge_data import load_merged_data
from scheduling_engine import generate_schedule
from scheduling_engine.shift_pool import load_open_shifts, generate_demo_open_shifts

data = load_merged_data("data")
shifts = load_open_shifts("data/open_shifts.csv")   # or generate_demo_open_shifts(data)
result = generate_schedule(data, shifts)
```

The result includes the same cluster summary shape as the recommendation
engine, plus `assignments` (each with the predicted fatigue score,
cluster context, and a plain-language rationale naming the runner-up and
margin), `unfilled_shifts` (with a summary of why every candidate was
excluded - useful for spotting a roster gap, e.g. only one qualified crew
member exists and they're mid-rest), and a `summary` block (fill rate,
mean fatigue across assigned shifts).

A shift pool CSV needs: `shift_id, flight_no, departure, arrival,
duty_start, duty_end, sectors, timezone_crossed, required_rank,
required_fleet, required_base` (`required_base` in the same `"City
(IATA)"` format as `crew.csv`).

## Notes for integrating into ShiftGuard

- `run_fatigue_model` accepts an explicit `as_of` timestamp, so you can
  score a crew member at *any* point — e.g. right before a scheduled duty
  (for pre-flight go/no-go checks), or at "now" for a live dashboard tile.
- Timezone/jet-lag adjustment to Process C is approximated from the most
  recent duty's `timezone_crossed` and a ~1h/day adaptation rule of thumb —
  tune `_timezone_adaptation_shift` in `engine.py` if you have better
  data (e.g. actual local-time-at-base vs local-time-at-layover).
- All weight/time constants are named module-level constants at the top
  of each file — tune them there rather than inline in the formulas.

## Future Risk Prediction Engine (Random Forest classification)

A fourth layer, and the only genuinely **supervised, predictive** piece
in the project: `risk_prediction_engine/` trains a classifier on every
historical entry in `fatigue_reports.csv` to answer a forward-looking
question none of the other engines can — *"what fatigue-risk tier will
this crew member report/experience at a specific future moment?"* —
before anyone self-reports anything.

```
risk_prediction_engine/
  feature_extraction.py   # leak-safe, per-timestamp feature rows: engine
                            # outputs re-run "as of" the target moment,
                            # trailing 7-day workload/sleep windows, a
                            # lagged prior self-report - never the
                            # current report's own score/label - plus
                            # the RISK_TIERS collapse described below
  model.py                  # ColumnTransformer + RandomForestClassifier
                            # (class_weight="balanced_subsample" so the
                            # smaller tiers aren't ignored), trained and
                            # evaluated on a CHRONOLOGICAL split
  predictor.py               # loads the saved model and predicts risk
                            # at any as_of - including timestamps beyond
                            # the latest data on record (e.g. an
                            # open_shifts.csv duty_start)
train_risk_model.py         # CLI: builds the training table, fits, and
                            # evaluates on a chronological holdout
run_risk_prediction.py      # CLI: predicts risk for one/all crew, or
                            # scores a batch of open shifts
```

**Why 3 risk tiers rather than the raw 5-class `fatigue_level`.**
`fatigue_reports.csv` records `fatigue_level` as one of Low / Mild /
Moderate / High / Severe. An earlier version of this engine trained
directly on all 5 classes and got 40% accuracy (macro F1 0.35) on a
chronological holdout. Diagnosing that result showed the errors were
almost entirely between *adjacent* classes (Mild vs. Moderate vs. High
blur together) while the extremes were rarely confused with each other
— the noise sits at the boundaries between neighbors. `RISK_TIERS` in
`feature_extraction.py` collapses the 5 classes into 3 broader bands
(`FATIGUE_LEVEL_TO_TIER`: Low+Mild → **Low-Risk**, Moderate →
**Elevated**, High+Severe → **High-Risk**), which removes most of that
boundary noise while keeping a distinction that's still operationally
useful — and reads consistently with the Low/Medium/High/Critical tier
language `recommendation_engine` already uses for `overall_priority`.
On the same chronological holdout, this took accuracy from 40% to
**53.6%** (macro F1 0.35 → 0.49, mean class-rank error 0.80 → 0.60).
Hyperparameter tuning alone (more/deeper trees, no class balancing,
gradient boosting) was tried first and only moved accuracy within a
~38–42% band — the 5-vs-3-class framing, not the algorithm, was the
actual lever.

**Why Random Forest, and why this is different from the clustering
layers above.** `recommendation_engine` and `scheduling_engine` are
both explicit that they use K-Means + rules rather than a predictive
model, because there's no ground-truth label to predict for "what
adjustment should this person get" or "who should take this shift" —
those are judgment calls, not classification tasks. A fatigue *report*,
in contrast, **is** a ground-truth label that already happened in the
past — which makes "predict the next one before it happens" a genuine
supervised ML task. Random Forest was chosen over gradient boosting for
this dataset size (~6k reports) because it needs less tuning to avoid
overfitting, over a simpler linear model because the fatigue processes
interact non-linearly (e.g. sleep debt matters far more when a duty
also lands in the WOCL window than either alone), and its
`feature_importances_` keep predictions inspectable rather than a black
box — important for a tool that has to justify flagging someone.

**Leak-safety and evaluation methodology are the load-bearing parts
here**, not the model choice. Every feature for a report at time *T* is
computed strictly from data before *T* (the fatigue engine's own
`as_of` filtering already guarantees this — see
`fatigue_engine/sleep_homeostat.py`), and the only self-report signal
used is the crew member's own *previous* report, not the current one.
Evaluation splits chronologically — train on the earlier ~80% of
reports by date, test on the later ~20% — rather than a random
shuffle-split, so the reported accuracy reflects generalizing to
reports that hadn't happened yet when the model trained, matching how
it would actually be used.

A variance decomposition of the raw `samn_perelli_score` found 88% of
its variance is within-person (day-to-day), not between-person (stable
traits) — meaning most of what's worth predicting really is time-
varying. Even so, most individual duty/sleep/physiological features
correlate weakly (<0.08) with the label; `fatigue_sensitivity` (a
static per-crew trait) is the strongest single predictor at 0.33. If
you extend this engine, that gap is worth keeping in mind: it's likely
a ceiling on how much a features-only model can explain from this
dataset, not a bug in the code.

```bash
python train_risk_model.py                      # trains + evaluates + saves the model
python run_risk_prediction.py --crew-id 101      # predict risk "now"
python run_risk_prediction.py --crew-id 101 --as-of "2026-07-10 05:30:00"
python run_risk_prediction.py --all --out risk_predictions.json
python run_risk_prediction.py --crew-id 101 --shifts data/open_shifts.csv
```

Or from Python:

```python
from merge_data import load_merged_data
from risk_prediction_engine import train_and_evaluate, predict_future_risk
from risk_prediction_engine.predictor import save_model, load_model

data = load_merged_data("data")
result = train_and_evaluate(data)     # TrainingResult: metrics + fitted pipeline
save_model(result.pipeline)

pipeline = load_model()
prediction = predict_future_risk(data, crew_id=101, as_of="2026-07-10 05:30:00", pipeline=pipeline)
```

The last usage — scoring every shift in an `open_shifts.csv` batch for
a given crew member — is meant as a pre-assignment check layered on top
of (not instead of) `scheduling_engine`'s hard eligibility rules: a
shift that legally clears eligibility can still be a bad idea if the
classifier flags High-Risk for that person at that shift's report time.

### Optional: plain-English briefings via the Groq API (`--explain`)

`risk_prediction_engine/explain.py` adds one optional integration point
for an LLM — **not** for the prediction itself (that stays with the
trained Random Forest, which is the right tool for a task with 6,085
labeled historical examples), but for translating its already-decided
output into a short briefing a scheduler can read in a few seconds
instead of parsing JSON.

```bash
pip install groq
export GROQ_API_KEY=your-key-here        # from https://console.groq.com

python run_risk_prediction.py --crew-id 101 --explain
```

This adds an `"explanation"` field to each prediction, e.g.:

> *"Crew 101 is Low-Risk for this shift, driven mainly by strong recent
> sleep (nearly 8 hours/night) and a healthy fatigue score — their
> slightly elevated fatigue sensitivity isn't enough to outweigh that."*

If `groq` isn't installed or `GROQ_API_KEY` isn't set, `--explain` fails
gracefully with a one-line message and the JSON output still prints
normally — it's additive, never required. `--explain-model` can override
the default model (`llama-3.3-70b-versatile`); check
[console.groq.com/docs/models](https://console.groq.com/docs/models) for
what's currently hosted, since Groq rotates its available models more
often than some other providers.
