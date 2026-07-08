"""
explain.py
==========
Turns a structured `RiskPrediction` (see `predictor.py`) into a short,
plain-English briefing using an LLM (Groq) — the one place in this
project where an AI API makes sense to add.

This is deliberately NOT used for the prediction itself. The risk tier
(Low-Risk / Elevated / High-Risk) is decided entirely by the trained
Random Forest in `model.py` — a statistical classifier trained on 6,085
historical fatigue reports is the right tool for that job, and an LLM
has no access to that training signal. This module only translates the
classifier's already-decided output into language a scheduler can read
at a glance, faster than parsing raw JSON.

Setup
-----
    pip install groq
    export GROQ_API_KEY=your-key-here   # https://console.groq.com

If GROQ_API_KEY isn't set, `explain_prediction` raises a clear error
rather than failing on a confusing network exception.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = (
    "You are briefing an airline crew scheduler on a fatigue-risk prediction "
    "produced by a machine learning model. Write 2-3 plain-English sentences: "
    "state the predicted risk tier, then explain what's driving it using the "
    "top risk factors provided. Be concrete and specific to the numbers given. "
    "No jargon, no bullet points, no headers - just a short briefing a busy "
    "scheduler can read in a few seconds."
)


def _client():
    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError(
            "The `groq` package is required for explain_prediction(). "
            "Install it with: pip install groq"
        ) from e

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a key at https://console.groq.com "
            "and run: export GROQ_API_KEY=your-key-here"
        )
    return Groq(api_key=api_key)


def _format_prediction(prediction: dict) -> str:
    """Render a RiskPrediction dict as compact text for the prompt."""
    lines = [
        f"Crew ID: {prediction['crew_id']}",
        f"As of: {prediction['as_of']}",
        f"Predicted risk tier: {prediction['predicted_risk']}",
        "Tier probabilities: "
        + ", ".join(f"{k} {v:.0%}" for k, v in prediction["risk_probabilities"].items()),
    ]
    factors = prediction.get("top_risk_factors") or []
    if factors:
        lines.append(
            "Top contributing factors: "
            + ", ".join(f"{name}={value}" for name, value in factors)
        )
    return "\n".join(lines)


def explain_prediction(prediction: dict, model: str = DEFAULT_MODEL) -> str:
    """
    Turn a structured RiskPrediction (as returned by
    `risk_prediction_engine.predictor.predict_future_risk`, via
    `vars(prediction)`) into a short natural-language briefing.

    Parameters
    ----------
    prediction: dict with at least `crew_id`, `as_of`, `predicted_risk`,
        `risk_probabilities`, and optionally `top_risk_factors` - i.e.
        exactly the shape `run_risk_prediction.py` already prints as JSON.
    model: Groq-hosted model id. Check https://console.groq.com/docs/models
        for what's currently available - hosted models rotate more often
        on Groq than on some other providers.
    """
    client = _client()
    completion = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _format_prediction(prediction)},
        ],
    )
    return completion.choices[0].message.content.strip()


def explain_predictions(predictions: list[dict], model: str = DEFAULT_MODEL) -> list[str]:
    """Explain a batch of predictions (e.g. from --all or --shifts).
    Makes one API call per prediction - fine for a handful, but see the
    module docstring's note on latency/cost before looping over an
    entire roster."""
    return [explain_prediction(p, model=model) for p in predictions]
