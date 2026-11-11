"""Explainable AI package for email spam detection.

Exposes:
- predict(text, model_name)
- explain(text, model_name, method, top_k)
"""
from __future__ import annotations

from typing import Any, Dict

from .local import explain_instance
from .utils import (
    CANONICAL_NAMES,
    get_prediction_and_prob,
    load_model,
    normalize_model_name,
)

__all__ = ["predict", "explain", "CANONICAL_NAMES"]


def predict(text: str, model_name: str = "logistic_regression") -> Dict[str, Any]:
    """Predict label and spam probability for input email text.

    Args:
        text: Raw email subject + body text.
        model_name: Name of model (e.g. 'logistic_regression', 'linear_svm',
            'naive_bayes', 'complement_nb', 'random_forest').

    Returns:
        Dict with keys: model, model_tag, label (0 or 1), label_name ('ham' or 'spam'),
        spam_probability (float between 0.0 and 1.0).
    """
    if not text or not text.strip():
        raise ValueError("Input email text is empty.")

    model_tag = normalize_model_name(model_name)
    pipeline = load_model(model_tag)
    label, prob = get_prediction_and_prob(pipeline, text)

    return {
        "model": CANONICAL_NAMES[model_tag],
        "model_tag": model_tag,
        "label": label,
        "label_name": "spam" if label == 1 else "ham",
        "spam_probability": prob,
    }


def explain(
    text: str,
    model_name: str = "logistic_regression",
    method: str = "auto",
    top_k: int = 15,
) -> Dict[str, Any]:
    """Generate local feature-attribution explanations for input email text.

    Args:
        text: Raw email subject + body text.
        model_name: Model identifier.
        method: 'auto' (exact for linear/NB, lime for RF), 'exact', or 'lime'.
        top_k: Maximum number of contributing tokens to return.

    Returns:
        Dict containing model, method, label, spam_probability, base_value,
        and sorted list of tokens with signed weights.
    """
    return explain_instance(
        text=text,
        model_name=model_name,
        method=method,
        top_k=top_k,
    )
