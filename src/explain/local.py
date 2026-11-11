"""Local explanation methods: Exact closed-form linear/log-odds and LIME."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from lime.lime_text import LimeTextExplainer

from .utils import (
    CANONICAL_NAMES,
    get_prediction_and_prob,
    load_model,
    normalize_model_name,
)


def explain_exact_linear(
    pipeline: Any,
    text: str,
    model_tag: str,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Exact, faithful contribution of each token for Logistic Regression and Linear SVM.

    Contribution of token j is: weight_j = coefficient_j * tfidf_j.
    This is an exact closed-form decomposition of the logit/decision score.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    feature_names = tfidf.get_feature_names_out()

    X = tfidf.transform([text])
    intercept = float(clf.intercept_[0]) if hasattr(clf, "intercept_") else 0.0
    coef = clf.coef_[0]

    active_indices = X.indices
    active_data = X.data

    tokens: List[Dict[str, Any]] = []
    total_contribution = 0.0

    for idx, val in zip(active_indices, active_data):
        c = float(coef[idx] * val)
        total_contribution += c
        tokens.append({
            "token": str(feature_names[idx]),
            "weight": c,
            "abs_weight": abs(c),
            "sign": "+" if c >= 0 else "-",
            "indication": "spam" if c >= 0 else "ham",
            "tfidf": float(val),
            "coefficient": float(coef[idx]),
        })

    # Sort tokens by magnitude of contribution
    tokens.sort(key=lambda item: abs(item["weight"]), reverse=True)

    score = intercept + total_contribution
    prob = float(1.0 / (1.0 + np.exp(-score)))
    label = int(prob >= 0.5)

    return {
        "model": CANONICAL_NAMES[model_tag],
        "model_tag": model_tag,
        "method": "exact_linear",
        "label": label,
        "label_name": "spam" if label == 1 else "ham",
        "spam_probability": prob,
        "base_value": intercept,
        "score": score,
        "tokens": tokens[:top_k],
        "all_tokens_count": len(tokens),
    }


def explain_exact_naive_bayes(
    pipeline: Any,
    text: str,
    model_tag: str,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Exact per-token log-probability contribution for Multinomial & Complement NB.

    For Multinomial NB:
      w_j = feature_log_prob_[1, j] - feature_log_prob_[0, j]
      base = class_log_prior_[1] - class_log_prior_[0]
      contribution_j = x_j * w_j

    For Complement NB:
      w_j = feature_log_prob_[1, j] - feature_log_prob_[0, j]
      base = 0.0 (unnormalized uniform prior in scikit-learn when norm=False)
      contribution_j = x_j * w_j

    In both cases, sigmoid(base + sum(contribution_j)) == predict_proba([text])[0, 1].
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    feature_names = tfidf.get_feature_names_out()

    X = tfidf.transform([text])
    active_indices = X.indices
    active_data = X.data

    if model_tag == "naive_bayes":
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        base_value = float(clf.class_log_prior_[1] - clf.class_log_prior_[0])
    elif model_tag == "complement_nb":
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        base_value = 0.0
    else:
        raise ValueError(f"Model {model_tag} is not a supported Naive Bayes variant.")

    tokens: List[Dict[str, Any]] = []
    total_contribution = 0.0

    for idx, val in zip(active_indices, active_data):
        c = float(w[idx] * val)
        total_contribution += c
        tokens.append({
            "token": str(feature_names[idx]),
            "weight": c,
            "abs_weight": abs(c),
            "sign": "+" if c >= 0 else "-",
            "indication": "spam" if c >= 0 else "ham",
            "tfidf": float(val),
            "log_odds_weight": float(w[idx]),
        })

    tokens.sort(key=lambda item: abs(item["weight"]), reverse=True)

    score = base_value + total_contribution
    prob = float(1.0 / (1.0 + np.exp(-score)))
    label = int(prob >= 0.5)

    return {
        "model": CANONICAL_NAMES[model_tag],
        "model_tag": model_tag,
        "method": "exact_log_odds",
        "label": label,
        "label_name": "spam" if label == 1 else "ham",
        "spam_probability": prob,
        "base_value": base_value,
        "score": score,
        "tokens": tokens[:top_k],
        "all_tokens_count": len(tokens),
    }


def explain_lime(
    pipeline: Any,
    text: str,
    model_tag: str,
    top_k: int = 15,
    num_samples: int = 250,
) -> Dict[str, Any]:
    """Model-agnostic LIME explanation for text classifiers (Random Forest, linear, etc.)."""
    # Bound num_samples to keep API fast and avoid latency spikes
    capped_samples = max(50, min(num_samples, 400))

    label, prob = get_prediction_and_prob(pipeline, text)

    def predict_fn(texts: List[str]) -> np.ndarray:
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(texts)
        # LinearSVC decision function mapping to probability
        dfunc = pipeline.decision_function(texts)
        p1 = 1.0 / (1.0 + np.exp(-dfunc))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    explainer = LimeTextExplainer(
        class_names=["ham", "spam"],
        bow=True,
        random_state=42,
    )

    exp = explainer.explain_instance(
        text,
        predict_fn,
        num_features=top_k,
        num_samples=capped_samples,
        labels=(1,),
    )

    tokens: List[Dict[str, Any]] = []
    for token, weight in exp.as_list(label=1):
        w_val = float(weight)
        tokens.append({
            "token": str(token),
            "weight": w_val,
            "abs_weight": abs(w_val),
            "sign": "+" if w_val >= 0 else "-",
            "indication": "spam" if w_val >= 0 else "ham",
        })

    # Sort by absolute weight
    tokens.sort(key=lambda item: abs(item["weight"]), reverse=True)

    intercept = float(exp.intercept[1]) if hasattr(exp, "intercept") and 1 in exp.intercept else 0.0

    return {
        "model": CANONICAL_NAMES[model_tag],
        "model_tag": model_tag,
        "method": "lime",
        "label": label,
        "label_name": "spam" if label == 1 else "ham",
        "spam_probability": prob,
        "base_value": intercept,
        "score": intercept + sum(t["weight"] for t in tokens),
        "tokens": tokens[:top_k],
        "all_tokens_count": len(tokens),
    }


def explain_instance(
    text: str,
    model_name: str = "logistic_regression",
    method: str = "auto",
    top_k: int = 15,
    num_samples: int = 250,
) -> Dict[str, Any]:
    """Explain a single email classification using the specified or optimal method.

    Methods:
    - 'auto': exact for linear & NB models, LIME for Random Forest.
    - 'exact': exact closed form (linear or NB). Raises ValueError for Random Forest.
    - 'lime': model-agnostic LIME surrogate explainer.
    """
    if not text or not text.strip():
        raise ValueError("Input email text is empty.")

    model_tag = normalize_model_name(model_name)
    pipeline = load_model(model_tag)

    chosen_method = method.strip().lower()

    if chosen_method == "auto":
        if model_tag in ("logistic_regression", "linear_svm"):
            return explain_exact_linear(pipeline, text, model_tag, top_k=top_k)
        elif model_tag in ("naive_bayes", "complement_nb"):
            return explain_exact_naive_bayes(pipeline, text, model_tag, top_k=top_k)
        elif model_tag == "random_forest":
            return explain_lime(pipeline, text, model_tag, top_k=top_k, num_samples=num_samples)
        else:
            raise ValueError(f"No auto explainer defined for {model_tag}")

    elif chosen_method == "exact":
        if model_tag in ("logistic_regression", "linear_svm"):
            return explain_exact_linear(pipeline, text, model_tag, top_k=top_k)
        elif model_tag in ("naive_bayes", "complement_nb"):
            return explain_exact_naive_bayes(pipeline, text, model_tag, top_k=top_k)
        else:
            raise ValueError(
                f"Exact linear/log-odds explanation is not applicable to {CANONICAL_NAMES[model_tag]}. "
                "Use method='lime' instead."
            )

    elif chosen_method == "lime":
        return explain_lime(pipeline, text, model_tag, top_k=top_k, num_samples=num_samples)

    else:
        raise ValueError(f"Unknown explanation method '{method}'. Valid options: 'auto', 'exact', 'lime'.")
