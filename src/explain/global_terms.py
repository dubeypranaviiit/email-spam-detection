"""Global feature attribution: Top 25 spam- and ham-indicating terms per model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .utils import (
    CANONICAL_NAMES,
    EXPLAIN_DIR,
    load_holdout_split,
    load_model,
    normalize_model_name,
)


def compute_global_terms(model_tag: str, top_k: int = 25) -> Dict[str, Any]:
    """Compute top-k spam-indicating and ham-indicating terms for a given model."""
    tag = normalize_model_name(model_tag)
    pipeline = load_model(tag)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    feature_names = tfidf.get_feature_names_out()

    if tag in ("logistic_regression", "linear_svm"):
        weights = clf.coef_[0]
        method_desc = "linear_coefficients"
        top_spam_indices = np.argsort(weights)[-top_k:][::-1]
        top_ham_indices = np.argsort(weights)[:top_k]

        top_spam = [
            {"term": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_spam_indices
        ]
        top_ham = [
            {"term": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_ham_indices
        ]

    elif tag in ("naive_bayes", "complement_nb"):
        # Log-odds difference: feature_log_prob_[1] - feature_log_prob_[0]
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        method_desc = "log_likelihood_ratio"
        top_spam_indices = np.argsort(weights)[-top_k:][::-1]
        top_ham_indices = np.argsort(weights)[:top_k]

        top_spam = [
            {"term": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_spam_indices
        ]
        top_ham = [
            {"term": str(feature_names[i]), "weight": float(weights[i])}
            for i in top_ham_indices
        ]

    elif tag == "random_forest":
        # Random Forest MDI importances signed by empirical class mean difference
        importances = clf.feature_importances_
        method_desc = "mean_decrease_in_impurity_signed_by_class_frequency"

        df, train_idx, _ = load_holdout_split()
        train_df = df.iloc[train_idx]
        X_train = tfidf.transform(train_df.text)
        y_train = train_df.label.to_numpy()

        spam_mask = (y_train == 1)
        ham_mask = (y_train == 0)

        mean_spam = np.asarray(X_train[spam_mask].mean(axis=0)).ravel()
        mean_ham = np.asarray(X_train[ham_mask].mean(axis=0)).ravel()
        diff = mean_spam - mean_ham

        spam_candidate_indices = np.where(diff > 0)[0]
        ham_candidate_indices = np.where(diff < 0)[0]

        top_spam_indices = spam_candidate_indices[np.argsort(importances[spam_candidate_indices])[-top_k:][::-1]]
        top_ham_indices = ham_candidate_indices[np.argsort(importances[ham_candidate_indices])[-top_k:][::-1]]

        top_spam = [
            {"term": str(feature_names[i]), "weight": float(importances[i])}
            for i in top_spam_indices
        ]
        top_ham = [
            {"term": str(feature_names[i]), "weight": -float(importances[i])}
            for i in top_ham_indices
        ]

    else:
        raise ValueError(f"Unsupported model tag {tag}")

    return {
        "model": CANONICAL_NAMES[tag],
        "model_tag": tag,
        "method": method_desc,
        "top_k": top_k,
        "top_spam_terms": top_spam,
        "top_ham_terms": top_ham,
    }


def export_all_global_terms(output_dir: Path = EXPLAIN_DIR, top_k: int = 25) -> Dict[str, Path]:
    """Generate and write global terms JSON files for all five models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    for tag in CANONICAL_NAMES:
        data = compute_global_terms(tag, top_k=top_k)
        target_path = output_dir / f"global_terms_{tag}.json"
        target_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        results[tag] = target_path

    return results
