"""Global feature attribution: Top 25 spam- and ham-indicating terms per model.

For Linear Models (Logistic Regression, Linear SVM):
  Terms are ranked by fitted model coefficients w_j in the decision boundary.
  Positive weights push toward Spam; negative weights push toward Ham.

For Naive Bayes (Multinomial NB, Complement NB):
  Terms are ranked by log-likelihood ratio weights:
  w_j = feature_log_prob_[1, j] - feature_log_prob_[0, j].
  Positive weights push toward Spam; negative weights push toward Ham.

For Random Forest:
  Standard tree Mean Decrease in Impurity (MDI / Gini importance) is unsigned and non-negative (I_j >= 0).
  To provide signed polarities, each feature's importance is signed using the empirical difference
  in class means on the training set:
    Delta mu_j = E[X_j | Spam] - E[X_j | Ham] = (1 / N_spam) sum_{i in Spam} X_{i, j} - (1 / N_ham) sum_{i in Ham} X_{i, j}
  A feature is designated as spam-indicating if Delta mu_j > 0 (signed +I_j) and ham-indicating if Delta mu_j < 0 (signed -I_j).
"""
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
        method_desc = "Fitted linear coefficients w_j from the decision boundary."
        formula = "w_j = classifier.coef_[0, j]"
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
        method_desc = "Log-likelihood ratio weight w_j = log P(w_j | spam) - log P(w_j | ham)."
        formula = "w_j = feature_log_prob_[1, j] - feature_log_prob_[0, j]"
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
        importances = clf.feature_importances_
        method_desc = (
            "Random Forest MDI (Gini importance) signed by empirical training class mean difference: "
            "Delta mu_j = mean(TF-IDF_j | spam) - mean(TF-IDF_j | ham). Terms with Delta mu_j > 0 are "
            "spam-indicating (+MDI); terms with Delta mu_j < 0 are ham-indicating (-MDI)."
        )
        formula = "signed_weight_j = MDI_j * sign(mean(X_{spam, j}) - mean(X_{ham, j}))"

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
        "formula": formula,
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
