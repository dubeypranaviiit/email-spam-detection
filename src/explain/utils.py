"""Shared utilities for model loading, split recovery, and inference."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "emails.csv"
AUDIT_PATH = ROOT / "data" / "processed" / "near_duplicate_audit.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
EXPLAIN_DIR = REPORTS_DIR / "explain"
SEED = 42

MODEL_TAGS: dict[str, str] = {
    "logistic regression": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "lr": "logistic_regression",
    "linear svm": "linear_svm",
    "linear_svm": "linear_svm",
    "svm": "linear_svm",
    "multinomial naive bayes": "naive_bayes",
    "naive_bayes": "naive_bayes",
    "nb": "naive_bayes",
    "multinomial_nb": "naive_bayes",
    "complement naive bayes": "complement_nb",
    "complement_nb": "complement_nb",
    "cnb": "complement_nb",
    "random forest": "random_forest",
    "random_forest": "random_forest",
    "rf": "random_forest",
}

CANONICAL_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "naive_bayes": "Multinomial Naive Bayes",
    "complement_nb": "Complement Naive Bayes",
    "random_forest": "Random Forest",
}


def normalize_model_name(name: str) -> str:
    """Normalize model string to canonical tag."""
    cleaned = name.strip().lower()
    if cleaned not in MODEL_TAGS:
        available = list(CANONICAL_NAMES.values())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_TAGS[cleaned]


@functools.lru_cache(maxsize=16)
def load_model(name: str) -> Any:
    """Load a fitted scikit-learn pipeline by model name or tag."""
    tag = normalize_model_name(name)
    path = MODELS_DIR / f"{tag}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model file {path} not found. Please train models first using "
            "python src/prepare_data.py && python src/train_evaluate.py"
        )
    return joblib.load(path)


def load_holdout_split() -> Tuple[pd.DataFrame, list[int], list[int]]:
    """Reproduce the exact grouped stratified holdout split from train_evaluate.py."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} missing. Run python src/prepare_data.py first.")

    df = pd.read_csv(DATA_PATH).fillna("").drop_duplicates("content_hash").reset_index(drop=True)
    if AUDIT_PATH.exists():
        pairs = pd.read_csv(AUDIT_PATH)
        if not pairs.empty:
            e1 = pairs[["index_a", "index_b"]].rename(columns={"index_a": "a", "index_b": "b"})
            e2 = pairs[["index_a", "index_b"]].rename(columns={"index_b": "a", "index_a": "b"})
            edges = pd.concat([e1, e2], ignore_index=True)
            adj = coo_matrix(
                ([1] * len(edges), (edges.a.to_numpy(), edges.b.to_numpy())),
                shape=(len(df), len(df)),
            ).tocsr()
            _, labels = connected_components(adj, directed=False)
            groups = pd.Series(labels, index=df.index)
        else:
            groups = pd.Series(range(len(df)), index=df.index)
    else:
        groups = pd.Series(range(len(df)), index=df.index)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, test_indices = list(splitter.split(df.text, df.label, groups=groups))[0]
    return df, train_indices.tolist(), test_indices.tolist()


def get_prediction_score_and_prob(pipeline: Any, text: str) -> Tuple[int, float, float]:
    """Return label (0 or 1), decision score / log-odds (float), and spam probability (float)."""
    clf = pipeline.named_steps["classifier"]
    tfidf = pipeline.named_steps["tfidf"]
    clf_type = type(clf).__name__

    if hasattr(clf, "decision_function"):
        # Logistic Regression or Linear SVM
        score = float(pipeline.decision_function([text])[0])
        prob = float(1.0 / (1.0 + np.exp(-score)))
        label = int(score > 0)
    elif clf_type == "ComplementNB":
        # Complement NB: jll = X * feature_log_prob_.T (no prior log-odds added)
        X = tfidf.transform([text])
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        score = float(X.dot(w)[0])
        prob = float(pipeline.predict_proba([text])[0, 1])
        label = int(score > 0)
    elif clf_type == "MultinomialNB":
        # Multinomial NB: jll = X * feature_log_prob_.T + class_log_prior_
        X = tfidf.transform([text])
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        b = float(clf.class_log_prior_[1] - clf.class_log_prior_[0])
        score = float(b + X.dot(w)[0])
        prob = float(pipeline.predict_proba([text])[0, 1])
        label = int(score > 0)
    else:
        # Random Forest
        prob = float(pipeline.predict_proba([text])[0, 1])
        p_clipped = min(max(prob, 1e-6), 1.0 - 1e-6)
        score = float(np.log(p_clipped / (1.0 - p_clipped)))
        label = int(prob >= 0.5)

    return label, score, prob


def get_prediction_and_prob(pipeline: Any, text: str) -> Tuple[int, float]:
    """Backward-compatible helper returning label and spam probability."""
    label, _, prob = get_prediction_score_and_prob(pipeline, text)
    return label, prob
