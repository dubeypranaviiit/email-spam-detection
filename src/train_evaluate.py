"""Tune and compare email classifiers without using the test set for tuning."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/emails.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def model_searches():
    """Small grids selected to keep the beginner project runnable locally."""
    shared_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )
    models = {
        "Multinomial Naive Bayes": (
            MultinomialNB(),
            {"classifier__alpha": [0.5, 1.0]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1500, solver="liblinear", random_state=SEED),
            {"classifier__C": [0.5, 2.0], "classifier__class_weight": [None, "balanced"]},
        ),
        "Linear SVM": (
            LinearSVC(random_state=SEED),
            {"classifier__C": [0.5, 2.0], "classifier__class_weight": [None, "balanced"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=120, max_features="sqrt", random_state=SEED, n_jobs=1),
            {"classifier__class_weight": [None, "balanced"]},
        ),
    }
    return shared_vectorizer, models


def near_duplicate_groups(row_count: int) -> tuple[list[int], int]:
    """Make connected-component groups from the preparation similarity audit."""
    audit_path = DATA.parent / "near_duplicate_audit.csv"
    if not audit_path.exists():
        return list(range(row_count)), row_count
    pairs = pd.read_csv(audit_path)
    if pairs.empty:
        return list(range(row_count)), row_count
    left = pairs[["index_a", "index_b"]].to_numpy(dtype=int)
    edges = pd.DataFrame({"a": left[:, 0], "b": left[:, 1]})
    reverse = edges.rename(columns={"a": "b", "b": "a"})
    edges = pd.concat([edges, reverse], ignore_index=True)
    adjacency = coo_matrix(
        ([1] * len(edges), (edges.a.to_numpy(), edges.b.to_numpy())),
        shape=(row_count, row_count),
    ).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    return labels.tolist(), int(count)


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} missing. Run python src/prepare_data.py first.")

    df = pd.read_csv(DATA).fillna("").drop_duplicates("content_hash").reset_index(drop=True)
    X = df.text.astype(str)
    X_without_html = df.text_without_html_fallback.astype(str)
    y = df.label.astype(int)
    hashes = df.content_hash.astype(str)
    groups_list, group_count = near_duplicate_groups(len(df))
    groups = pd.Series(groups_list, index=df.index)
    holdout_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    split_indices = list(holdout_splitter.split(X, y, groups=groups))
    # Use one fold as a reproducible 20% holdout. Similarity groups stay whole.
    train_indices, test_indices = split_indices[0]
    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    hash_train = hashes.iloc[train_indices]
    hash_test = hashes.iloc[test_indices]
    groups_train = groups.iloc[train_indices].reset_index(drop=True)
    groups_test = groups.iloc[test_indices].reset_index(drop=True)

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    vectorizer, configs = model_searches()
    rows = []
    details = {
        "seed": SEED,
        "test_size": float(len(X_test) / len(X)),
        "requested_test_size": TEST_SIZE,
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "cv": {"strategy": "StratifiedGroupKFold", "folds": CV_FOLDS, "shuffle": True, "seed": SEED,
               "selection_metric": "spam F1", "performed_on": "training split only",
               "groups": "connected components of >=0.95 character-TFIDF similarity pairs"},
        "class_distribution": {"ham": int((y == 0).sum()), "spam": int((y == 1).sum())},
        "models": {},
    }
    all_cv_rows = []
    exact_overlap = sorted(set(hash_train) & set(hash_test))

    for name, (classifier, param_grid) in configs.items():
        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ])
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=cv,
            refit=True,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        print(f"Tuning {name} with 5-fold CV...", flush=True)
        search.fit(X_train, y_train, groups=groups_train)
        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
        }
        rows.append({"model": name, **metrics, "cv_best_f1": float(search.best_score_)})
        tag = {
            "Multinomial Naive Bayes": "naive_bayes",
            "Logistic Regression": "logistic_regression",
            "Linear SVM": "linear_svm",
            "Random Forest": "random_forest",
        }[name]
        joblib.dump(best_model, MODELS / f"{tag}.joblib")

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        class_report = classification_report(
            y_test, predictions, target_names=["ham", "spam"], output_dict=True, zero_division=0
        )
        details["models"][name] = {
            **metrics,
            "cv_best_spam_f1": float(search.best_score_),
            "cv_best_parameters": search.best_params_,
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
        }

        error_rows = pd.DataFrame({
            "text": X_test.to_numpy(),
            "true_label": y_test.to_numpy(),
            "predicted_label": predictions,
        })
        error_rows = error_rows[error_rows.true_label != error_rows.predicted_label]
        error_rows.to_csv(REPORTS / f"{tag}_errors.csv", index=False, encoding="utf-8")

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{name} confusion matrix")
        plt.tight_layout()
        plt.savefig(REPORTS / f"{tag}_confusion_matrix.png", dpi=160)
        plt.close()

        cv_results = pd.DataFrame(search.cv_results_)
        for _, result in cv_results.iterrows():
            all_cv_rows.append({
                "model": name,
                "parameters": json.dumps(result["params"], sort_keys=True),
                "mean_cv_spam_f1": float(result["mean_test_score"]),
                "std_cv_spam_f1": float(result["std_test_score"]),
                "rank": int(result["rank_test_score"]),
            })

    # Controlled extraction ablation: same grouped holdout, algorithms and
    # training-only CV grids, with HTML-only fallback text removed.
    ablation_rows = []
    for name in ["Multinomial Naive Bayes", "Logistic Regression"]:
        classifier, param_grid = configs[name]
        ablation_search = GridSearchCV(
            estimator=Pipeline([("tfidf", clone(vectorizer)), ("classifier", clone(classifier))]),
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=cv,
            refit=True,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        X_without_html_train = X_without_html.iloc[train_indices].reset_index(drop=True)
        X_without_html_test = X_without_html.iloc[test_indices].reset_index(drop=True)
        print(f"Running no-HTML ablation for {name}...", flush=True)
        ablation_search.fit(X_without_html_train, y_train, groups=groups_train)
        ablation_predictions = ablation_search.best_estimator_.predict(X_without_html_test)
        ablation_metrics = {
            "accuracy": float(accuracy_score(y_test, ablation_predictions)),
            "precision": float(precision_score(y_test, ablation_predictions, zero_division=0)),
            "recall": float(recall_score(y_test, ablation_predictions, zero_division=0)),
            "f1": float(f1_score(y_test, ablation_predictions, zero_division=0)),
        }
        full_text_metrics = {key: details["models"][name][key] for key in ["accuracy", "precision", "recall", "f1"]}
        ablation_rows.extend([
            {"model": name, "body_representation": "plain plus HTML fallback", **full_text_metrics,
             "cv_best_f1": details["models"][name]["cv_best_spam_f1"]},
            {"model": name, "body_representation": "plain text only (HTML fallback removed)", **ablation_metrics,
             "cv_best_f1": float(ablation_search.best_score_)},
        ])
        details["html_body_ablation"] = details.get("html_body_ablation", {})
        details["html_body_ablation"][name] = {
            "with_html_fallback": {**full_text_metrics, "cv_best_spam_f1": details["models"][name]["cv_best_spam_f1"]},
            "plain_text_only": {**ablation_metrics, "cv_best_spam_f1": float(ablation_search.best_score_),
                                "cv_best_parameters": ablation_search.best_params_},
        }
    pd.DataFrame(ablation_rows).to_csv(REPORTS / "html_ablation.csv", index=False)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    pd.DataFrame(all_cv_rows).to_csv(REPORTS / "cv_grid_results.csv", index=False)
    details["duplicate_audit"] = {
        "cross_split_exact_hash_overlap": len(exact_overlap),
        "cross_split_exact_hash_overlap_sample": exact_overlap[:10],
    }
    prep_file = DATA.parent / "preparation.json"
    if prep_file.exists():
        prep = json.loads(prep_file.read_text(encoding="utf-8"))
        details["duplicate_audit"]["exact_duplicates_removed_during_preparation"] = prep["exact_duplicates_removed"]
    near_file = DATA.parent / "near_duplicate_audit.csv"
    if near_file.exists():
        near = pd.read_csv(near_file)
        train_ids = set(train_indices.tolist())
        test_ids = set(test_indices.tolist())
        cross = near[((near.index_a.isin(train_ids)) & (near.index_b.isin(test_ids))) |
                     ((near.index_a.isin(test_ids)) & (near.index_b.isin(train_ids)))]
        details["duplicate_audit"].update({
            "near_duplicate_heuristic": "character TF-IDF cosine similarity >= 0.95 using 4-5 character ngrams",
            "near_duplicate_candidate_pairs": int(len(near)),
            "near_duplicate_pairs_crossing_train_test": int(len(cross)),
            "near_duplicate_pairs_crossing_train_test_with_different_labels": int((cross.label_a != cross.label_b).sum()),
            "near_duplicate_connected_components": group_count,
            "near_duplicate_groups_crossing_train_test": int(len(set(groups_train) & set(groups_test))),
        })

    (REPORTS / "metrics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    comparison.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot(
        kind="bar", ylim=(0, 1), figsize=(11, 5), rot=15
    )
    plt.ylabel("Score")
    plt.title("SpamAssassin email classification comparison")
    plt.tight_layout()
    plt.savefig(REPORTS / "model_comparison.png", dpi=160)
    plt.close()

    print(f"Train={len(X_train)}, test={len(X_test)}; class counts={details['class_distribution']}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Exact hash overlap between train/test: {len(exact_overlap)}")
    print(f"Artifacts saved in {REPORTS}")


if __name__ == "__main__":
    main()
