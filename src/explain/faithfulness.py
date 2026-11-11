"""Faithfulness and explanation agreement evaluation."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .local import explain_exact_linear, explain_exact_naive_bayes, explain_lime
from .utils import (
    CANONICAL_NAMES,
    EXPLAIN_DIR,
    get_prediction_and_prob,
    load_holdout_split,
    load_model,
    normalize_model_name,
)


def delete_tokens_from_text(text: str, tokens: List[str]) -> str:
    """Remove occurrences of specified unigrams and bigrams from raw text."""
    modified = text
    # Remove bigrams before unigrams to avoid partial matching
    for t in sorted(tokens, key=lambda x: len(x.split()), reverse=True):
        words = t.strip().split()
        if len(words) == 2:
            pattern = r"\b" + re.escape(words[0]) + r"\s+" + re.escape(words[1]) + r"\b"
        elif len(words) == 1:
            pattern = r"\b" + re.escape(words[0]) + r"\b"
        else:
            continue
        modified = re.sub(pattern, " ", modified, flags=re.IGNORECASE)
    return modified


def get_token_attributions(
    pipeline: Any,
    text: str,
    model_tag: str,
    method: str = "exact",
) -> List[Tuple[str, float]]:
    """Return list of (token, signed_weight) sorted by positive contribution to spam."""
    if method == "lime" or model_tag == "random_forest":
        exp = explain_lime(pipeline, text, model_tag, top_k=50, num_samples=200)
        return [(t["token"], t["weight"]) for t in exp["tokens"]]

    if model_tag in ("logistic_regression", "linear_svm"):
        exp = explain_exact_linear(pipeline, text, model_tag, top_k=500)
        return [(t["token"], t["weight"]) for t in exp["tokens"]]

    if model_tag in ("naive_bayes", "complement_nb"):
        exp = explain_exact_naive_bayes(pipeline, text, model_tag, top_k=500)
        return [(t["token"], t["weight"]) for t in exp["tokens"]]

    raise ValueError(f"Unknown configuration: {model_tag}, {method}")


def run_deletion_test_for_model(
    model_tag: str,
    test_df: pd.DataFrame,
    k_values: List[int] = [1, 3, 5, 10],
    num_random_repeats: int = 20,
    seed: int = 42,
    max_spam_samples: int = 50,
) -> Dict[str, Any]:
    """Run deletion test on correctly classified test spam emails.

    Measures probability drop when deleting top-k spam-indicating tokens versus
    a baseline of deleting k random tokens from the same email (repeated 20 times).
    """
    tag = normalize_model_name(model_tag)
    pipeline = load_model(tag)

    # Filter correctly classified test spam emails
    spam_df = test_df[test_df.label == 1].copy().reset_index(drop=True)
    preds = pipeline.predict(spam_df.text)
    correct_spam = spam_df[preds == 1].reset_index(drop=True)

    # Deterministic fixed subset if corpus is larger
    if len(correct_spam) > max_spam_samples:
        correct_spam = correct_spam.sample(max_spam_samples, random_state=seed).reset_index(drop=True)

    # Method to use: exact for linear/NB, lime for RF
    method = "lime" if tag == "random_forest" else "exact"

    results_by_k: Dict[int, Dict[str, float]] = {}

    for k in k_values:
        top_k_drops: List[float] = []
        rand_mean_drops: List[float] = []
        rand_std_drops: List[float] = []
        top_beat_rand_count = 0

        for i, row in correct_spam.iterrows():
            email_text = str(row["text"])
            _, p_orig = get_prediction_and_prob(pipeline, email_text)

            attributions = get_token_attributions(pipeline, email_text, tag, method=method)
            # Filter positive spam-indicating tokens
            spam_tokens = [token for token, weight in attributions if weight > 0]
            all_active_tokens = [token for token, _ in attributions]

            if not spam_tokens or not all_active_tokens:
                continue

            # 1. Top-k deletion
            tokens_to_del = spam_tokens[:k]
            perturbed_text = delete_tokens_from_text(email_text, tokens_to_del)
            _, p_top_k = get_prediction_and_prob(pipeline, perturbed_text)
            drop_top_k = float(p_orig - p_top_k)
            top_k_drops.append(drop_top_k)

            # 2. Random baseline (20 repeats from active tokens in this same email)
            rand_drops_for_email: List[float] = []
            rng = random.Random(seed + i * 1000 + k * 10)
            sample_k = min(k, len(all_active_tokens))

            for r in range(num_random_repeats):
                rand_tokens = rng.sample(all_active_tokens, sample_k)
                rand_perturbed = delete_tokens_from_text(email_text, rand_tokens)
                _, p_rand = get_prediction_and_prob(pipeline, rand_perturbed)
                rand_drops_for_email.append(float(p_orig - p_rand))

            email_rand_mean = float(np.mean(rand_drops_for_email))
            email_rand_std = float(np.std(rand_drops_for_email))
            rand_mean_drops.append(email_rand_mean)
            rand_std_drops.append(email_rand_std)

            if drop_top_k > email_rand_mean:
                top_beat_rand_count += 1

        n_eval = max(1, len(top_k_drops))
        results_by_k[k] = {
            "k": k,
            "mean_drop_top_k": float(np.mean(top_k_drops)) if top_k_drops else 0.0,
            "std_drop_top_k": float(np.std(top_k_drops)) if top_k_drops else 0.0,
            "mean_drop_random": float(np.mean(rand_mean_drops)) if rand_mean_drops else 0.0,
            "std_drop_random": float(np.mean(rand_std_drops)) if rand_std_drops else 0.0,
            "win_rate_vs_random": float(top_beat_rand_count / n_eval),
            "samples_evaluated": n_eval,
        }

    return {
        "model": CANONICAL_NAMES[tag],
        "model_tag": tag,
        "method_used": method,
        "total_test_spam": int(len(spam_df)),
        "correctly_classified_spam": int(len(correct_spam)),
        "deletion_results": results_by_k,
    }


def run_method_agreement_evaluation(
    test_df: pd.DataFrame,
    sample_size: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure rank correlation (Spearman rho) and top-k Jaccard overlap between methods."""
    # Deterministic fixed test sample
    sample = test_df.sample(sample_size, random_state=seed).reset_index(drop=True)

    comparisons: Dict[str, List[Dict[str, float]]] = {
        "logistic_regression_exact_vs_lime": [],
        "linear_svm_exact_vs_lime": [],
        "naive_bayes_exact_vs_lime": [],
    }

    pipe_lr = load_model("logistic_regression")
    pipe_svm = load_model("linear_svm")
    pipe_nb = load_model("naive_bayes")

    for i, row in sample.iterrows():
        text = str(row["text"])
        if len(text.strip()) < 20:
            continue

        # 1. Logistic Regression
        exp_exact_lr = explain_exact_linear(pipe_lr, text, "logistic_regression", top_k=50)
        exp_lime_lr = explain_lime(pipe_lr, text, "logistic_regression", top_k=20, num_samples=200)
        m_lr = _compute_agreement_metrics(exp_exact_lr["tokens"], exp_lime_lr["tokens"])
        if m_lr is not None:
            comparisons["logistic_regression_exact_vs_lime"].append(m_lr)

        # 2. Linear SVM
        exp_exact_svm = explain_exact_linear(pipe_svm, text, "linear_svm", top_k=50)
        exp_lime_svm = explain_lime(pipe_svm, text, "linear_svm", top_k=20, num_samples=200)
        m_svm = _compute_agreement_metrics(exp_exact_svm["tokens"], exp_lime_svm["tokens"])
        if m_svm is not None:
            comparisons["linear_svm_exact_vs_lime"].append(m_svm)

        # 3. Naive Bayes
        exp_exact_nb = explain_exact_naive_bayes(pipe_nb, text, "naive_bayes", top_k=50)
        exp_lime_nb = explain_lime(pipe_nb, text, "naive_bayes", top_k=20, num_samples=200)
        m_nb = _compute_agreement_metrics(exp_exact_nb["tokens"], exp_lime_nb["tokens"])
        if m_nb is not None:
            comparisons["naive_bayes_exact_vs_lime"].append(m_nb)

    summary: Dict[str, Any] = {}
    for comp_name, metric_list in comparisons.items():
        if metric_list:
            rhos = [m["spearman_rho"] for m in metric_list if not np.isnan(m["spearman_rho"])]
            top5s = [m["top_5_jaccard"] for m in metric_list]
            top10s = [m["top_10_jaccard"] for m in metric_list]
            summary[comp_name] = {
                "evaluated_emails": len(metric_list),
                "mean_spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
                "std_spearman_rho": float(np.std(rhos)) if rhos else 0.0,
                "mean_top_5_jaccard": float(np.mean(top5s)) if top5s else 0.0,
                "mean_top_10_jaccard": float(np.mean(top10s)) if top10s else 0.0,
            }

    return summary


def _compute_agreement_metrics(
    tokens_a: List[Dict[str, Any]],
    tokens_b: List[Dict[str, Any]],
) -> Any:
    dict_a = {t["token"].lower(): t["weight"] for t in tokens_a}
    dict_b = {t["token"].lower(): t["weight"] for t in tokens_b}

    common_tokens = sorted(set(dict_a.keys()) & set(dict_b.keys()))
    if len(common_tokens) < 3:
        return None

    vals_a = [dict_a[k] for k in common_tokens]
    vals_b = [dict_b[k] for k in common_tokens]

    rho, _ = spearmanr(vals_a, vals_b)

    top5_a = set(sorted(dict_a, key=dict_a.get, reverse=True)[:5])
    top5_b = set(sorted(dict_b, key=dict_b.get, reverse=True)[:5])
    jaccard_5 = len(top5_a & top5_b) / max(1, len(top5_a | top5_b))

    top10_a = set(sorted(dict_a, key=dict_a.get, reverse=True)[:10])
    top10_b = set(sorted(dict_b, key=dict_b.get, reverse=True)[:10])
    jaccard_10 = len(top10_a & top10_b) / max(1, len(top10_a | top10_b))

    return {
        "spearman_rho": float(rho),
        "top_5_jaccard": float(jaccard_5),
        "top_10_jaccard": float(jaccard_10),
        "common_token_count": len(common_tokens),
    }


def run_faithfulness_suite(output_dir: Path = EXPLAIN_DIR) -> Dict[str, Any]:
    """Execute complete faithfulness and agreement experiment and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df, _, test_indices = load_holdout_split()
    test_df = df.iloc[test_indices].reset_index(drop=True)

    print("Running deletion tests across models...")
    models_to_test = [
        "logistic_regression",
        "linear_svm",
        "complement_nb",
        "naive_bayes",
        "random_forest",
    ]

    deletion_results = {}
    csv_rows = []

    for tag in models_to_test:
        print(f"  Testing deletion fidelity for {CANONICAL_NAMES[tag]}...")
        res = run_deletion_test_for_model(tag, test_df, k_values=[1, 3, 5, 10], num_random_repeats=20)
        deletion_results[tag] = res

        for k, k_data in res["deletion_results"].items():
            csv_rows.append({
                "model": CANONICAL_NAMES[tag],
                "model_tag": tag,
                "method": res["method_used"],
                "k": k,
                "mean_drop_top_k": k_data["mean_drop_top_k"],
                "std_drop_top_k": k_data["std_drop_top_k"],
                "mean_drop_random": k_data["mean_drop_random"],
                "std_drop_random": k_data["std_drop_random"],
                "win_rate_vs_random": k_data["win_rate_vs_random"],
            })

    print("Running explanation agreement evaluation (Exact vs LIME)...")
    agreement_results = run_method_agreement_evaluation(test_df, sample_size=30, seed=42)

    full_output = {
        "deletion_tests": deletion_results,
        "method_agreement": agreement_results,
        "experimental_notes": (
            "Deletion test uses correctly classified test spam emails only, deleting top-k "
            "tokens from actual raw text. Random baseline uses 20 repeats per email from active "
            "vocabulary tokens. For linear models, the deletion test is partly circular because "
            "weights are exact closed-form coefficients; hence the most informative comparisons "
            "are LIME vs exact and Random Forest vs linear models."
        ),
    }

    json_path = output_dir / "faithfulness.json"
    csv_path = output_dir / "faithfulness.csv"

    json_path.write_text(json.dumps(full_output, indent=2), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"Faithfulness results saved to {json_path} and {csv_path}")
    return full_output
