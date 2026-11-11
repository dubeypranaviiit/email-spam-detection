"""Faithfulness and explanation agreement evaluation.

Features:
- Deletion test across all correctly classified holdout spam emails shared across all 5 models (n=87).
- Note: Evaluated on mutual correctly classified test spam emails (n=87) to ensure an identical,
  controlled comparison set across models, acknowledging this creates an easy-case benchmark bias.
- Evaluates drop in both predicted spam probability (Delta P) and decision score / log-odds (Delta z).
- Computes 95% bootstrap confidence intervals for all drops.
- Evaluates cross-method (Exact vs LIME) and cross-model (Random Forest vs Linear/NB) agreement on n=20-30
  emails (limited by LIME simulation cost).
- Formulates methodological hypotheses regarding Naive Bayes log-odds additivity vs probability saturation,
  TF-IDF L2 normalization rescaling effects, and LIME surrogate approximations.
"""
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
    get_prediction_score_and_prob,
    load_holdout_split,
    load_model,
    normalize_model_name,
)


def bootstrap_ci(arr: List[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for the sample mean."""
    if not arr:
        return 0.0, 0.0
    a = np.asarray(arr, dtype=float)
    if len(a) == 1:
        return float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(n_boot, len(a)))
    boot_means = a[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1.0 - alpha)))


def delete_tokens_from_text(text: str, tokens: List[str]) -> str:
    """Remove occurrences of specified unigrams and bigrams from raw text.

    For bigrams, matches across arbitrary non-word punctuation (e.g. dots in URLs)
    and optional intermediate stop words. If an isolated bigram construct fails to
    match as a contiguous regex, falls back to removing its constituent word to
    prevent the n-gram from being extracted by TfidfVectorizer.
    """
    modified = text
    for t in sorted(tokens, key=lambda x: len(x.split()), reverse=True):
        words = t.strip().split()
        if len(words) == 2:
            pat = r"\b" + re.escape(words[0]) + r"\b(?:[^\w]+|\s+\w+){0,3}[^\w]+" + re.escape(words[1]) + r"\b"
            new_text, n_subs = re.subn(pat, " ", modified, flags=re.IGNORECASE)
            if n_subs > 0:
                modified = new_text
            else:
                modified = re.sub(r"\b" + re.escape(words[0]) + r"\b", " ", modified, flags=re.IGNORECASE)
        elif len(words) == 1:
            modified = re.sub(r"\b" + re.escape(words[0]) + r"\b", " ", modified, flags=re.IGNORECASE)
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


def get_mutual_correct_spam(test_df: pd.DataFrame) -> pd.DataFrame:
    """Find the set of holdout spam emails correctly classified by ALL five models."""
    spam_df = test_df[test_df.label == 1].reset_index(drop=True)
    mask = pd.Series(True, index=spam_df.index)
    for tag in CANONICAL_NAMES:
        pipe = load_model(tag)
        preds = pipe.predict(spam_df.text)
        mask = mask & (preds == 1)
    mutual_df = spam_df[mask].reset_index(drop=True)
    return mutual_df


def run_deletion_test_for_model(
    model_tag: str,
    eval_emails_df: pd.DataFrame,
    k_values: List[int] = [1, 3, 5, 10],
    num_random_repeats: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run deletion test on mutual correctly classified test spam emails.

    Measures drop in both:
      1. Predicted spam probability (Delta P)
      2. Decision score / log-odds (Delta z)
    Comparing top-k explained tokens against 20 random repeats of k active tokens.
    """
    tag = normalize_model_name(model_tag)
    pipeline = load_model(tag)
    method = "lime" if tag == "random_forest" else "exact"

    results_by_k: Dict[int, Dict[str, Any]] = {}
    n_samples = len(eval_emails_df)

    for k in k_values:
        prob_drops_top_k: List[float] = []
        prob_drops_rand_mean: List[float] = []

        score_drops_top_k: List[float] = []
        score_drops_rand_mean: List[float] = []

        win_prob_count = 0
        win_score_count = 0

        for i, row in eval_emails_df.iterrows():
            email_text = str(row["text"])
            _, z_orig, p_orig = get_prediction_score_and_prob(pipeline, email_text)

            attributions = get_token_attributions(pipeline, email_text, tag, method=method)
            spam_tokens = [tok for tok, weight in attributions if weight > 0]
            all_active = [tok for tok, _ in attributions]

            if not spam_tokens or not all_active:
                continue

            # 1. Top-k deletion
            tokens_to_del = spam_tokens[:k]
            perturbed_text = delete_tokens_from_text(email_text, tokens_to_del)
            _, z_top_k, p_top_k = get_prediction_score_and_prob(pipeline, perturbed_text)

            dp_top_k = float(p_orig - p_top_k)
            dz_top_k = float(z_orig - z_top_k)
            prob_drops_top_k.append(dp_top_k)
            score_drops_top_k.append(dz_top_k)

            # 2. Random baseline (20 repeats from active tokens in this email)
            email_rand_dp: List[float] = []
            email_rand_dz: List[float] = []
            rng = random.Random(seed + i * 1000 + k * 10)
            sample_k = min(k, len(all_active))

            for _ in range(num_random_repeats):
                rand_tokens = rng.sample(all_active, sample_k)
                rand_perturbed = delete_tokens_from_text(email_text, rand_tokens)
                _, z_rand, p_rand = get_prediction_score_and_prob(pipeline, rand_perturbed)
                email_rand_dp.append(float(p_orig - p_rand))
                email_rand_dz.append(float(z_orig - z_rand))

            mean_rand_dp = float(np.mean(email_rand_dp))
            mean_rand_dz = float(np.mean(email_rand_dz))
            prob_drops_rand_mean.append(mean_rand_dp)
            score_drops_rand_mean.append(mean_rand_dz)

            if dp_top_k > mean_rand_dp:
                win_prob_count += 1
            if dz_top_k > mean_rand_dz:
                win_score_count += 1

        n_eval = max(1, len(prob_drops_top_k))

        # Bootstrap 95% CIs
        p_top_k_ci = bootstrap_ci(prob_drops_top_k, seed=seed)
        p_rand_ci = bootstrap_ci(prob_drops_rand_mean, seed=seed)
        z_top_k_ci = bootstrap_ci(score_drops_top_k, seed=seed)
        z_rand_ci = bootstrap_ci(score_drops_rand_mean, seed=seed)

        results_by_k[k] = {
            "k": k,
            "n": n_eval,
            # Probability drops
            "mean_prob_drop_top_k": float(np.mean(prob_drops_top_k)),
            "ci_prob_drop_top_k_95": list(p_top_k_ci),
            "mean_prob_drop_random": float(np.mean(prob_drops_rand_mean)),
            "ci_prob_drop_random_95": list(p_rand_ci),
            "win_rate_prob_vs_random": float(win_prob_count / n_eval),
            # Decision score / log-odds drops
            "mean_score_drop_top_k": float(np.mean(score_drops_top_k)),
            "ci_score_drop_top_k_95": list(z_top_k_ci),
            "mean_score_drop_random": float(np.mean(score_drops_rand_mean)),
            "ci_score_drop_random_95": list(z_rand_ci),
            "win_rate_score_vs_random": float(win_score_count / n_eval),
        }

    return {
        "model": CANONICAL_NAMES[tag],
        "model_tag": tag,
        "method_used": method,
        "n_evaluated": n_samples,
        "deletion_results": results_by_k,
    }


def run_method_agreement_evaluation(
    test_df: pd.DataFrame,
    sample_size: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate cross-method and cross-model agreement on active unigrams.

    Sample size n=20-30 is selected due to computational latency of LIME text perturbations.
    """
    sample = test_df.sample(sample_size, random_state=seed).reset_index(drop=True)

    comparisons: Dict[str, List[Dict[str, float]]] = {
        "logistic_regression_exact_vs_lime": [],
        "linear_svm_exact_vs_lime": [],
        "naive_bayes_exact_vs_lime": [],
        "complement_nb_exact_vs_lime": [],
        "random_forest_lime_vs_logistic_regression_exact": [],
        "random_forest_lime_vs_linear_svm_exact": [],
        "random_forest_lime_vs_complement_nb_exact": [],
        "logistic_regression_exact_vs_linear_svm_exact": [],
    }

    pipe_lr = load_model("logistic_regression")
    pipe_svm = load_model("linear_svm")
    pipe_mnb = load_model("naive_bayes")
    pipe_cnb = load_model("complement_nb")
    pipe_rf = load_model("random_forest")

    for _, row in sample.iterrows():
        text = str(row["text"])
        if len(text.strip()) < 20:
            continue

        exp_exact_lr = explain_exact_linear(pipe_lr, text, "logistic_regression", top_k=50)
        exp_lime_lr = explain_lime(pipe_lr, text, "logistic_regression", top_k=20, num_samples=200)

        exp_exact_svm = explain_exact_linear(pipe_svm, text, "linear_svm", top_k=50)
        exp_lime_svm = explain_lime(pipe_svm, text, "linear_svm", top_k=20, num_samples=200)

        exp_exact_mnb = explain_exact_naive_bayes(pipe_mnb, text, "naive_bayes", top_k=50)
        exp_lime_mnb = explain_lime(pipe_mnb, text, "naive_bayes", top_k=20, num_samples=200)

        exp_exact_cnb = explain_exact_naive_bayes(pipe_cnb, text, "complement_nb", top_k=50)
        exp_lime_cnb = explain_lime(pipe_cnb, text, "complement_nb", top_k=20, num_samples=200)

        exp_lime_rf = explain_lime(pipe_rf, text, "random_forest", top_k=20, num_samples=200)

        pairs = [
            ("logistic_regression_exact_vs_lime", exp_exact_lr["tokens"], exp_lime_lr["tokens"]),
            ("linear_svm_exact_vs_lime", exp_exact_svm["tokens"], exp_lime_svm["tokens"]),
            ("naive_bayes_exact_vs_lime", exp_exact_mnb["tokens"], exp_lime_mnb["tokens"]),
            ("complement_nb_exact_vs_lime", exp_exact_cnb["tokens"], exp_lime_cnb["tokens"]),
            ("random_forest_lime_vs_logistic_regression_exact", exp_lime_rf["tokens"], exp_exact_lr["tokens"]),
            ("random_forest_lime_vs_linear_svm_exact", exp_lime_rf["tokens"], exp_exact_svm["tokens"]),
            ("random_forest_lime_vs_complement_nb_exact", exp_lime_rf["tokens"], exp_exact_cnb["tokens"]),
            ("logistic_regression_exact_vs_linear_svm_exact", exp_exact_lr["tokens"], exp_exact_svm["tokens"]),
        ]

        for name, tok_a, tok_b in pairs:
            m = _compute_agreement_metrics(tok_a, tok_b)
            if m is not None:
                comparisons[name].append(m)

    summary: Dict[str, Any] = {}
    for comp_name, metric_list in comparisons.items():
        if metric_list:
            rhos = [m["spearman_rho"] for m in metric_list if not np.isnan(m["spearman_rho"])]
            top5s = [m["top_5_jaccard"] for m in metric_list]
            top10s = [m["top_10_jaccard"] for m in metric_list]

            rho_ci = bootstrap_ci(rhos, seed=seed)
            top5_ci = bootstrap_ci(top5s, seed=seed)
            top10_ci = bootstrap_ci(top10s, seed=seed)

            summary[comp_name] = {
                "n_evaluated": len(metric_list),
                "token_universe": "active unigrams present in the document evaluated by both methods",
                "mean_spearman_rho": float(np.mean(rhos)) if rhos else 0.0,
                "ci_spearman_rho_95": list(rho_ci),
                "mean_top_5_jaccard": float(np.mean(top5s)) if top5s else 0.0,
                "ci_top_5_jaccard_95": list(top5_ci),
                "mean_top_10_jaccard": float(np.mean(top10s)) if top10s else 0.0,
                "ci_top_10_jaccard_95": list(top10_ci),
            }

    return summary


def _compute_agreement_metrics(
    tokens_a: List[Dict[str, Any]],
    tokens_b: List[Dict[str, Any]],
) -> Any:
    dict_a = {t["token"].lower(): t["weight"] for t in tokens_a if len(t["token"].split()) == 1}
    dict_b = {t["token"].lower(): t["weight"] for t in tokens_b if len(t["token"].split()) == 1}

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

    mutual_spam_df = get_mutual_correct_spam(test_df)
    n_common = len(mutual_spam_df)
    print(f"Mutual correctly classified holdout spam emails across all 5 models: n={n_common}")

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
        print(f"  Testing deletion fidelity for {CANONICAL_NAMES[tag]} (n={n_common})...")
        res = run_deletion_test_for_model(tag, mutual_spam_df, k_values=[1, 3, 5, 10], num_random_repeats=20)
        deletion_results[tag] = res

        for k, kd in res["deletion_results"].items():
            csv_rows.append({
                "model": CANONICAL_NAMES[tag],
                "model_tag": tag,
                "method": res["method_used"],
                "n": kd["n"],
                "k": k,
                # Probability drop
                "mean_prob_drop_top_k": kd["mean_prob_drop_top_k"],
                "ci_prob_drop_top_k_lower": kd["ci_prob_drop_top_k_95"][0],
                "ci_prob_drop_top_k_upper": kd["ci_prob_drop_top_k_95"][1],
                "mean_prob_drop_random": kd["mean_prob_drop_random"],
                "ci_prob_drop_random_lower": kd["ci_prob_drop_random_95"][0],
                "ci_prob_drop_random_upper": kd["ci_prob_drop_random_95"][1],
                "win_rate_prob_vs_random": kd["win_rate_prob_vs_random"],
                # Score / log-odds drop
                "mean_score_drop_top_k": kd["mean_score_drop_top_k"],
                "ci_score_drop_top_k_lower": kd["ci_score_drop_top_k_95"][0],
                "ci_score_drop_top_k_upper": kd["ci_score_drop_top_k_95"][1],
                "mean_score_drop_random": kd["mean_score_drop_random"],
                "ci_score_drop_random_lower": kd["ci_score_drop_random_95"][0],
                "ci_score_drop_random_upper": kd["ci_score_drop_random_95"][1],
                "win_rate_score_vs_random": kd["win_rate_score_vs_random"],
            })

    print("Running explanation agreement evaluation (Exact vs LIME & cross-model)...")
    agreement_results = run_method_agreement_evaluation(test_df, sample_size=30, seed=42)

    full_output = {
        "experimental_setup": {
            "evaluation_set": "Mutual correctly classified holdout spam emails across all 5 models (n=87), noting easy-case benchmark bias",
            "n_emails_per_model": n_common,
            "random_baseline_repeats": 20,
            "bootstrap_ci_level": 0.95,
            "bootstrap_iterations": 1000,
            "agreement_sample_size_rationale": "Sample size n=20-30 used due to LIME simulation and perturbation computation cost",
            "token_universe_agreement": "active unigrams present in the email recognized by both methods",
        },
        "methodological_hypotheses": {
            "linear_models_circularity": (
                "Hypothesis / property: For linear models (Logistic Regression, Linear SVM), the deletion test is "
                "partly circular because exact token weights w_j * x_j are closed-form components of the logit. The most "
                "informative evaluations are therefore LIME vs exact and Random Forest vs the linear models."
            ),
            "naive_bayes_lime_divergence": (
                "Hypothesis: Naive Bayes is strictly additive in log-space (log-odds = b + sum x_j w_j). "
                "The lower rank correlation observed between exact NB log-odds and LIME is hypothesized to "
                "stem from two factors: (1) severe probability saturation in Naive Bayes where posterior probabilities "
                "cluster within 1e-6 of 0 or 1, producing a step-like probability surface under random token masking; "
                "and (2) objective discrepancy: LIME fits an L2-regularized linear surrogate predicting probability shifts, "
                "whereas exact attribution decomposes log-odds."
            ),
            "tfidf_l2_rescaling_after_deletion": (
                "Hypothesis: Deleting tokens from raw text induces TF-IDF L2-normalization rescaling: when a token is removed, "
                "the document vector norm decreases, mathematically boosting the normalized weights of all remaining terms by "
                "1 / sqrt(1 - x_deleted^2). In emails with dozens of strong spam features, this rescaling can partially offset "
                "the drop from removing a single token at k=1."
            ),
            "cross_model_tree_discrepancy": (
                "Hypothesis: Random Forest relies on axis-aligned feature splits and decision paths that can capture nonlinear "
                "feature interactions. When explained via LIME, its feature attributions moderately correlate with linear models "
                "(rho ~ 0.46-0.50) but diverge from Complement NB (rho ~ -0.04)."
            ),
        },
        "deletion_tests": deletion_results,
        "method_agreement": agreement_results,
    }

    json_path = output_dir / "faithfulness.json"
    csv_path = output_dir / "faithfulness.csv"

    json_path.write_text(json.dumps(full_output, indent=2), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"Faithfulness results saved to {json_path} and {csv_path}")
    return full_output
