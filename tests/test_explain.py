"""Unit tests for the explainable AI module and file integrity."""
import hashlib
from pathlib import Path
import numpy as np
import pytest

from src.explain import predict, explain
from src.explain.faithfulness import delete_tokens_from_text
from src.explain.local import explain_exact_linear, explain_exact_naive_bayes
from src.explain.utils import load_model, load_holdout_split, get_prediction_score_and_prob

ROOT = Path(__file__).resolve().parents[1]
METRICS_JSON_PATH = ROOT / "reports" / "metrics.json"
EXPECTED_METRICS_SHA256 = "260016373A4BE41865F7F676F93ADA1F587D3F1BA9F23A362491CC3C8E9DB719".lower()


def test_metrics_json_integrity_unmodified():
    """Verify that reports/metrics.json has not been altered or overwritten."""
    assert METRICS_JSON_PATH.exists(), "metrics.json must exist"
    actual_hash = hashlib.sha256(METRICS_JSON_PATH.read_bytes()).hexdigest().lower()
    assert actual_hash == EXPECTED_METRICS_SHA256, (
        f"Integrity violation: reports/metrics.json SHA256 changed!\n"
        f"Expected: {EXPECTED_METRICS_SHA256}\n"
        f"Actual:   {actual_hash}"
    )


def test_predict_returns_valid_probabilities():
    """Verify predict returns probabilities in [0, 1] and matching binary label."""
    sample = "Subject: Important meeting tomorrow at 10 AM regarding project status."
    for model_name in ["logistic_regression", "linear_svm", "naive_bayes", "complement_nb", "random_forest"]:
        res = predict(sample, model_name=model_name)
        assert "label" in res
        assert "spam_probability" in res
        assert 0.0 <= res["spam_probability"] <= 1.0
        assert res["label"] in (0, 1)
        assert res["label_name"] in ("ham", "spam")
        assert (res["label"] == 1) == (res["spam_probability"] >= 0.5)


def test_exact_explainers_completeness_to_1e9():
    """Verify intercept + sum(per-token contributions) equals model decision score / log-odds to 1e-9."""
    df, _, test_idx = load_holdout_split()
    test_df = df.iloc[test_idx].reset_index(drop=True)
    sample_emails = test_df.sample(20, random_state=42).text.tolist()

    for model_tag in ["logistic_regression", "linear_svm", "naive_bayes", "complement_nb"]:
        pipe = load_model(model_tag)
        for text in sample_emails:
            _, score_pipe, _ = get_prediction_score_and_prob(pipe, text)

            if model_tag in ("logistic_regression", "linear_svm"):
                exp = explain_exact_linear(pipe, text, model_tag, top_k=100000)
            else:
                exp = explain_exact_naive_bayes(pipe, text, model_tag, top_k=100000)

            score_reconstructed = exp["base_value"] + sum(t["weight"] for t in exp["tokens"])
            diff = abs(score_reconstructed - score_pipe)
            assert diff < 1e-9, (
                f"Completeness violation for {model_tag}: diff={diff:.2e} exceeds 1e-9 limit! "
                f"Reconstructed: {score_reconstructed}, Pipe: {score_pipe}"
            )


def test_exact_linear_math_identity():
    """Verify exact linear attribution satisfies logit = intercept + sum(weights)."""
    sample = "Subject: Urgent offer! Claim your free money and discount today!"
    pipe = load_model("logistic_regression")
    exp = explain_exact_linear(pipe, sample, "logistic_regression", top_k=100)

    intercept = exp["base_value"]
    token_weights_sum = sum(t["weight"] for t in exp["tokens"])
    reconstructed_score = intercept + token_weights_sum
    reconstructed_prob = 1.0 / (1.0 + np.exp(-reconstructed_score))

    pipe_prob = pipe.predict_proba([sample])[0, 1]
    assert np.isclose(reconstructed_prob, pipe_prob, atol=1e-5), (
        f"Math mismatch: reconstructed {reconstructed_prob} vs pipeline {pipe_prob}"
    )


def test_exact_naive_bayes_math_identity():
    """Verify Multinomial NB log-odds decomposition matches predict_proba exactly."""
    sample = "Subject: Special conference invitation for machine learning research."
    pipe = load_model("naive_bayes")
    exp = explain_exact_naive_bayes(pipe, sample, "naive_bayes", top_k=100)

    base_val = exp["base_value"]
    token_weights_sum = sum(t["weight"] for t in exp["tokens"])
    reconstructed_score = base_val + token_weights_sum
    reconstructed_prob = 1.0 / (1.0 + np.exp(-reconstructed_score))

    pipe_prob = pipe.predict_proba([sample])[0, 1]
    assert np.isclose(reconstructed_prob, pipe_prob, atol=1e-5), (
        f"NB math mismatch: reconstructed {reconstructed_prob} vs pipeline {pipe_prob}"
    )


def test_explain_auto_selects_expected_methods():
    """Verify explain(..., method='auto') selects exact for linear/NB and lime for RF."""
    sample = "Subject: Hello friend\nLet's have lunch."
    exp_lr = explain(sample, model_name="logistic_regression", method="auto")
    assert exp_lr["method"] == "exact_linear"

    exp_nb = explain(sample, model_name="naive_bayes", method="auto")
    assert exp_nb["method"] == "exact_log_odds"

    exp_rf = explain(sample, model_name="random_forest", method="auto")
    assert exp_rf["method"] == "lime"


def test_delete_tokens_from_text():
    """Verify regex token deletion removes unigrams and bigrams case-insensitively."""
    text = "Get your free money today. Big discounts on click here links. Visit http://202.101.163.34:81"
    modified = delete_tokens_from_text(text, ["free", "money", "click here", "101 163"])
    assert "free" not in modified.lower()
    assert "money" not in modified.lower()
    assert "click here" not in modified.lower()
    assert "101.163" not in modified.lower()
    assert "today" in modified
    assert "discounts" in modified


def test_holdout_split_preserves_counts():
    """Verify holdout split loader returns exactly 2392 train and 563 test samples."""
    df, train_idx, test_idx = load_holdout_split()
    assert len(train_idx) == 2392
    assert len(test_idx) == 563
    test_df = df.iloc[test_idx]
    assert (test_df.label == 0).sum() == 456
    assert (test_df.label == 1).sum() == 107
