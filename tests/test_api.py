"""Automated tests for FastAPI endpoints."""
from fastapi.testclient import TestClient
import pytest

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert len(data["models_loaded"]) >= 5


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 5
    first = data["models"][0]
    assert "model" in first
    assert "accuracy" in first
    assert "f1" in first


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "model_comparison" in data
    assert "cv_grid_results" in data
    assert "html_ablation" in data


def test_predict_endpoint_valid(client):
    sample_text = "Subject: Urgent wire transfer required today.\nPlease click here to claim your cash."
    response = client.post("/predict", json={"text": sample_text, "model": "logistic_regression"})
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "Logistic Regression"
    assert data["label"] in (0, 1)
    assert data["label_name"] in ("ham", "spam")
    assert 0.0 <= data["spam_probability"] <= 1.0


def test_predict_endpoint_input_length_limit(client):
    huge_text = "spam " * 5000  # 25,000 chars, exceeds 20k limit
    response = client.post("/predict", json={"text": huge_text, "model": "logistic_regression"})
    assert response.status_code == 422  # Validation error from Pydantic


def test_explain_endpoint_exact(client):
    sample_text = "Subject: Free discount rolex watches online!"
    response = client.post("/explain", json={
        "text": sample_text,
        "model": "logistic_regression",
        "method": "exact",
        "top_k": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "exact_linear"
    assert "tokens" in data
    assert len(data["tokens"]) <= 5
    assert "base_value" in data
    assert "score" in data


def test_explain_global_terms(client):
    response = client.get("/explain/global?model=logistic_regression&top_k=10")
    assert response.status_code == 200
    data = response.json()
    assert "top_spam_terms" in data
    assert "top_ham_terms" in data
    assert len(data["top_spam_terms"]) > 0
    assert len(data["top_ham_terms"]) > 0


def test_explain_faithfulness_endpoint(client):
    response = client.get("/explain/faithfulness")
    assert response.status_code == 200
    data = response.json()
    assert "deletion_tests" in data
    assert "method_agreement" in data


def test_explain_errors_endpoint(client):
    response = client.get("/explain/errors")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "complement_nb" in data["models"]

