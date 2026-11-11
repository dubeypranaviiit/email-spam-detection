"""Pydantic schemas for the Email Spam Detection API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    models_loaded: List[str] = Field(default_factory=list)


class ModelMetrics(BaseModel):
    model: str
    model_tag: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cv_best_spam_f1: Optional[float] = None
    cv_best_parameters: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    models: List[ModelMetrics]


class MetricsBundleResponse(BaseModel):
    summary: Dict[str, Any]
    model_comparison: List[Dict[str, Any]]
    cv_grid_results: List[Dict[str, Any]]
    html_ablation: List[Dict[str, Any]]


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Raw email subject and body text (max 20,000 characters).",
    )
    model: str = Field(
        default="logistic_regression",
        description="Model identifier (e.g. logistic_regression, linear_svm, naive_bayes, complement_nb, random_forest).",
    )


class PredictResponse(BaseModel):
    model: str
    model_tag: str
    label: int = Field(description="0 for ham, 1 for spam")
    label_name: str = Field(description="'ham' or 'spam'")
    spam_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExplainRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Raw email subject and body text (max 20,000 characters).",
    )
    model: str = Field(
        default="logistic_regression",
        description="Model identifier.",
    )
    method: str = Field(
        default="auto",
        description="Explanation method: 'auto', 'exact', or 'lime'.",
    )
    top_k: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Number of top contributing tokens to return.",
    )


class TokenAttribution(BaseModel):
    token: str
    weight: float
    abs_weight: Optional[float] = None
    sign: str
    indication: Optional[str] = None
    tfidf: Optional[float] = None
    coefficient: Optional[float] = None


class ExplainResponse(BaseModel):
    model: str
    model_tag: str
    method: str
    label: int
    label_name: str
    spam_probability: float
    base_value: float
    score: float
    tokens: List[TokenAttribution]
    all_tokens_count: int


class GlobalTermsResponse(BaseModel):
    model: str
    model_tag: str
    method: str
    formula: Optional[str] = None
    top_k: int
    top_spam_terms: List[Dict[str, Any]]
    top_ham_terms: List[Dict[str, Any]]
