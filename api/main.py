"""FastAPI Backend for Email Spam Detection and Explainable AI."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.explain.global_terms import compute_global_terms
from src.explain.local import explain_instance
from src.explain.utils import (
    CANONICAL_NAMES,
    EXPLAIN_DIR,
    REPORTS_DIR,
    get_prediction_score_and_prob,
    load_model,
    normalize_model_name,
)
from .schemas import (
    ExplainRequest,
    ExplainResponse,
    GlobalTermsResponse,
    HealthResponse,
    MetricsBundleResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)

# Logging configuration (strictly logs metadata; NEVER logs email text)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spam_detector_api")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter per client IP."""

    def __init__(self, app: Any, max_requests: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Exclude health check from rate limits
        if request.url.path == "/health":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        timestamps = self.requests[client_ip]

        # Prune old timestamps
        self.requests[client_ip] = [t for t in timestamps if now - t < self.window_seconds]

        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning("Rate limit exceeded for client %s", client_ip)
            return Response(
                content=json.dumps({"detail": "Too many requests. Please slow down."}),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
            )

        self.requests[client_ip].append(now)
        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Enforces request timeout (e.g. 15s) to avoid hanging worker threads."""

    def __init__(self, app: Any, timeout_seconds: float = 15.0):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            logger.error("Request timed out on %s %s", request.method, request.url.path)
            return Response(
                content=json.dumps({"detail": "Request processing timed out."}),
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                media_type="application/json",
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load all trained models at startup and check artifacts."""
    loaded = []
    missing = []
    for tag in CANONICAL_NAMES:
        try:
            load_model(tag)
            loaded.append(tag)
        except Exception as exc:
            missing.append(tag)

    if missing:
        logger.warning(
            "Some models could not be loaded: %s. Run 'python src/prepare_data.py && python src/train_evaluate.py' to generate them.",
            missing,
        )
    else:
        logger.info("Successfully loaded all %d classification models into memory.", len(loaded))

    app.state.loaded_models = loaded
    yield
    logger.info("Shutting down Email Spam Detector API.")


app = FastAPI(
    title="Email Spam Detection & XAI API",
    description="Production-grade API providing spam classification and faithful token explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
raw_cors = os.environ.get("CORS_ORIGINS", "*")
origins = [o.strip() for o in raw_cors.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware, max_requests=120, window_seconds=60)
app.add_middleware(TimeoutMiddleware, timeout_seconds=15.0)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health and model readiness probe."""
    loaded = getattr(app.state, "loaded_models", [])
    return HealthResponse(
        status="ok",
        version="1.0.0",
        models_loaded=loaded,
    )


@app.get("/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    """List all available models along with their verified holdout metrics."""
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="reports/metrics.json missing. Run python src/train_evaluate.py first.",
        )

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed reading metrics: {exc}")

    results = []
    for tag, canonical in CANONICAL_NAMES.items():
        m_info = data.get("models", {}).get(canonical, {})
        results.append({
            "model": canonical,
            "model_tag": tag,
            "accuracy": float(m_info.get("accuracy", 0.0)),
            "precision": float(m_info.get("precision", 0.0)),
            "recall": float(m_info.get("recall", 0.0)),
            "f1": float(m_info.get("f1", 0.0)),
            "cv_best_spam_f1": m_info.get("cv_best_spam_f1"),
            "cv_best_parameters": m_info.get("cv_best_parameters"),
        })

    return ModelsResponse(models=results)


@app.get("/metrics", response_model=MetricsBundleResponse)
async def get_metrics_bundle() -> MetricsBundleResponse:
    """Return consolidated metric tables: metrics.json, model_comparison, cv_grid, html_ablation."""
    metrics_json_path = REPORTS_DIR / "metrics.json"
    if not metrics_json_path.exists():
        raise HTTPException(status_code=503, detail="reports/metrics.json not found.")

    try:
        summary_data = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading metrics.json: {exc}")

    def read_csv_safe(fname: str) -> List[Dict[str, Any]]:
        path = REPORTS_DIR / fname
        if path.exists():
            try:
                return pd.read_csv(path).fillna("").to_dict(orient="records")
            except Exception:
                return []
        return []

    return MetricsBundleResponse(
        summary=summary_data,
        model_comparison=read_csv_safe("model_comparison.csv"),
        cv_grid_results=read_csv_safe("cv_grid_results.csv"),
        html_ablation=read_csv_safe("html_ablation.csv"),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_spam(request: PredictRequest) -> PredictResponse:
    """Predict whether an email is Spam (1) or Ham (0). Input body is strictly never logged."""
    try:
        tag = normalize_model_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        pipeline = load_model(tag)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    label, _, prob = get_prediction_score_and_prob(pipeline, request.text)
    logger.info("Prediction request handled: model=%s label=%d prob=%.4f", tag, label, prob)

    return PredictResponse(
        model=CANONICAL_NAMES[tag],
        model_tag=tag,
        label=label,
        label_name="spam" if label == 1 else "ham",
        spam_probability=prob,
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest) -> ExplainResponse:
    """Generate local per-token signed attributions. Input body is strictly never logged."""
    try:
        tag = normalize_model_name(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        # Cap LIME num_samples to max 250 for low latency
        result = explain_instance(
            text=request.text,
            model_name=tag,
            method=request.method,
            top_k=request.top_k,
            num_samples=250,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Explanation error for model %s: %s", tag, exc)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")

    logger.info("Explanation request handled: model=%s method=%s top_k=%d", tag, result["method"], request.top_k)
    return ExplainResponse(**result)


@app.get("/explain/global", response_model=GlobalTermsResponse)
async def get_global_terms(
    model: str = Query(default="logistic_regression", description="Model name or tag"),
    top_k: int = Query(default=25, ge=5, le=50),
) -> GlobalTermsResponse:
    """Return top spam-indicating and ham-indicating vocabulary terms globally for a model."""
    try:
        tag = normalize_model_name(model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cache_file = EXPLAIN_DIR / f"global_terms_{tag}.json"
    if cache_file.exists() and top_k == 25:
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            return GlobalTermsResponse(**cached_data)
        except Exception:
            pass

    try:
        data = compute_global_terms(tag, top_k=top_k)
        return GlobalTermsResponse(**data)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed computing global terms: {exc}")


@app.get("/explain/faithfulness")
async def get_faithfulness_results() -> Dict[str, Any]:
    """Return precomputed faithfulness deletion and cross-method agreement results."""
    json_path = EXPLAIN_DIR / "faithfulness.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=503,
            detail="reports/explain/faithfulness.json missing. Run python -m src.explain.runner first.",
        )
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading faithfulness results: {exc}")


@app.get("/explain/errors")
async def get_error_analysis() -> Dict[str, Any]:
    """Return aggregate error patterns and misleading token summaries (no raw text stored)."""
    json_path = EXPLAIN_DIR / "error_analysis_summary.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=503,
            detail="reports/explain/error_analysis_summary.json missing. Run python -m src.explain.runner first.",
        )
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading error analysis: {exc}")
