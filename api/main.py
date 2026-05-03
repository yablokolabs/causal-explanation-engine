from __future__ import annotations

from functools import lru_cache
from fastapi import FastAPI

from core.config import load_settings
from core.logging import configure_logging
from core.pipeline import CausalExplanationEngine
from core.schemas import ExplainRequest, ExplanationResult, PredictionRequest, PredictionResponse, ValidateRequest, ValidationResult


@lru_cache(maxsize=1)
def get_engine() -> CausalExplanationEngine:
    settings = load_settings()
    configure_logging(settings.app.log_level)
    return CausalExplanationEngine(settings)


app = FastAPI(
    title="Causal Explanation Engine",
    version="0.1.0",
    description="Grounded, low-hallucination explanations for CRE model predictions.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    return get_engine().predict(request)


@app.post("/explain", response_model=ExplanationResult)
def explain(request: ExplainRequest) -> ExplanationResult:
    return get_engine().explain(request)


@app.post("/validate", response_model=ValidationResult)
def validate(request: ValidateRequest) -> ValidationResult:
    return get_engine().validate(request.explanation)
