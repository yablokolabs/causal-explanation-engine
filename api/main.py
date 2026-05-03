from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.config import load_settings
from core.logging import configure_logging
from core.pipeline import CausalExplanationEngine
from core.schemas import ExplainRequest, ExplanationResult, PredictionRequest, PredictionResponse, ValidateRequest, ValidationResult

_engine: CausalExplanationEngine | None = None


def get_engine() -> CausalExplanationEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialised — application did not start correctly")
    return _engine


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _engine  # noqa: PLW0603
    settings = load_settings()
    configure_logging(settings.app.log_level)
    _engine = CausalExplanationEngine(settings)
    yield
    _engine = None


app = FastAPI(
    title="Causal Explanation Engine",
    version="0.1.0",
    description="Grounded, low-hallucination explanations for CRE model predictions.",
    lifespan=lifespan,
)

allowed_origins = os.getenv("CEE_CORS_ORIGINS", "").split(",") if os.getenv("CEE_CORS_ORIGINS") else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if request.model_type == "gnn":
        raise HTTPException(status_code=501, detail="GNN backend is not yet available.")
    return get_engine().predict(request)


@app.post("/explain", response_model=ExplanationResult)
def explain(request: ExplainRequest) -> ExplanationResult:
    if request.model_type == "gnn":
        raise HTTPException(status_code=501, detail="GNN backend is not yet available.")
    return get_engine().explain(request)


@app.post("/validate", response_model=ValidationResult)
def validate(request: ValidateRequest) -> ValidationResult:
    return get_engine().validate(request.explanation)
