from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class AppConfig(BaseModel):
    name: str = "causal-explanation-engine"
    environment: str = "local"
    log_level: str = "INFO"


class ModelConfig(BaseModel):
    backend: str = "xgboost"
    artifact_path: str = "artifacts/xgb_cre.json"
    feature_order: list[str]


class AttributionConfig(BaseModel):
    top_k: int = 5
    method: str = "shap_tree_with_ablation_fallback"


class CausalConfig(BaseModel):
    min_abs_effect: float = 0.01
    bootstrap_samples: int = 64


class RetrievalConfig(BaseModel):
    top_k: int = 6
    lexical_weight: float = 0.55
    graph_weight: float = 0.45


class LLMConfig(BaseModel):
    mode: str = "deterministic_constrained"
    max_claims: int = 8


class ValidationConfig(BaseModel):
    max_unsupported_claims: int = 0
    min_causal_alignment_score: float = 0.75
    shap_tolerance: float = 1e-5


class EvaluationConfig(BaseModel):
    queries: int = 1000
    seed: int = 7


class Settings(BaseModel):
    app: AppConfig
    model: ModelConfig
    attribution: AttributionConfig = Field(default_factory=AttributionConfig)
    causal: CausalConfig = Field(default_factory=CausalConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


class ModelArtifactPath(BaseModel):
    """Validates that a model artifact path is safe and relative."""

    path: str = Field(..., min_length=1)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if ".." in Path(v).parts:
            raise ValueError(f"Path traversal not allowed in artifact path: {v}")
        return v


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    config_path = Path(path or os.getenv("CEE_CONFIG", "configs/default.yaml"))
    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    settings = Settings.model_validate(raw)
    env_artifact = os.getenv("CEE_MODEL_ARTIFACT")
    if env_artifact:
        ModelArtifactPath(path=env_artifact)
        settings.model.artifact_path = env_artifact
    return settings
