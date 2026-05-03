from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FeatureValue = float | int | str | bool


class CREFeatures(BaseModel):
    """Canonical CRE feature vector accepted by the engine."""

    cap_rate: float = Field(..., description="Going-in capitalization rate as decimal")
    occupancy_rate: float = Field(..., ge=0, le=1)
    noi_growth: float = Field(..., description="Expected NOI growth as decimal")
    interest_rate: float = Field(..., ge=0)
    unemployment_rate: float = Field(..., ge=0, le=1)
    population_growth: float
    transit_score: float = Field(..., ge=0, le=100)
    crime_rate: float = Field(..., ge=0)
    property_age: float = Field(..., ge=0)
    lease_term_months: float = Field(..., ge=0)
    market_liquidity: float = Field(..., ge=0, le=1)
    supply_pipeline: float = Field(..., ge=0)
    submarket_score: float = Field(..., ge=0, le=100)

    model_config = ConfigDict(extra="forbid")


class PredictionRequest(BaseModel):
    features: CREFeatures
    model_type: Literal["xgboost", "gnn"] = "xgboost"
    trace_id: str | None = None


class PredictionResponse(BaseModel):
    prediction: float
    model_type: str
    model_version: str
    feature_order: list[str]
    trace_id: str


class AttributionItem(BaseModel):
    feature: str
    value: FeatureValue
    baseline_value: float
    contribution: float
    normalized_weight: float
    direction: Literal["increases", "decreases", "neutral"]
    rank: int


class AttributionResult(BaseModel):
    method: str
    base_value: float
    prediction: float
    residual: float
    top_k: list[AttributionItem]
    all_attributions: list[AttributionItem]


class CausalStatus(str, Enum):
    causal = "causal"
    correlated = "correlated"
    constrained = "constrained"
    unknown = "unknown"


class CausalEffect(BaseModel):
    treatment: str
    outcome: str
    effect_type: Literal["ATE", "CATE"]
    estimate: float
    lower_bound: float
    upper_bound: float
    status: CausalStatus
    adjustment_set: list[str] = Field(default_factory=list)
    evidence: str


class CausalGraph(BaseModel):
    nodes: list[str]
    edges: list[tuple[str, str]]
    constraints: list[str]


class CausalResult(BaseModel):
    graph: CausalGraph
    effects: list[CausalEffect]
    causal_drivers: list[str]
    correlated_only_drivers: list[str]


class RetrievedFact(BaseModel):
    fact_id: str
    text: str
    source: str
    related_features: list[str]
    relationship: Literal["causal", "constraint", "definition", "context"]
    score: float


class RetrievalResult(BaseModel):
    facts: list[RetrievedFact]
    query_terms: list[str]


class EnrichedContext(BaseModel):
    market: str
    market_cycle: Literal["expansion", "stable", "softening", "stress"]
    macro_summary: str
    location_summary: str
    signals: dict[str, float | str]


class ExplanationClaim(BaseModel):
    claim: str
    supported_by_fact_ids: list[str]
    features: list[str]
    causal_status: CausalStatus


class ExplanationResult(BaseModel):
    explanation_text: str
    justification: list[ExplanationClaim]
    attribution: AttributionResult
    causal: CausalResult
    retrieved: RetrievalResult
    context: EnrichedContext
    trace_id: str


class ExplainRequest(BaseModel):
    features: CREFeatures
    prediction: float | None = None
    model_type: Literal["xgboost", "gnn"] = "xgboost"
    top_k: int | None = Field(default=None, ge=1, le=13)
    trace_id: str | None = None


class ValidationDiagnostic(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"]


class ValidationResult(BaseModel):
    passed: bool
    hallucination_rate: float
    factual_consistency: float
    causal_alignment_score: float
    diagnostics: list[ValidationDiagnostic]
    trace_id: str


class ValidateRequest(BaseModel):
    explanation: ExplanationResult


class RegressionMetrics(BaseModel):
    queries: int
    hallucination_rate: float
    factual_consistency: float
    causal_alignment_score: float
    passed: bool
