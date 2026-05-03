from __future__ import annotations

from core.attribution import ShapAttributionLayer
from core.config import Settings
from core.enrichment import ContextualDataEnrichmentLayer
from core.logging import ensure_trace_id, traced_span
from core.schemas import ExplainRequest, ExplanationResult, PredictionRequest, PredictionResponse, ValidationResult
from causal.engine import CausalInferenceLayer
from llm.narrator import ConstrainedNarrationLayer
from models.gnn import GNNPlaceholderModel
from models.xgboost_model import XGBoostCREModel
from retrieval.knowledge_base import HybridKnowledgeRetriever
from validation.validator import DeterministicValidationLayer


class CausalExplanationEngine:
    """Orchestrates prediction, explanation, and validation for CRE models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.xgb = XGBoostCREModel(settings.model.artifact_path, settings.model.feature_order).ensure_loaded()
        self.gnn = GNNPlaceholderModel()
        self.attribution = ShapAttributionLayer(self.xgb, top_k=settings.attribution.top_k)
        self.causal = CausalInferenceLayer(settings.causal.min_abs_effect, settings.causal.bootstrap_samples)
        self.retriever = HybridKnowledgeRetriever(settings.retrieval.top_k, settings.retrieval.lexical_weight, settings.retrieval.graph_weight)
        self.enrichment = ContextualDataEnrichmentLayer()
        self.narrator = ConstrainedNarrationLayer(settings.llm.max_claims)
        self.validator = DeterministicValidationLayer(
            settings.validation.shap_tolerance,
            settings.validation.min_causal_alignment_score,
            settings.validation.max_unsupported_claims,
        )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run model inference and return a prediction with metadata."""
        trace_id = ensure_trace_id(request.trace_id)
        features = request.features.model_dump()
        with traced_span("predict", model_type=request.model_type):
            if request.model_type == "gnn":
                raise NotImplementedError("GNN backend is not yet available.")
            prediction = self.xgb.predict_one(features)
            version = self.xgb.model_version
            order = self.xgb.feature_order
            return PredictionResponse(prediction=prediction, model_type=request.model_type, model_version=version, feature_order=order, trace_id=trace_id)

    def explain(self, request: ExplainRequest) -> ExplanationResult:
        """Generate a grounded causal explanation for a prediction.

        Attribution is always computed from the model matching ``request.model_type``.
        """
        trace_id = ensure_trace_id(request.trace_id)
        features = request.features.model_dump()
        with traced_span("explain", model_type=request.model_type):
            if request.model_type == "gnn":
                raise NotImplementedError("GNN backend is not yet available.")
            attribution = self.attribution.compute(features, top_k=request.top_k)
            causal = self.causal.estimate(attribution, features)
            context = self.enrichment.enrich(features)
            driver_features = [i.feature for i in attribution.top_k] + causal.causal_drivers
            retrieved = self.retriever.retrieve(driver_features, [context.market, context.market_cycle, context.location_summary])
            return self.narrator.narrate(attribution, causal, retrieved, context, trace_id)

    def validate(self, explanation: ExplanationResult) -> ValidationResult:
        """Run deterministic validation checks on an explanation."""
        ensure_trace_id(explanation.trace_id)
        with traced_span("validate"):
            return self.validator.validate(explanation)
