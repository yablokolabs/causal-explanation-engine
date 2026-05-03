from __future__ import annotations

from core.config import load_settings
from core.pipeline import CausalExplanationEngine
from core.schemas import CREFeatures, ExplainRequest, PredictionRequest
from core.synthetic import feature_dict, generate_cre_dataset


def make_engine() -> CausalExplanationEngine:
    return CausalExplanationEngine(load_settings("configs/default.yaml"))


def sample_features(seed: int = 123) -> CREFeatures:
    x, _, order = generate_cre_dataset(n=1, seed=seed)
    return CREFeatures.model_validate(feature_dict(x[0], order))


def test_predict_explain_validate_end_to_end() -> None:
    engine = make_engine()
    features = sample_features()
    pred = engine.predict(PredictionRequest(features=features, trace_id="test-e2e"))
    assert pred.prediction > 0
    explanation = engine.explain(ExplainRequest(features=features, prediction=pred.prediction, trace_id="test-e2e"))
    validation = engine.validate(explanation)
    assert explanation.retrieved.facts
    assert explanation.justification
    assert validation.passed, validation.model_dump()
    assert validation.hallucination_rate == 0
    assert validation.causal_alignment_score >= 0.75


def test_attribution_contract_sorted_and_normalized() -> None:
    engine = make_engine()
    explanation = engine.explain(ExplainRequest(features=sample_features(456), trace_id="test-attr"))
    weights = [item.normalized_weight for item in explanation.attribution.all_attributions]
    assert all(a >= b - 1e-9 for a, b in zip(weights, weights[1:]))  # noqa: B905
    assert abs(sum(weights) - 1.0) < 1e-6
    assert len(explanation.attribution.top_k) == engine.settings.attribution.top_k


def test_no_unsupported_claims() -> None:
    engine = make_engine()
    explanation = engine.explain(ExplainRequest(features=sample_features(789), trace_id="test-grounding"))
    fact_ids = {fact.fact_id for fact in explanation.retrieved.facts}
    for claim in explanation.justification:
        assert claim.supported_by_fact_ids
        assert set(claim.supported_by_fact_ids) <= fact_ids
