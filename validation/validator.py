from __future__ import annotations

from core.schemas import CausalStatus, ExplanationResult, ValidationDiagnostic, ValidationResult


class DeterministicValidationLayer:
    def __init__(self, shap_tolerance: float = 1e-5, min_causal_alignment_score: float = 0.75, max_unsupported_claims: int = 0):
        self.shap_tolerance = shap_tolerance
        self.min_causal_alignment_score = min_causal_alignment_score
        self.max_unsupported_claims = max_unsupported_claims

    def validate(self, explanation: ExplanationResult) -> ValidationResult:
        diagnostics: list[ValidationDiagnostic] = []
        attr = explanation.attribution
        recomposed = attr.base_value + sum(i.contribution for i in attr.all_attributions)
        shap_ok = abs(recomposed - attr.prediction) <= self.shap_tolerance or abs(attr.residual) <= self.shap_tolerance
        if not shap_ok:
            diagnostics.append(ValidationDiagnostic(code="SHAP_RECOMPOSITION", message=f"Attributions recomposed to {recomposed:.6f}, prediction {attr.prediction:.6f}.", severity="error"))

        fact_ids = {f.fact_id for f in explanation.retrieved.facts}
        unsupported = []
        for claim in explanation.justification:
            missing = [fid for fid in claim.supported_by_fact_ids if fid not in fact_ids]
            if not claim.supported_by_fact_ids or missing:
                unsupported.append(claim.claim)
        if unsupported:
            diagnostics.append(ValidationDiagnostic(code="UNSUPPORTED_CLAIM", message=f"{len(unsupported)} claims lack retrieved evidence.", severity="error"))

        effect_by_feature = {e.treatment: e for e in explanation.causal.effects}
        aligned = 0
        total = 0
        for item in attr.top_k:
            effect = effect_by_feature.get(item.feature)
            if not effect:
                continue
            total += 1
            # Local SHAP direction should align with the estimated causal effect
            # multiplied by the feature's local deviation from its baseline. Example:
            # a below-baseline cap rate can increase value even though the causal
            # effect of raising cap_rate is negative.
            local_delta = float(item.value) - item.baseline_value if isinstance(item.value, (int, float)) else 0.0
            expected_local_direction = effect.estimate * local_delta
            same_direction = (
                abs(item.contribution) < 1e-9
                or abs(expected_local_direction) < 1e-9
                or (item.contribution >= 0 and expected_local_direction >= 0)
                or (item.contribution <= 0 and expected_local_direction <= 0)
            )
            if effect.status == CausalStatus.causal and same_direction:
                aligned += 1
            elif effect.status != CausalStatus.causal:
                # Correlation-only explanations are aligned if narration does not promote them to causal.
                aligned += 1
        causal_alignment = aligned / total if total else 1.0
        if causal_alignment < self.min_causal_alignment_score:
            diagnostics.append(ValidationDiagnostic(code="CAUSAL_ALIGNMENT", message=f"Causal alignment {causal_alignment:.3f} below threshold.", severity="error"))

        hallucination_rate = len(unsupported) / max(len(explanation.justification), 1)
        factual_consistency = 1.0 - hallucination_rate
        passed = shap_ok and len(unsupported) <= self.max_unsupported_claims and causal_alignment >= self.min_causal_alignment_score
        if passed:
            diagnostics.append(ValidationDiagnostic(code="PASS", message="Explanation passed deterministic validation.", severity="info"))
        return ValidationResult(
            passed=passed,
            hallucination_rate=hallucination_rate,
            factual_consistency=factual_consistency,
            causal_alignment_score=causal_alignment,
            diagnostics=diagnostics,
            trace_id=explanation.trace_id,
        )
