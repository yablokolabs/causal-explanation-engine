from __future__ import annotations

from core.schemas import (
    AttributionResult,
    CausalResult,
    CausalStatus,
    EnrichedContext,
    ExplanationClaim,
    ExplanationResult,
    RetrievalResult,
)


class ConstrainedNarrationLayer:
    """Schema-constrained narration.

    Default mode is deterministic and extractive: every sentence is assembled
    from attribution, causal estimates, context, and retrieved fact IDs. A hosted
    LLM can be added behind this class only if it returns the same JSON schema
    and passes validation.
    """

    def __init__(self, max_claims: int = 8):
        self.max_claims = max_claims

    def narrate(
        self,
        attribution: AttributionResult,
        causal: CausalResult,
        retrieved: RetrievalResult,
        context: EnrichedContext,
        trace_id: str,
    ) -> ExplanationResult:
        facts_by_feature = self._facts_by_feature(retrieved)
        effect_by_feature = {e.treatment: e for e in causal.effects}
        claims: list[ExplanationClaim] = []
        sentences: list[str] = [
            f"Prediction: {attribution.prediction:.2f}. The explanation is limited to the top model attributions, causal graph links, and retrieved CRE evidence."
        ]
        for item in attribution.top_k:
            effect = effect_by_feature.get(item.feature)
            status = effect.status if effect else CausalStatus.unknown
            fact_ids = facts_by_feature.get(item.feature, []) or [f.fact_id for f in retrieved.facts if f.relationship == "constraint"][:1]
            causal_phrase = "causal driver" if status == CausalStatus.causal else "model driver without a direct causal claim"
            effect_phrase = ""
            if effect:
                effect_phrase = f" Estimated {effect.effect_type} is {effect.estimate:.3f} with interval [{effect.lower_bound:.3f}, {effect.upper_bound:.3f}]."
            sentence = (
                f"{item.feature} {item.direction} the model output by {item.contribution:.3f} "
                f"({item.normalized_weight:.1%} of absolute attribution) and is treated as a {causal_phrase}."
                f"{effect_phrase} Evidence: {', '.join(fact_ids)}."
            )
            sentences.append(sentence)
            claims.append(
                ExplanationClaim(
                    claim=sentence,
                    supported_by_fact_ids=fact_ids,
                    features=[item.feature],
                    causal_status=status,
                )
            )
            if len(claims) >= self.max_claims:
                break
        context_sentence = f"Context: {context.macro_summary} Location is {context.location_summary} in a {context.market} market."
        sentences.append(context_sentence)
        claims.append(
            ExplanationClaim(
                claim=context_sentence,
                supported_by_fact_ids=[f.fact_id for f in retrieved.facts if f.relationship in {"context", "constraint", "definition"}][:1],
                features=[],
                causal_status=CausalStatus.constrained,
            )
        )
        return ExplanationResult(
            explanation_text="\n".join(sentences),
            justification=claims,
            attribution=attribution,
            causal=causal,
            retrieved=retrieved,
            context=context,
            trace_id=trace_id,
        )

    def _facts_by_feature(self, retrieved: RetrievalResult) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for fact in retrieved.facts:
            for feature in fact.related_features:
                mapping.setdefault(feature, []).append(fact.fact_id)
        return mapping
