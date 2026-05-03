from __future__ import annotations

from core.schemas import EnrichedContext


class ContextualDataEnrichmentLayer:
    """Deterministic external/synthetic context enrichment.

    In production this module is where FRED, broker, mobility, geospatial, and
    internal market feeds are joined. The default implementation is deterministic
    and auditable so regression tests can verify claims.
    """

    def enrich(self, features: dict[str, float]) -> EnrichedContext:
        ir = features["interest_rate"]
        unemployment = features["unemployment_rate"]
        liquidity = features["market_liquidity"]
        pop = features["population_growth"]
        supply = features["supply_pipeline"]
        transit = features["transit_score"]
        crime = features["crime_rate"]

        stress_index = 0.38 * (ir / 0.08) + 0.27 * (unemployment / 0.10) + 0.2 * supply + 0.15 * (1 - liquidity)
        if stress_index > 0.78:
            cycle = "stress"
        elif stress_index > 0.58:
            cycle = "softening"
        elif pop > 0.018 and liquidity > 0.62:
            cycle = "expansion"
        else:
            cycle = "stable"

        market = "urban-core" if transit >= 70 else "suburban" if transit >= 40 else "exurban"
        location_summary = "strong access" if transit >= 70 and crime < 22 else "mixed access/risk" if crime < 45 else "elevated location risk"
        macro_summary = f"{cycle} market with interest_rate={ir:.3f}, unemployment_rate={unemployment:.3f}, liquidity={liquidity:.2f}."
        return EnrichedContext(
            market=market,
            market_cycle=cycle,  # type: ignore[arg-type]
            macro_summary=macro_summary,
            location_summary=location_summary,
            signals={
                "stress_index": round(float(stress_index), 4),
                "interest_rate_regime": "high" if ir >= 0.06 else "normal",
                "labor_market": "weak" if unemployment >= 0.065 else "normal",
                "supply_pressure": "high" if supply >= 0.08 else "normal",
                "location_summary": location_summary,
            },
        )
