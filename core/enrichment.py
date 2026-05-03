from __future__ import annotations

from core.schemas import EnrichedContext

# Stress index weights and normalization reference values
_STRESS_WEIGHT_IR = 0.38
_STRESS_WEIGHT_UNEMPLOYMENT = 0.27
_STRESS_WEIGHT_SUPPLY = 0.20
_STRESS_WEIGHT_LIQUIDITY = 0.15
_IR_NORM_REF = 0.08
_UNEMPLOYMENT_NORM_REF = 0.10

# Cycle thresholds
_STRESS_THRESHOLD = 0.78
_SOFTENING_THRESHOLD = 0.58
_EXPANSION_POP_MIN = 0.018
_EXPANSION_LIQUIDITY_MIN = 0.62

# Market classification thresholds
_URBAN_CORE_TRANSIT_MIN = 70
_SUBURBAN_TRANSIT_MIN = 40
_LOW_CRIME_MAX = 22
_MODERATE_CRIME_MAX = 45

# Regime thresholds
_HIGH_IR_THRESHOLD = 0.06
_WEAK_LABOR_THRESHOLD = 0.065
_HIGH_SUPPLY_THRESHOLD = 0.08


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

        stress_index = (
            _STRESS_WEIGHT_IR * (ir / _IR_NORM_REF)
            + _STRESS_WEIGHT_UNEMPLOYMENT * (unemployment / _UNEMPLOYMENT_NORM_REF)
            + _STRESS_WEIGHT_SUPPLY * supply
            + _STRESS_WEIGHT_LIQUIDITY * (1 - liquidity)
        )
        if stress_index > _STRESS_THRESHOLD:
            cycle = "stress"
        elif stress_index > _SOFTENING_THRESHOLD:
            cycle = "softening"
        elif pop > _EXPANSION_POP_MIN and liquidity > _EXPANSION_LIQUIDITY_MIN:
            cycle = "expansion"
        else:
            cycle = "stable"

        market = "urban-core" if transit >= _URBAN_CORE_TRANSIT_MIN else "suburban" if transit >= _SUBURBAN_TRANSIT_MIN else "exurban"
        location_summary = "strong access" if transit >= _URBAN_CORE_TRANSIT_MIN and crime < _LOW_CRIME_MAX else "mixed access/risk" if crime < _MODERATE_CRIME_MAX else "elevated location risk"
        macro_summary = f"{cycle} market with interest_rate={ir:.3f}, unemployment_rate={unemployment:.3f}, liquidity={liquidity:.2f}."
        return EnrichedContext(
            market=market,
            market_cycle=cycle,  # type: ignore[arg-type]
            macro_summary=macro_summary,
            location_summary=location_summary,
            signals={
                "stress_index": round(float(stress_index), 4),
                "interest_rate_regime": "high" if ir >= _HIGH_IR_THRESHOLD else "normal",
                "labor_market": "weak" if unemployment >= _WEAK_LABOR_THRESHOLD else "normal",
                "supply_pressure": "high" if supply >= _HIGH_SUPPLY_THRESHOLD else "normal",
                "location_summary": location_summary,
            },
        )
