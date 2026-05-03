from __future__ import annotations

import hashlib

import numpy as np

from core.schemas import AttributionResult, CausalEffect, CausalGraph, CausalResult, CausalStatus
from core.synthetic import generate_cre_dataset

OUTCOME = "predicted_value_score"

CRE_CAUSAL_EDGES: list[tuple[str, str]] = [
    ("interest_rate", "cap_rate"),
    ("interest_rate", OUTCOME),
    ("cap_rate", OUTCOME),
    ("occupancy_rate", OUTCOME),
    ("noi_growth", OUTCOME),
    ("unemployment_rate", "occupancy_rate"),
    ("unemployment_rate", OUTCOME),
    ("population_growth", "occupancy_rate"),
    ("population_growth", OUTCOME),
    ("transit_score", "submarket_score"),
    ("transit_score", OUTCOME),
    ("crime_rate", "submarket_score"),
    ("crime_rate", OUTCOME),
    ("property_age", OUTCOME),
    ("lease_term_months", OUTCOME),
    ("market_liquidity", OUTCOME),
    ("supply_pipeline", "cap_rate"),
    ("supply_pipeline", OUTCOME),
    ("submarket_score", OUTCOME),
]

CONSTRAINTS = [
    "Claims about drivers must reference either a SHAP contribution or a causal effect estimate.",
    "Correlation-only drivers cannot be narrated as direct causes.",
    "CRE valuation claims must be supported by retrieved facts or graph constraints.",
]

ADJUSTMENTS: dict[str, list[str]] = {
    "cap_rate": ["interest_rate", "supply_pipeline", "submarket_score"],
    "occupancy_rate": ["unemployment_rate", "population_growth", "submarket_score"],
    "noi_growth": ["population_growth", "market_liquidity"],
    "interest_rate": ["unemployment_rate", "market_liquidity"],
    "unemployment_rate": ["population_growth"],
    "population_growth": ["unemployment_rate"],
    "transit_score": ["submarket_score", "crime_rate"],
    "crime_rate": ["submarket_score", "transit_score"],
    "property_age": ["submarket_score"],
    "lease_term_months": ["occupancy_rate", "market_liquidity"],
    "market_liquidity": ["interest_rate", "submarket_score"],
    "supply_pipeline": ["population_growth", "submarket_score"],
    "submarket_score": ["transit_score", "crime_rate", "population_growth"],
}


class CausalInferenceLayer:
    """Causal effect estimator with optional DoWhy/EconML seam.

    Effects are estimated from the synthetic CRE data generating process via
    adjusted linear regression. The interface mirrors ATE/CATE outputs used by
    DoWhy/EconML, and this adapter can be swapped without touching downstream
    validation or narration.
    """

    def __init__(self, min_abs_effect: float = 0.01, bootstrap_samples: int = 64):
        self.min_abs_effect = min_abs_effect
        self.bootstrap_samples = bootstrap_samples
        self.x, self.y, self.feature_order = generate_cre_dataset(n=3200, seed=99)
        self.index = {name: i for i, name in enumerate(self.feature_order)}
        self._effect_cache: dict[str, tuple[float, float, float]] = {}

    def _get_effect(self, feature: str) -> tuple[float, float, float]:
        """Return cached adjusted-effect estimate, computing lazily."""
        if feature not in self._effect_cache:
            self._effect_cache[feature] = self._adjusted_effect(feature)
        return self._effect_cache[feature]

    def estimate(self, attribution: AttributionResult, features: dict[str, float]) -> CausalResult:
        effects: list[CausalEffect] = []
        causal_drivers: list[str] = []
        correlated: list[str] = []
        edge_set = set(CRE_CAUSAL_EDGES)
        for item in attribution.top_k:
            feature = item.feature
            status = CausalStatus.causal if (feature, OUTCOME) in edge_set else CausalStatus.correlated
            estimate, lo, hi = self._get_effect(feature)
            if status == CausalStatus.causal and abs(estimate) >= self.min_abs_effect:
                causal_drivers.append(feature)
            else:
                correlated.append(feature)
            effects.append(
                CausalEffect(
                    treatment=feature,
                    outcome=OUTCOME,
                    effect_type="CATE",
                    estimate=float(estimate),
                    lower_bound=float(lo),
                    upper_bound=float(hi),
                    status=status,
                    adjustment_set=ADJUSTMENTS.get(feature, []),
                    evidence=(
                        "Adjusted linear CATE proxy over synthetic CRE structural data; "
                        f"controls={ADJUSTMENTS.get(feature, [])}."
                    ),
                )
            )
        graph = CausalGraph(
            nodes=sorted({n for edge in CRE_CAUSAL_EDGES for n in edge}),
            edges=CRE_CAUSAL_EDGES,
            constraints=CONSTRAINTS,
        )
        return CausalResult(
            graph=graph, effects=effects, causal_drivers=causal_drivers, correlated_only_drivers=correlated
        )

    def _adjusted_effect(self, treatment: str) -> tuple[float, float, float]:
        cols = [treatment] + [c for c in ADJUSTMENTS.get(treatment, []) if c in self.index]
        matrix = np.column_stack([np.ones(len(self.x))] + [self.x[:, self.index[c]] for c in cols])
        beta, *_ = np.linalg.lstsq(matrix, self.y, rcond=None)
        estimate = float(beta[1])
        # Deterministic bootstrap for a conservative interval.
        stable_seed = int(hashlib.sha256(treatment.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(stable_seed)
        boots = []
        n = len(self.y)
        for _ in range(self.bootstrap_samples):
            sample = rng.integers(0, n, n // 2)
            b, *_ = np.linalg.lstsq(matrix[sample], self.y[sample], rcond=None)
            boots.append(float(b[1]))
        lo, hi = np.quantile(boots, [0.025, 0.975]) if boots else (estimate, estimate)
        return estimate, float(lo), float(hi)
