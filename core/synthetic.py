from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

NDArrayFloat = np.ndarray[Any, np.dtype[np.floating[Any]]]

FEATURE_ORDER = [
    "cap_rate",
    "occupancy_rate",
    "noi_growth",
    "interest_rate",
    "unemployment_rate",
    "population_growth",
    "transit_score",
    "crime_rate",
    "property_age",
    "lease_term_months",
    "market_liquidity",
    "supply_pipeline",
    "submarket_score",
]


def generate_cre_dataset(
    n: int = 2500, seed: int = 42
) -> tuple[NDArrayFloat, NDArrayFloat, list[str]]:
    rng = np.random.default_rng(seed)
    cap_rate = rng.normal(0.055, 0.012, n).clip(0.025, 0.11)
    occupancy_rate = rng.normal(0.91, 0.07, n).clip(0.55, 0.995)
    noi_growth = rng.normal(0.025, 0.018, n).clip(-0.06, 0.10)
    interest_rate = rng.normal(0.047, 0.011, n).clip(0.015, 0.09)
    unemployment_rate = rng.normal(0.045, 0.016, n).clip(0.015, 0.12)
    population_growth = rng.normal(0.012, 0.015, n).clip(-0.04, 0.07)
    transit_score = rng.normal(62, 18, n).clip(0, 100)
    crime_rate = rng.gamma(2.0, 9.0, n).clip(0, 75)
    property_age = rng.gamma(3.0, 8.0, n).clip(0, 90)
    lease_term_months = rng.normal(48, 18, n).clip(3, 144)
    market_liquidity = rng.beta(5, 3, n).clip(0.05, 0.98)
    supply_pipeline = rng.gamma(2.0, 0.018, n).clip(0, 0.16)
    submarket_score = (
        0.45 * transit_score
        + 160 * population_growth
        - 0.45 * crime_rate
        - 95 * unemployment_rate
        + rng.normal(25, 8, n)
    ).clip(0, 100)

    x = np.column_stack(
        [
            cap_rate,
            occupancy_rate,
            noi_growth,
            interest_rate,
            unemployment_rate,
            population_growth,
            transit_score,
            crime_rate,
            property_age,
            lease_term_months,
            market_liquidity,
            supply_pipeline,
            submarket_score,
        ]
    )

    # Value score in basis-point-like units. Formula encodes the domain causal truth used by tests.
    y = (
        100
        - 520 * cap_rate
        + 42 * occupancy_rate
        + 310 * noi_growth
        - 380 * interest_rate
        - 160 * unemployment_rate
        + 125 * population_growth
        + 0.115 * transit_score
        - 0.052 * crime_rate
        - 0.033 * property_age
        + 0.028 * lease_term_months
        + 16.0 * market_liquidity
        - 45.0 * supply_pipeline
        + 0.19 * submarket_score
        + 7.0 * occupancy_rate * market_liquidity
        - 90.0 * interest_rate * cap_rate
        + rng.normal(0, 0.55, n)
    )
    return x.astype(float), y.astype(float), FEATURE_ORDER.copy()


def feature_dict(row: Iterable[float], feature_order: list[str] | None = None) -> dict[str, float]:
    order = feature_order or FEATURE_ORDER
    return {name: float(value) for name, value in zip(order, row, strict=True)}


def write_golden(path: str | Path, n: int = 1000, seed: int = 7) -> Path:
    x, y, feature_order = generate_cre_dataset(n=n, seed=seed)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"id": f"golden-{i:04d}", "features": feature_dict(x[i], feature_order), "target": float(y[i])}
        for i in range(n)
    ]
    target.write_text(json.dumps({"feature_order": feature_order, "records": records}, indent=2), encoding="utf-8")
    return target
