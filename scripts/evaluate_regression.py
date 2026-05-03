from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core.config import load_settings
from core.logging import configure_logging
from core.pipeline import CausalExplanationEngine
from core.schemas import CREFeatures, ExplainRequest, RegressionMetrics
from core.synthetic import feature_dict, generate_cre_dataset, write_golden

logger = logging.getLogger(__name__)


def run(n: int, seed: int, golden_path: Path | None = None) -> RegressionMetrics:
    settings = load_settings()
    configure_logging("WARNING")
    engine = CausalExplanationEngine(settings)
    x, _, order = generate_cre_dataset(n=n, seed=seed)
    if golden_path:
        write_golden(golden_path, n=n, seed=seed)
    hallucination = []
    factual = []
    alignment = []
    failures = 0
    for i in range(n):
        if (i + 1) % 100 == 0 or i == n - 1:
            logger.warning("regression progress: %d / %d", i + 1, n)
        features = CREFeatures.model_validate(feature_dict(x[i], order))
        explanation = engine.explain(ExplainRequest(features=features, trace_id=f"regression-{i:04d}"))
        validation = engine.validate(explanation)
        hallucination.append(validation.hallucination_rate)
        factual.append(validation.factual_consistency)
        alignment.append(validation.causal_alignment_score)
        failures += 0 if validation.passed else 1
    metrics = RegressionMetrics(
        queries=n,
        hallucination_rate=sum(hallucination) / n,
        factual_consistency=sum(factual) / n,
        causal_alignment_score=sum(alignment) / n,
        passed=(sum(hallucination) / n) < 0.01
        and (sum(factual) / n) >= 0.99
        and (sum(alignment) / n) >= settings.validation.min_causal_alignment_score,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--golden", type=Path, default=Path("data/golden/cre_regression_1000.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/regression_metrics.json"))
    args = parser.parse_args()
    settings = load_settings()
    n = args.n or settings.evaluation.queries
    seed = args.seed or settings.evaluation.seed
    metrics = run(n=n, seed=seed, golden_path=args.golden)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics.model_dump(), indent=2), encoding="utf-8")
    print(json.dumps(metrics.model_dump(), indent=2))


if __name__ == "__main__":
    main()
