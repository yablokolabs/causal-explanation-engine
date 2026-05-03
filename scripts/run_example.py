from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_settings
from core.logging import configure_logging
from core.pipeline import CausalExplanationEngine
from core.schemas import CREFeatures, ExplainRequest, PredictionRequest
from core.synthetic import generate_cre_dataset, feature_dict


def main() -> None:
    settings = load_settings()
    configure_logging(settings.app.log_level)
    engine = CausalExplanationEngine(settings)
    x, _, order = generate_cre_dataset(n=1, seed=123)
    features = CREFeatures.model_validate(feature_dict(x[0], order))
    prediction = engine.predict(PredictionRequest(features=features, trace_id="example-run"))
    explanation = engine.explain(ExplainRequest(features=features, prediction=prediction.prediction, trace_id="example-run"))
    validation = engine.validate(explanation)
    print(json.dumps({
        "prediction": prediction.model_dump(),
        "explanation_text": explanation.explanation_text,
        "validation": validation.model_dump(),
    }, indent=2))


if __name__ == "__main__":
    main()
