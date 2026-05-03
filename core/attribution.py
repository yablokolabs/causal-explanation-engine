from __future__ import annotations

import logging
import numpy as np

from core.schemas import AttributionItem, AttributionResult
from core.synthetic import generate_cre_dataset
from models.xgboost_model import XGBoostCREModel

logger = logging.getLogger(__name__)


class ShapAttributionLayer:
    """Computes SHAP values with deterministic ablation fallback.

    The fallback keeps the repo runnable in constrained environments while the
    normal path uses shap.TreeExplainer for XGBoost artifacts.
    """

    def __init__(self, model: XGBoostCREModel, top_k: int = 5, background_size: int = 128):
        self.model = model.ensure_loaded()
        self.top_k = top_k
        bg, _, _ = generate_cre_dataset(n=background_size, seed=2025)
        self.background = bg
        self.baseline = np.mean(bg, axis=0)
        self._explainer = None
        try:
            import shap

            self._explainer = shap.TreeExplainer(self.model.raw_model)
        except Exception as exc:
            logger.warning("shap explainer unavailable; ablation fallback will be used", extra={"extra_fields": {"error": str(exc)}})

    def compute(self, features: dict[str, float], top_k: int | None = None) -> AttributionResult:
        k = top_k or self.top_k
        x = self.model.vectorize(features)
        prediction = float(self.model.predict_array(x)[0])
        base_value, contribs, method = self._shap_values(x)
        residual = prediction - (base_value + float(np.sum(contribs)))
        items = self._items(features, contribs, k)
        return AttributionResult(
            method=method,
            base_value=float(base_value),
            prediction=prediction,
            residual=float(residual),
            top_k=items[:k],
            all_attributions=items,
        )

    def _shap_values(self, x: np.ndarray) -> tuple[float, np.ndarray, str]:
        try:
            if self._explainer is None:
                raise RuntimeError("SHAP explainer is not initialized")
            explainer = self._explainer
            values = explainer.shap_values(x)
            if isinstance(values, list):
                values = values[0]
            contribs = np.asarray(values, dtype=float)[0]
            expected = explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                expected = float(np.asarray(expected).ravel()[0])
            return float(expected), contribs, "shap.TreeExplainer"
        except Exception as exc:
            logger.warning("shap unavailable; using deterministic ablation fallback", extra={"extra_fields": {"error": str(exc)}})
            return self._ablation_values(x)

    def _ablation_values(self, x: np.ndarray) -> tuple[float, np.ndarray, str]:
        baseline_x = self.baseline.reshape(1, -1)
        base_value = float(self.model.predict_array(baseline_x)[0])
        prediction = float(self.model.predict_array(x)[0])
        raw = []
        for idx in range(x.shape[1]):
            counterfactual = x.copy()
            counterfactual[0, idx] = self.baseline[idx]
            raw.append(prediction - float(self.model.predict_array(counterfactual)[0]))
        contribs = np.asarray(raw, dtype=float)
        total = float(np.sum(contribs))
        target_total = prediction - base_value
        if abs(total) > 1e-12:
            contribs = contribs * (target_total / total)
        return base_value, contribs, "deterministic_ablation_shap_fallback"

    def _items(self, features: dict[str, float], contribs: np.ndarray, k: int) -> list[AttributionItem]:
        abs_total = float(np.sum(np.abs(contribs))) or 1.0
        order = sorted(range(len(contribs)), key=lambda i: abs(float(contribs[i])), reverse=True)
        ranked: list[AttributionItem] = []
        for rank, idx in enumerate(order, start=1):
            feature = self.model.feature_order[idx]
            c = float(contribs[idx])
            if c > 1e-9:
                direction = "increases"
            elif c < -1e-9:
                direction = "decreases"
            else:
                direction = "neutral"
            ranked.append(
                AttributionItem(
                    feature=feature,
                    value=features[feature],
                    baseline_value=float(self.baseline[idx]),
                    contribution=c,
                    normalized_weight=abs(c) / abs_total,
                    direction=direction,
                    rank=rank,
                )
            )
        return ranked
