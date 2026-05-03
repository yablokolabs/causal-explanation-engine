from __future__ import annotations

from pathlib import Path
from typing import Protocol
import numpy as np

from core.synthetic import FEATURE_ORDER, generate_cre_dataset


class PredictionModel(Protocol):
    feature_order: list[str]
    model_version: str

    def predict_array(self, x: np.ndarray) -> np.ndarray: ...
    def predict_one(self, features: dict[str, float]) -> float: ...


class XGBoostCREModel:
    """Production adapter around XGBoost with deterministic sklearn fallback only if xgboost is unavailable."""

    def __init__(self, artifact_path: str = "artifacts/xgb_cre.json", feature_order: list[str] | None = None):
        self.artifact_path = Path(artifact_path)
        self.feature_order = feature_order or FEATURE_ORDER.copy()
        self.model_version = "xgboost-cre-v1"
        self._model = None
        self._backend = "xgboost"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def raw_model(self):
        return self._model

    def ensure_loaded(self) -> "XGBoostCREModel":
        if self._model is not None:
            return self
        if self.artifact_path.exists():
            self._load()
        else:
            self.train_and_save()
        return self

    def train_and_save(self, n: int = 2500, seed: int = 42) -> None:
        x, y, feature_order = generate_cre_dataset(n=n, seed=seed)
        self.feature_order = feature_order
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.045,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=1,
            )
            model.fit(x, y)
            model.save_model(str(self.artifact_path))
            self._backend = "xgboost"
            self._model = model
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor
            import pickle

            model = GradientBoostingRegressor(random_state=seed, n_estimators=180, max_depth=3, learning_rate=0.05)
            model.fit(x, y)
            with self.artifact_path.with_suffix(".pkl").open("wb") as fh:
                pickle.dump(model, fh)
            self._backend = "sklearn_gradient_boosting_fallback"
            self._model = model

    def _load(self) -> None:
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor()
            model.load_model(str(self.artifact_path))
            self._backend = "xgboost"
            self._model = model
        except Exception:
            import pickle

            with self.artifact_path.with_suffix(".pkl").open("rb") as fh:
                self._model = pickle.load(fh)
            self._backend = "sklearn_gradient_boosting_fallback"

    def vectorize(self, features: dict[str, float]) -> np.ndarray:
        return np.array([[float(features[name]) for name in self.feature_order]], dtype=float)

    def predict_array(self, x: np.ndarray) -> np.ndarray:
        self.ensure_loaded()
        return np.asarray(self._model.predict(x), dtype=float)

    def predict_one(self, features: dict[str, float]) -> float:
        return float(self.predict_array(self.vectorize(features))[0])
