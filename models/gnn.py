from __future__ import annotations


class GNNPlaceholderModel:
    """Contract-preserving placeholder for graph neural network models.

    The production seam is intentional: a real GNN only needs to implement
    predict_one(features) and expose feature_order/model_version. The API rejects
    this backend until an artifact is configured, avoiding fake predictions.
    """

    feature_order: list[str] = []
    model_version = "gnn-placeholder-unconfigured"

    def predict_one(self, features: dict[str, float]) -> float:
        raise NotImplementedError("GNN backend is configured but no production artifact adapter is installed.")
