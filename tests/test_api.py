from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from core.synthetic import feature_dict, generate_cre_dataset


def payload() -> dict:
    x, _, order = generate_cre_dataset(n=1, seed=321)
    return {"features": feature_dict(x[0], order), "trace_id": "api-test"}


def test_api_predict_explain_validate() -> None:
    with TestClient(app) as client:
        r = client.post("/predict", json=payload())
        assert r.status_code == 200, r.text
        prediction = r.json()["prediction"]

        explain_payload = payload()
        explain_payload["prediction"] = prediction
        r = client.post("/explain", json=explain_payload)
        assert r.status_code == 200, r.text
        explanation = r.json()
        assert "explanation_text" in explanation

        r = client.post("/validate", json={"explanation": explanation})
        assert r.status_code == 200, r.text
        assert r.json()["passed"] is True


def test_health() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_predict_gnn_returns_501() -> None:
    with TestClient(app) as client:
        p = payload()
        p["model_type"] = "gnn"
        r = client.post("/predict", json=p)
        assert r.status_code == 501


def test_explain_gnn_returns_501() -> None:
    with TestClient(app) as client:
        p = payload()
        p["model_type"] = "gnn"
        r = client.post("/explain", json=p)
        assert r.status_code == 501


def test_predict_invalid_features_returns_422() -> None:
    with TestClient(app) as client:
        r = client.post("/predict", json={"features": {"cap_rate": 0.05}})
        assert r.status_code == 422


def test_predict_extra_field_rejected() -> None:
    with TestClient(app) as client:
        p = payload()
        p["features"]["unknown_field"] = 99
        r = client.post("/predict", json=p)
        assert r.status_code == 422
