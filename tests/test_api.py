from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from core.synthetic import generate_cre_dataset, feature_dict


def payload() -> dict:
    x, _, order = generate_cre_dataset(n=1, seed=321)
    return {"features": feature_dict(x[0], order), "trace_id": "api-test"}


def test_api_predict_explain_validate() -> None:
    client = TestClient(app)
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
