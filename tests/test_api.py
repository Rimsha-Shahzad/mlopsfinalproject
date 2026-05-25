# tests/test_api.py  ← ~/fraud-detection-mlops/tests/test_api.py

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_valid():
    payload = {"features": [0.0]*30}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] in ["FRAUD", "LEGITIMATE"]

def test_predict_wrong_features():
    payload = {"features": [1.0, 2.0]}   # only 2 features — should fail
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
