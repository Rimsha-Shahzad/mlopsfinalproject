# app/main.py  ← ~/fraud-detection-mlops/app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import pandas as pd
import numpy as np
import time
import mlflow

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from app.model_loader import get_model, get_scaler

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using Random Forest + MLOps",
    version="1.0.0"
)

# ── Prometheus metrics ────────────────────────────────────────────────────────
FRAUD_COUNT  = Counter("fraud_predictions_total",      "Total fraud predictions")
LEGIT_COUNT  = Counter("legit_predictions_total",      "Total legitimate predictions")
REQUEST_COUNT= Counter("api_requests_total",           "Total API requests")
LATENCY      = Histogram("prediction_latency_seconds", "Prediction latency in seconds")
FRAUD_GAUGE  = Gauge("fraud_rate_current",             "Current fraud detection rate")

_total_preds = 0
_fraud_preds = 0

# ── Request schema ─────────────────────────────────────────────────────────────
class Transaction(BaseModel):
    """
    Send all 30 features from the creditcard.csv dataset.
    Features: Time, V1-V28, Amount  (in that order)
    """
    features: List[float]

    @validator("features")
    def check_length(cls, v):
        if len(v) != 30:
            raise ValueError(f"Expected 30 features, got {len(v)}")
        return v

class PredictionResponse(BaseModel):
    prediction:  str
    confidence:  float
    fraud_score: float
    latency_ms:  float

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Fraud Detection API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "FraudDetector", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    global _total_preds, _fraud_preds

    REQUEST_COUNT.inc()
    start = time.time()

    try:
        model  = get_model()
        scaler = get_scaler()

        # Build DataFrame with correct column names
        columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = pd.DataFrame([transaction.features], columns=columns)

        # Scale Amount and Time same way as training
        df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

        # Predict
        pred        = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]
        fraud_score = float(probabilities[1])

        # Update counters
        _total_preds += 1
        if pred == 1:
            _fraud_preds += 1
            FRAUD_COUNT.inc()
        else:
            LEGIT_COUNT.inc()

        # Update fraud rate gauge
        FRAUD_GAUGE.set(_fraud_preds / _total_preds)

        elapsed_ms = (time.time() - start) * 1000
        LATENCY.observe(elapsed_ms / 1000)

        return PredictionResponse(
            prediction  = "FRAUD" if pred == 1 else "LEGITIMATE",
            confidence  = round(float(max(probabilities)), 4),
            fraud_score = round(fraud_score, 4),
            latency_ms  = round(elapsed_ms, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/stats")
def stats():
    return {
        "total_predictions": _total_preds,
        "fraud_detected":    _fraud_preds,
        "legitimate":        _total_preds - _fraud_preds,
        "fraud_rate":        round(_fraud_preds / _total_preds, 4) if _total_preds > 0 else 0
    }
