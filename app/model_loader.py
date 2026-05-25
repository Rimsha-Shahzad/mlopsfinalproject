# app/model_loader.py  ← ~/fraud-detection-mlops/app/model_loader.py

import mlflow.sklearn
import joblib
import os

_model  = None
_scaler = None

def get_model():
    global _model
    if _model is None:
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            _model = mlflow.sklearn.load_model("models:/FraudDetector/Production")
            print("Model loaded from MLflow registry")
        except Exception:
            # Fallback: load from local file
            _model = joblib.load("model/fraud_model.pkl")
            print("Model loaded from local file")
    return _model

def get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load("model/scaler.pkl")
    return _scaler
