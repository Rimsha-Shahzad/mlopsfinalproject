# train.py  ← this file lives at: ~/fraud-detection-mlops/train.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/creditcard.csv")
print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} / {len(df)} ({df['Class'].mean()*100:.3f}%)")

# ── 2. Features and target ────────────────────────────────────────────────────
X = df.drop("Class", axis=1)
y = df["Class"]

# ── 3. Train/test split (stratified to preserve fraud ratio) ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Scale Amount and Time columns ──────────────────────────────────────────
scaler = StandardScaler()
X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
X_test[["Amount", "Time"]]  = scaler.transform(X_test[["Amount", "Time"]])

# Save scaler for use in API later
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")
print("Scaler saved to model/scaler.pkl")

# ── 5. Handle class imbalance with SMOTE ──────────────────────────────────────
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — fraud: {y_train_bal.sum()}, legit: {(y_train_bal==0).sum()}")

# ── 6. MLflow experiment tracking ─────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection")

params = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1
}

with mlflow.start_run(run_name="random-forest-v1"):

    # ── 7. Train model ─────────────────────────────────────────────────────────
    print("Training Random Forest...")
    model = RandomForestClassifier(**params)
    model.fit(X_train_bal, y_train_bal)

    # ── 8. Evaluate ────────────────────────────────────────────────────────────
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auc       = roc_auc_score(y_test, y_pred_prob)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)

    print(f"\n--- Results ---")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # ── 9. Log params and metrics to MLflow ───────────────────────────────────
    mlflow.log_params(params)
    mlflow.log_metric("roc_auc",   auc)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)

    # ── 10. Register model in MLflow Model Registry ───────────────────────────
    mlflow.sklearn.log_model(
        model,
        artifact_path="fraud_model",
        registered_model_name="FraudDetector"
    )
    print("\nModel registered in MLflow as 'FraudDetector'")

    # ── 11. Save model locally too ────────────────────────────────────────────
    joblib.dump(model, "model/fraud_model.pkl")
    print("Model saved locally to model/fraud_model.pkl")
