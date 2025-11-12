

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import csv
import datetime
from preprocess_pipeline import preprocess_transaction
from model_loader import load_model

app = FastAPI(
    title="Mobile Payment Fraud Detection API",
    description="Real-time fraud scoring service for mobile payment transactions.",
    version="1.3"
)

model, booster = load_model("../models/xgboost_tuned.pkl")
FEATURES = booster.feature_names

class Transaction(BaseModel):
    amount: float = 0.0
    hour: int = 0
    day_of_week: str = "Monday"
    device_type: str = "Android"
    transaction_type: str = "purchase"
    user_txn_count: float = 0.0
    user_avg_amount: float = 0.0
    user_std_amount: float = 0.0
    rolling_fraud_rate_user_7d: float = 0.0
    user_fraud_rate: float = 0.0
    amount_zscore_user: float = 0.0
    merchant_fraud_rate: float = 0.0
    peer_spend_ratio: float = 0.0
    weighted_amount: float = 0.0

@app.get("/")
def home():
    return {
        "status": "API Running Successfully 🚀",
        "model_features": len(FEATURES),
        "log_file": "../reports/api_logs.csv"
    }

@app.post("/predict")
def predict(txn: Transaction):
    txn_dict = txn.dict()

    # --- Step 1: Preprocess Input ---
    data = preprocess_transaction(txn_dict, FEATURES)

    # --- Step 2: Model Inference ---
    prob = model.predict_proba(data)[:, 1][0]
    fraud_flag = bool(prob > 0.5)

    # --- Step 3: Logging ---
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        **txn_dict,
        "fraud_probability": round(float(prob), 4),
        "fraud_flag": fraud_flag
    }

    log_dir = "../reports"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "api_logs.csv")

    file_exists = os.path.exists(log_path)
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    # --- Step 4: Response ---
    return {
        "fraud_probability": round(float(prob), 4),
        "fraud_flag": fraud_flag,
        "message": "Prediction logged successfully ✅"
    }

@app.get("/health")
def health():
    return {"status": "Healthy", "timestamp": datetime.datetime.now().isoformat()}
