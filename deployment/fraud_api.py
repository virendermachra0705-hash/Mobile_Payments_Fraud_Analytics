# fraud_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import datetime
import os

from model_loader import load_model_and_schema
from feature_builder import (
    build_features_from_raw,
    append_api_log,
    save_history_row,
    load_history
)

# Load model, schema, training_df
model, MODEL_FEATURE_LIST, TRAINING_DF = load_model_and_schema()

app = FastAPI(title="Mobile Fraud Detection - Raw Input API")

# Ensure directories exist
os.makedirs("../data/state", exist_ok=True)
os.makedirs("../reports", exist_ok=True)


# ---------------------------
#  INPUT SCHEMA
# ---------------------------
class RawTransaction(BaseModel):
    transaction_id: str | None = None
    user_id: int | None = None
    merchant_id: int | None = None
    amount: float
    timestamp: str | None = None
    transaction_type: str | None = "purchase"
    device_type: str | None = "Android"
    location: str | None = "Unknown"


# ---------------------------
#  ROOT ENDPOINT
# ---------------------------
@app.get("/")
def root():
    return {
        "status": "API Running Successfully 🚀",
        "model_features": len(MODEL_FEATURE_LIST),
        "schema_endpoint": "/schema"
    }


# ---------------------------
#  FEATURE SCHEMA PREVIEW
# ---------------------------
@app.get("/schema")
def get_schema():
    return {
        "feature_count": len(MODEL_FEATURE_LIST),
        "features": MODEL_FEATURE_LIST
    }


# ---------------------------
#  PREDICTION ENDPOINT
# ---------------------------
@app.post("/predict")
def predict(txn: RawTransaction):

    # convert input to dict
    try:
        raw = txn.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Build features (safe + stable)
    try:
        feature_vector, enriched_for_history = build_features_from_raw(
            raw, MODEL_FEATURE_LIST
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature error: {e}")

    # Convert to numeric (ensure no dtype issues)
    feature_vector = feature_vector.apply(
        pd.to_numeric, errors="coerce").fillna(0)

    # Ensure correct shape
    try:
        X_in = feature_vector[MODEL_FEATURE_LIST]
    except Exception:
        # fallback fill for missing columns
        for feat in MODEL_FEATURE_LIST:
            if feat not in feature_vector.columns:
                feature_vector[feat] = 0
        X_in = feature_vector[MODEL_FEATURE_LIST]

    # -----------------
    # MODEL PREDICTION
    # -----------------
    try:
        prob = float(model.predict_proba(X_in.values)[:, 1][0])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model prediction Error: {e}")

    fraud_flag = bool(prob > 0.5)

    # -----------------
    # LOGGING + HISTORY
    # -----------------
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "transaction_id": raw.get("transaction_id"),
        "user_id": raw.get("user_id"),
        "merchant_id": raw.get("merchant_id"),
        "amount": raw.get("amount"),
        "transaction_type": raw.get("transaction_type"),
        "device_type": raw.get("device_type"),
        "location": raw.get("location"),
        "fraud_probability": round(prob, 4),
        "fraud_flag": fraud_flag
    }

    # enriched state row (store predicted flag)
    enriched_for_history["timestamp"] = datetime.datetime.utcnow().isoformat()
    enriched_for_history["is_fraud"] = int(fraud_flag)

    append_api_log(log_entry)
    save_history_row(enriched_for_history)

    return {
        "fraud_probability": round(prob, 4),
        "fraud_flag": fraud_flag
    }
