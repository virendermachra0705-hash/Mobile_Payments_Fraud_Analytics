# fraud_api.py (Option A production-ready)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import datetime
import os

from model_loader import load_model_and_schema
from feature_builder import build_features_from_raw, append_api_log, save_history_row, load_history

# Load model + schema
model, MODEL_FEATURE_LIST = load_model_and_schema()

# Ensure we never include target/timestamp in schema
MODEL_FEATURE_LIST = [
    f for f in MODEL_FEATURE_LIST if f not in ("timestamp", "is_fraud")]

app = FastAPI(title="Mobile Fraud Detection - Real-time (History-driven)")

# ensure directories exist
os.makedirs("../data/state", exist_ok=True)
os.makedirs("../reports", exist_ok=True)
os.makedirs("../models", exist_ok=True)


class RawTransaction(BaseModel):
    transaction_id: str | None = None
    user_id: int | None = None
    merchant_id: int | None = None
    amount: float
    timestamp: str | None = None
    transaction_type: str | None = "purchase"
    device_type: str | None = "Android"
    location: str | None = "Unknown"


@app.get("/")
def root():
    return {
        "status": "API Running Successfully 🚀",
        "model_features": len(MODEL_FEATURE_LIST),
        "schema_preview": MODEL_FEATURE_LIST[:10]
    }


@app.post("/predict")
def predict(txn: RawTransaction):
    try:
        raw = txn.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Build feature vector from history only
    feature_vector, enriched = build_features_from_raw(raw, MODEL_FEATURE_LIST)

    # Force numeric and fillna
    feature_vector = feature_vector.apply(
        pd.to_numeric, errors="coerce").fillna(0)

    # Ensure exact ordering and presence
    feature_vector = feature_vector.reindex(
        columns=MODEL_FEATURE_LIST, fill_value=0)

    # Convert to numpy for XGBoost
    X_in = feature_vector.values

    # Predict probability
    try:
        prob = float(model.predict_proba(X_in)[:, 1][0])
    except Exception as e:
        # provide a helpful log message
        raise HTTPException(
            status_code=500, detail=f"Model predict error: {e}")

    fraud_flag = bool(prob > 0.5)

    # Prepare API log
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "transaction_id": raw.get("transaction_id"),
        "user_id": raw.get("user_id"),
        "merchant_id": raw.get("merchant_id"),
        "amount": raw.get("amount"),
        "device_type": raw.get("device_type"),
        "transaction_type": raw.get("transaction_type"),
        "location": raw.get("location"),
        "fraud_probability": round(prob, 4),
        "fraud_flag": int(fraud_flag)
    }
    append_api_log(log_entry)

    # Save enriched history (update is_fraud with predicted flag)
    enriched["is_fraud"] = int(fraud_flag)
    # timestamp: ensure ISO string (already provided by feature_builder) but override to now for consistent ordering
    enriched["timestamp"] = datetime.datetime.utcnow().isoformat()
    save_history_row(enriched)

    return {"fraud_probability": round(prob, 4), "fraud_flag": fraud_flag}
