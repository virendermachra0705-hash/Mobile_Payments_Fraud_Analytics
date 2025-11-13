# fraud_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import datetime
import os

from model_loader import load_model_and_schema
from feature_builder import build_features_from_raw, append_api_log, save_history_row, load_history

# Load model & schema at import
model, MODEL_FEATURE_LIST = load_model_and_schema()

app = FastAPI(title="Mobile Fraud Detection - Real-time (Stateful)")

# Make sure no timestamp/is_fraud in schema
MODEL_FEATURE_LIST = [
    f for f in MODEL_FEATURE_LIST if f not in ("timestamp", "is_fraud")]

# ensure folders
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
    return {"status": "API Running Successfully 🚀", "model_features": len(MODEL_FEATURE_LIST)}


@app.get("/schema")
def schema():
    return {"feature_count": len(MODEL_FEATURE_LIST), "features": MODEL_FEATURE_LIST}


@app.post("/predict")
def predict(txn: RawTransaction):
    try:
        raw = txn.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build features (history-driven)
    feature_vector, enriched = build_features_from_raw(raw, MODEL_FEATURE_LIST)

    # Ensure numeric and correct shape/order
    feature_vector = feature_vector.apply(
        pd.to_numeric, errors="coerce").fillna(0)
    feature_vector = feature_vector.reindex(
        columns=MODEL_FEATURE_LIST, fill_value=0)

    # Convert to numpy array for model
    X_in = feature_vector.values

    try:
        prob = float(model.predict_proba(X_in)[:, 1][0])
    except Exception as e:
        # log some diagnostics
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Model prediction error: {e}")

    fraud_flag = bool(prob > 0.5)

    # log and history
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

    # update history record and save
    enriched["is_fraud"] = int(fraud_flag)
    enriched["timestamp"] = datetime.datetime.utcnow().isoformat()
    save_history_row(enriched)

    return {"fraud_probability": round(prob, 4), "fraud_flag": fraud_flag}
