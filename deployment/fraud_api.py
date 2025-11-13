# fraud_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import datetime
import os

from model_loader import load_model_and_schema
from feature_builder import build_features_from_raw, append_api_log, save_history_row, load_history

# Load model + schema at startup
model, MODEL_FEATURE_LIST, TRAINING_DF = load_model_and_schema()

app = FastAPI(title="Mobile Fraud Detection - Raw Input API")

# Create required directories
os.makedirs("../data/state", exist_ok=True)
os.makedirs("../reports", exist_ok=True)

# Pydantic schema for raw transaction (minimal)


class RawTransaction(BaseModel):
    transaction_id: str | None = None
    user_id: int | None = None
    merchant_id: int | None = None
    amount: float
    timestamp: str | None = None   # ISO string or parseable
    transaction_type: str | None = "purchase"
    device_type: str | None = "Android"
    location: str | None = "Unknown"


@app.get("/")
def root():
    return {"status": "API Running Successfully 🚀", "model_features": len(MODEL_FEATURE_LIST), "log_file": "../reports/api_logs.csv"}


@app.post("/predict")
def predict(txn: RawTransaction):
    # Validate minimal fields
    try:
        raw = txn.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build features using history + training schema
    feature_vector, enriched_for_history = build_features_from_raw(
        raw, MODEL_FEATURE_LIST, TRAINING_DF
    )

    # Ensure numeric
    feature_vector = feature_vector.apply(
        pd.to_numeric, errors="coerce").fillna(0)

    # ---------------------------------------------------------
    # FIX FEATURE MISMATCH (Remove extra columns)
    # ---------------------------------------------------------
    extra_cols = ["timestamp", "is_fraud"]
    for col in extra_cols:
        if col in feature_vector.columns:
            feature_vector.drop(columns=[col], inplace=True)

    # Keep ONLY model-required columns (exact order)
    feature_vector = feature_vector.reindex(
        columns=MODEL_FEATURE_LIST, fill_value=0)

    X_in = feature_vector  # now perfectly aligned
    # ---------------------------------------------------------

    # Predict
    try:
        prob = float(model.predict_proba(X_in)[:, 1][0])
    except:
        prob = float(model.predict_proba(X_in.values)[:, 1][0])

    fraud_flag = bool(prob > 0.5)

    # Prepare log entry
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
        "fraud_flag": fraud_flag,
    }

    # Full enriched record for history
    enriched_for_history_record = enriched_for_history.copy()
    enriched_for_history_record.update({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "is_fraud": int(fraud_flag),
    })

    # Save logs + history
    append_api_log(log_entry)
    save_history_row(enriched_for_history_record)

    return {"fraud_probability": round(prob, 4), "fraud_flag": fraud_flag}
