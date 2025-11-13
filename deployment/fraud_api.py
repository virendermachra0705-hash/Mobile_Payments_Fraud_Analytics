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
        raw, MODEL_FEATURE_LIST, TRAINING_DF)

    # Align dtypes and ensure numeric
    feature_vector = feature_vector.apply(
        pd.to_numeric, errors="coerce").fillna(0)

    # Final alignment to model's feature names (safety)
    try:
        # If model has booster and feature_names, use them. Else use MODEL_FEATURE_LIST from training.
        try:
            booster = model.get_booster()
            # assign booster.feature_names to model_feature_list if mismatch
            if booster.feature_names is None or list(booster.feature_names) != MODEL_FEATURE_LIST:
                booster.feature_names = MODEL_FEATURE_LIST
        except Exception:
            pass
        drop_cols = ["timestamp", "is_fraud"]
        for c in drop_cols:
            if c in feature_vector.columns:
                feature_vector = feature_vector.drop(columns=[c])

        X_in = feature_vector[MODEL_FEATURE_LIST]
    except Exception as e:
        # fallback: ensure all columns present
        for c in MODEL_FEATURE_LIST:
            if c not in feature_vector.columns:
                feature_vector[c] = 0
        X_in = feature_vector[MODEL_FEATURE_LIST]

    # Predict
    try:
        prob = float(model.predict_proba(X_in)[:, 1][0])
    except Exception as e:
        # Try converting to numpy array if model wrapper expects it
        prob = float(model.predict_proba(X_in.values)[:, 1][0])

    fraud_flag = bool(prob > 0.5)

    # Prepare log entry (flat)
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
        "fraud_flag": fraud_flag
    }

    # Add enriched fields to history row (for full state)
    enriched_for_history_record = enriched_for_history.copy()
    enriched_for_history_record.update({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        # store predicted flag; if you have ground truth later, you can update
        "is_fraud": int(fraud_flag)
    })

    # Append to logs and history
    append_api_log(log_entry)
    # save full enriched record into history (so subsequent transactions see it)
    save_history_row(enriched_for_history_record)

    # Return minimal response
    return {"fraud_probability": round(prob, 4), "fraud_flag": fraud_flag}
