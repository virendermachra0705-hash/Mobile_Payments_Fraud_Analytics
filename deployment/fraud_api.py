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
    try:
        feat_count = len(model.get_booster().feature_names)
    except Exception:
        feat_count = len(MODEL_FEATURE_LIST)
        return {"status": "API Running Successfully 🚀", "model_features": feat_count, "log_file": "../reports/api_logs.csv"}


@app.post("/predict")
def predict(txn: RawTransaction):

    try:
        raw = txn.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    feature_vector, enriched_for_history = build_features_from_raw(
        raw, MODEL_FEATURE_LIST, TRAINING_DF
    )

    feature_vector = feature_vector.apply(
        pd.to_numeric, errors="coerce").fillna(0)

    # Use model's actual feature names if available (most robust)
    try:
        model_feats = model.get_booster().feature_names
        if model_feats is None:
            # fallback to MODEL_FEATURE_LIST
            model_feats = MODEL_FEATURE_LIST
    except Exception:
        model_feats = MODEL_FEATURE_LIST

    # Align exactly to model features (order + count)
    feature_vector = feature_vector.reindex(
        columns=list(model_feats), fill_value=0)
    X_in = feature_vector

    # ---------------- Robust prediction ----------------
    try:
        proba_all = model.predict_proba(X_in)
        # find index of positive class (1) if available
        pos_idx = None
        if hasattr(model, "classes_") and model.classes_ is not None:
            try:
                pos_idx = list(model.classes_).index(1)
            except ValueError:
                # class '1' not present; fallback to last column
                pos_idx = proba_all.shape[1] - 1
        else:
            pos_idx = proba_all.shape[1] - 1

        prob = float(proba_all[:, pos_idx][0])
    except Exception:
        # last-resort try with numpy array input
        proba_all = model.predict_proba(X_in.values)
        pos_idx = proba_all.shape[1] - 1
        prob = float(proba_all[:, pos_idx][0])
    # ---------------------------------------------------

    # Use threshold 0.5 as before (you can tune later)
    FRAUD_THRESHOLD = 0.10   # 10%
    fraud_flag = bool(prob > FRAUD_THRESHOLD)

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

    enriched_record = enriched_for_history.copy()
    enriched_record.update({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "is_fraud": int(fraud_flag)
    })

    append_api_log(log_entry)
    save_history_row(enriched_record)

    return {"fraud_probability": round(prob, 4), "fraud_flag": fraud_flag}
