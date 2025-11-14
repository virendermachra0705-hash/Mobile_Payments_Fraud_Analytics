# fraud_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import datetime
import os
import traceback

from model_loader import load_model_and_schema
from feature_builder import build_features_from_raw, append_api_log, save_history_row, load_history

# Load model + schema at startup. Support both return shapes (2-tuple or 3-tuple)
try:
    res = load_model_and_schema()
    if isinstance(res, tuple) and len(res) == 3:
        model, MODEL_FEATURE_LIST, TRAINING_DF = res
    elif isinstance(res, tuple) and len(res) == 2:
        model, MODEL_FEATURE_LIST = res
        TRAINING_DF = None
    else:
        raise RuntimeError("load_model_and_schema returned unexpected shape")
except Exception as e:
    # fail-fast on startup
    raise

app = FastAPI(title="Mobile Fraud Detection - Raw Input API")

# Create required directories
os.makedirs("../data/state", exist_ok=True)
os.makedirs("../reports", exist_ok=True)

# Threshold: tune per business (default 3%)
THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", 0.03))


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
    return {"status": "API Running Successfully 🚀", "model_features": len(MODEL_FEATURE_LIST), "log_file": "../reports/api_logs.csv"}


@app.get("/schema")
def schema():
    return {"feature_count": len(MODEL_FEATURE_LIST), "features": MODEL_FEATURE_LIST}


@app.post("/predict")
def predict(txn: RawTransaction):
    try:
        raw = txn.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        feature_vector, enriched_for_history = build_features_from_raw(raw, MODEL_FEATURE_LIST, TRAINING_DF)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=422, detail=f"Feature error: {str(e)}\n{tb}")

    # numeric alignment
    feature_vector = feature_vector.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Make sure columns strictly match model feature list
    for c in MODEL_FEATURE_LIST:
        if c not in feature_vector.columns:
            feature_vector[c] = 0
    feature_vector = feature_vector[MODEL_FEATURE_LIST]

    # Predict
    try:
        # model.predict_proba may accept dataframe or numpy
        probs = model.predict_proba(feature_vector)[:, 1]
    except Exception:
        probs = model.predict_proba(feature_vector.values)[:, 1]

    prob = float(probs[0])
    fraud_flag = bool(prob >= THRESHOLD)

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
        "fraud_probability": round(prob, 6),
        "fraud_flag": fraud_flag
    }

    # Save enriched history row (append predicted flag)
    enriched_record = enriched_for_history.copy()
    enriched_record.update({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "is_fraud": int(fraud_flag)
    })

    try:
        append_api_log(log_entry)
        save_history_row(enriched_record)
    except Exception:
        # don't fail prediction if logging fails
        pass

    return {"fraud_probability": round(prob, 6), "fraud_flag": fraud_flag}


@app.post("/debug")
def debug(txn: RawTransaction):
    """
    Return feature vector and model probability for one transaction for debugging.
    DO NOT expose this in public production without auth.
    """
    try:
        raw = txn.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        feature_vector, enriched_for_history = build_features_from_raw(raw, MODEL_FEATURE_LIST, TRAINING_DF)
        feature_vector = feature_vector.apply(pd.to_numeric, errors="coerce").fillna(0)
        for c in MODEL_FEATURE_LIST:
            if c not in feature_vector.columns:
                feature_vector[c] = 0
        feature_vector = feature_vector[MODEL_FEATURE_LIST]
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=422, detail=f"Feature error: {str(e)}\n{tb}")

    try:
        probs = model.predict_proba(feature_vector)[:, 1]
    except Exception:
        probs = model.predict_proba(feature_vector.values)[:, 1]

    prob = float(probs[0])
    top_feats = feature_vector.iloc[0].sort_values(ascending=False).head(30).to_dict()

    return {
        "probability": prob,
        "threshold": THRESHOLD,
        "fraud_flag": bool(prob >= THRESHOLD),
        "top_features": top_feats,
        "feature_vector": feature_vector.iloc[0].to_dict(),
        "enriched_for_history": enriched_for_history
    }
