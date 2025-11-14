# model_loader.py (example)
import joblib
import os
import json
import pandas as pd
MODEL_PATH = "../models/xgboost_tuned.pkl"
FEATURES_JSON = "../models/xgboost_tuned_schema.json"  # optional
TRAINING_CSV = "../data/processed/features_enriched_v2.csv"


def load_model_and_schema():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    model_feature_list = None
    try:
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            model_feature_list = list(booster.feature_names)
    except Exception:
        pass

    if model_feature_list is None and os.path.exists(FEATURES_JSON):
        with open(FEATURES_JSON) as f:
            j = json.load(f)
            model_feature_list = j.get("feature_list")

    if model_feature_list is None:
        raise RuntimeError("Cannot determine feature list")

    # remove time/target if present
    model_feature_list = [
        f for f in model_feature_list if f not in ("timestamp", "is_fraud")]

    TRAINING_DF = pd.read_csv(TRAINING_CSV) if os.path.exists(
        TRAINING_CSV) else None

    return model, model_feature_list, TRAINING_DF
